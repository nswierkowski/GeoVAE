import math
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vae.vae import VAE
from src.models.treevi.TreeStructure import TreeStructure
from src.models.treevi.TreeOptimizer import TreeOptimizer


class TreeVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        residual_hiddens: int = 64,
        num_residual_layers: int = 2,
        latent_dim: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        image_size: int = 64,
        tree_optimizer_config: Optional[dict] = None,
    ):
        super().__init__()

        self.device = device
        self.latent_dim = latent_dim

        self.vanilla_vae = VAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            residual_hiddens=residual_hiddens,
            num_residual_layers=num_residual_layers,
            latent_dim=latent_dim,
            device=device,
            image_size=image_size,
        )

        self.tree_optimizer = TreeOptimizer(tree_optimizer_config or {})
        self.current_tree: Optional[TreeStructure] = None

    # -------------------------
    # Optimized: build chain tree using tensor ops
    # -------------------------
    def _init_chain_tree(self, num_nodes: int) -> TreeStructure:
        device = torch.device(self.device)

        # adjacency as upper + lower band
        A = torch.eye(num_nodes, device=device)
        A[:-1, 1:] += torch.eye(num_nodes - 1, device=device)
        A[1:, :-1] += torch.eye(num_nodes - 1, device=device)
        A = (A > 0).float()

        # vectorized edge list [(i, i+1)]
        idx = torch.arange(num_nodes - 1, device=device)
        edge_list = [(int(i), int(i) + 1) for i in idx.tolist()]

        # one gamma tensor shared across all edges
        gamma_val = 0.5 * torch.ones(self.latent_dim, device=device)
        gamma_dict = {(int(i), int(i) + 1): gamma_val.clone() for i in idx.tolist()}

        # keep adj_matrix on GPU (TreeStructure can handle it)
        return TreeStructure(adj_matrix=A, edge_list=edge_list, gamma_dict=gamma_dict)

    # -------------------------
    # Encode/decode passthrough
    # -------------------------
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.vanilla_vae.encode(x)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vanilla_vae.decode(z)

    # -------------------------
    # Optimized tree reparameterization
    # -------------------------
    def tree_reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor, tree: TreeStructure) -> torch.Tensor:
        B, D = mu.size()
        device = mu.device

        if tree is None:
            return mu + sigma * torch.randn_like(sigma)

        ordering = tree.get_ordering()
        if len(ordering) != B:
            return mu + sigma * torch.randn_like(sigma)

        e = torch.randn((B, D), device=device)
        z = torch.zeros_like(mu, device=device)

        # Precompute paths and correlations to reduce Python calls
        parent_map = {i: tree.get_parent(i) for i in ordering}
        path_map = {i: tree.get_path(i) for i in ordering}

        for node in ordering:
            p = parent_map[node]
            if p is None:
                z[node] = mu[node] + sigma[node] * e[node]
            else:
                path = path_map[node]
                root = path[0]
                eff_root_corr = tree.compute_effective_correlation(root, node).to(device)

                # vectorized contrib from root
                contrib = eff_root_corr * e[root] * sigma[node]

                # collect all ancestors except root and node
                if len(path) > 1:
                    ancestors = path[1:-1]
                    if ancestors:
                        # vectorized gamma/scaling computation
                        for ancestor in ancestors:
                            eff_anc = tree.compute_effective_correlation(ancestor, node).to(device)
                            parent_of_ancestor = parent_map[ancestor]
                            if parent_of_ancestor is None:
                                gamma_val = torch.ones(D, device=device)
                            else:
                                gamma_val = tree.gamma.get((parent_of_ancestor, ancestor),
                                                           torch.ones(D, device=device)).to(device)
                            scaling = torch.sqrt(torch.clamp(1.0 - gamma_val.pow(2), min=1e-8))
                            contrib = contrib + eff_anc * scaling * e[ancestor] * sigma[node]

                z[node] = mu[node] + contrib

        return z

    # -------------------------
    # Tree update (unchanged but safe)
    # -------------------------
    def update_tree(self, mu: torch.Tensor, sigma: torch.Tensor, loss_term: Optional[torch.Tensor] = None):
        B = mu.size(0)
        init_tree = self._init_chain_tree(B)

        try:
            new_tree = self.tree_optimizer.optimize_tree(init_tree)
        except TypeError:
            try:
                new_tree = self.tree_optimizer.optimize_tree(init_tree, embeddings=None)
            except TypeError:
                try:
                    new_tree = self.tree_optimizer.optimize_tree(
                        init_tree, num_nodes=B,
                        loss_term=(loss_term.detach() if loss_term is not None else None)
                    )
                except Exception:
                    new_tree = init_tree

        self.current_tree = new_tree

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, sigma = self.encode(x)
        B = mu.size(0)

        if (self.current_tree is None) or (len(self.current_tree.get_ordering()) != B):
            self.current_tree = self._init_chain_tree(B)

        z = self.tree_reparameterize(mu, sigma, self.current_tree)
        recon = self.decode(z)

        log_px_z = -F.mse_loss(recon, x, reduction="sum")
        prior_log_prob = -0.5 * torch.sum(z.pow(2) + math.log(2 * math.pi))
        entropy = 0.5 * torch.sum(1 + torch.log(sigma.pow(2) + 1e-8))
        elbo = log_px_z + prior_log_prob + entropy

        logvar = torch.log(sigma.pow(2) + 1e-8)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        mean_kl_per_dim = kl_per_dim.mean(dim=0)
        num_active_dims = (mean_kl_per_dim > 0.01).sum().item()

        try:
            self.update_tree(mu, sigma, loss_term=elbo.detach())
        except Exception:
            pass

        return {
            "recon": recon,
            "elbo": elbo,
            "partial_loss": -prior_log_prob - entropy,
            "num_active_dims": num_active_dims,
            "tree": self.current_tree,
        }
