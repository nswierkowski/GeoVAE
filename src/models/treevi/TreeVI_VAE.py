import math
from typing import Dict, Optional, Tuple
from torch import nn
import torch
import torch.nn.functional as F
from src.models.treevi.tree_variational_interface.tree_vi_structure.tree_vi import (
    TreeStructure,
)
from src.models.treevi.tree_variational_interface.vi_structure import VIStructure
from src.models.vae.vae import VAE
from src.models.treevi.tree_variational_interface.vi_optimizer import TreeOptimizer


def _compute_cholesky_tensor(
    batch_size: int,
    latent_dim: int,
    parent: torch.LongTensor,
    gamma_dict: Dict[Tuple[int, int], torch.Tensor],
    device: torch.device,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Returns tensor L of shape [B, B, D] such that:
    - For each d=0..D-1, L[:,:,d] is the lower triangular Cholesky matrix
      corresponding to the covariance of latents in the tree defined by parent/gamma_dict.
    Args:
        batch_size (int): Number of nodes in the tree.
        latent_dim (int): Dimensionality of the latent space.
        parent (torch.LongTensor): Parent indices for each node in the tree, shape [B].
            parent[i] = -1 if i is the root node.
        gamma_dict (Dict[Tuple[int,int], torch.Tensor]): Dictionary mapping edges (parent, child)
            to their correlation vectors (shape [latent_dim]).
        device (torch.device): Device to perform computations on (e.g., "cuda" or "cpu").
    """
    B, D = batch_size, latent_dim
    L = torch.zeros(B, B, D, device=device)

    root_mask = parent == -1
    non_root_mask = ~root_mask
    root_indices = root_mask.nonzero(as_tuple=False).view(-1)
    p_idx = parent.clone()
    p_idx[root_mask] = 0

    gamma_tensor = torch.zeros(B, D, device=device)
    for i in range(B):
        p = parent[i].item()
        if p != -1:
            gamma_tensor[i] = gamma_dict[(p, i)]

    for d in range(D):
        Lambda = L[..., d]
        gamma_d = gamma_tensor[:, d]
        parent_rows = Lambda[p_idx, :]
        Lambda[:, :] = parent_rows * gamma_d.view(B, 1)

        Lambda[root_indices, root_indices] = 1.0

    sumsq = torch.sum(L**2, dim=1)
    under = torch.clamp(1.0 - sumsq, min=eps)
    L[torch.arange(B), torch.arange(B), :] = torch.sqrt(under)

    return torch.nan_to_num(L, nan=eps * 1e-2)


class VAEWithVIStructure(nn.Module):
    """
    VAETreeVI implements the variational autoencoder (VAE) with tree-structured reparameterization support.

    Methods:
        __init__(input_dim: int, hidden_dim: int, residual_hiddens: int, num_residual_layers: int, latent_dim: int, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        encode(x: Tensor) -> Tuple[Tensor, Tensor]: Returns the mean and standard deviation.
        decode(z: Tensor) -> Tensor: Decodes latent variable z into the reconstruction.
        tree_reparameterize(mu: Tensor, sigma: Tensor, tree: TreeStructure) -> Tensor:
            Implements the tree‐structured reparameterization procedure.
        forward(x: Tensor, tree: Optional[TreeStructure] = None) -> Tuple[Tensor, Tensor, Tensor]:
            Performs a forward pass using either tree‐structured or standard reparameterization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        residual_hiddens: int,
        num_residual_layers: int,
        latent_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        faster_version=True,
        vi_optimizer=None,
    ) -> None:
        super(VAEWithVIStructure, self).__init__()

        self.vanilla_vae = VAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            residual_hiddens=residual_hiddens,
            num_residual_layers=num_residual_layers,
            latent_dim=latent_dim,
            device=device,
        )
        self.device = device
        self.to(device)

        self.latent_dim = latent_dim
        self.current_vi_structure: Optional[VIStructure] = None
        self.current_elbo: Optional[torch.Tensor] = None

        self.faster_version = faster_version

        self.vi_optimizer = (
            vi_optimizer
            if vi_optimizer is not None
            else TreeOptimizer(latent_dim, {}, loss_callback=self.loss_callback)
        )

    def loss_callback(self, A: torch.Tensor) -> torch.Tensor:
        return self.current_elbo

    def _init_chain_tree(self, num_nodes: int) -> TreeStructure:
        """
        Initializes a simple chain tree structure with num_nodes nodes and latent_dim latent dimensions.
        This creates a linear chain where each node is connected to the next one, forming a simple tree structure.
        The adjacency matrix A is initialized to zeros, and edges are added between consecutive nodes.
        The gamma parameters for each edge are initialized to 0.5 for each latent dimension.

        Args:
            num_nodes (int): Number of nodes in the chain tree.

        Returns:
            TreeStructure: An instance of TreeStructure representing the chain tree.
        """
        A = torch.zeros((num_nodes, num_nodes), device=self.device)
        idx = torch.arange(num_nodes - 1, device=self.device)
        A[idx, idx + 1] = 1.0
        A[idx + 1, idx] = 1.0

        edge_list = [(i, i + 1) for i in range(num_nodes - 1)]

        gamma_tensor = 0.5 * torch.ones(self.latent_dim, device=self.device)
        gamma_dict = {(i, i + 1): gamma_tensor.clone() for i in range(num_nodes - 1)}

        return TreeStructure(adj_matrix=A, edge_list=edge_list, gamma_dict=gamma_dict)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.vanilla_vae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vanilla_vae.decode(z)

    def classic_tree_reparameterize(
        self, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Implements the orginal authors tree reparameterization for a tree structure.

        Args:
            mu (torch.Tensor): _description_
            sigma (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        batch_size, latent_dim = mu.size()
        ordering = (
            self.current_vi_structure.get_ordering()
        )  # A list of node indices (should have length == batch_size)
        device = mu.device
        z = torch.zeros_like(mu)
        e_dict: Dict[int, torch.Tensor] = {}

        for node in ordering:
            if self.current_vi_structure.get_parent(node) is None:
                # Node is root; sample e ~ N(0, I) for this node
                e_node = torch.randn(latent_dim, device=device)
                e_dict[node] = e_node
                z_val = mu[node] + sigma[node] * e_node
                z[node] = z_val
            else:
                # Non-root node: get the unique path from root to this node.
                path = self.current_vi_structure.get_path(
                    node
                )  # e.g., [root, ..., node]
                # Initialize contribution accumulator (shape: [latent_dim])
                contribution = torch.zeros(latent_dim, device=device)
                # First, contribution from the root branch.
                root_node = path[0]
                # Compute effective correlation from root to node.
                eff_corr_root = self.current_vi_structure.compute_effective_correlation(
                    root_node, node
                )
                # e for root should have been computed already.
                if root_node not in e_dict:
                    e_dict[root_node] = torch.randn(latent_dim, device=device)
                contribution = (
                    contribution + eff_corr_root * e_dict[root_node] * sigma[node]
                )
                # Now, for every intermediate node in the path (excluding the root and the current node)
                # Note: path[-1] is the node itself.
                if len(path) > 1:
                    for idx in range(1, len(path) - 1):
                        ancestor = path[idx]
                        # Compute effective correlation from current ancestor to node.
                        eff_corr_ancestor = (
                            self.current_vi_structure.compute_effective_correlation(
                                ancestor, node
                            )
                        )
                        # Get gamma for edge (parent(ancestor), ancestor)
                        parent_of_ancestor = self.current_vi_structure.get_parent(
                            ancestor
                        )
                        if parent_of_ancestor is None:
                            gamma_val = torch.ones(
                                latent_dim, device=device
                            )  # default for root
                        else:
                            gamma_val = self.current_vi_structure.gamma.get(
                                (parent_of_ancestor, ancestor),
                                torch.ones(latent_dim, device=device),
                            )
                        scaling = torch.sqrt(1.0 - gamma_val.pow(2))
                        if ancestor not in e_dict:
                            e_dict[ancestor] = torch.randn(latent_dim, device=device)
                        term = (
                            eff_corr_ancestor * scaling * e_dict[ancestor] * sigma[node]
                        )
                        contribution = contribution + term
                # Finally, set z[node] = mu[node] + sum of contributions.
                z[node] = mu[node] + contribution
        return z

    def fast_tree_reparameterize(
        self, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized version of the tree reparameterization:
        1) Build L: [B, B, D] – Cholesky matrix for each latent dimension D
        2) Sample e ~ N(0,I) [B,D]
        3) v = einsum("ijd,jd->id", L, e)  # (B,B,D) x (B,D) -> (B,D)
        4) z = mu + sigma * v

        Args:
            mu (torch.Tensor): Mean tensor of shape [B, D].
            sigma (torch.Tensor): Standard deviation tensor of shape [B, D].
        Returns:
            torch.Tensor: Reparameterized latent variable z of shape [B, D].
        """
        batch_size, latent_dim = mu.size()
        device = mu.device

        parent: torch.LongTensor = self.current_vi_structure.parent
        gamma_dict: Dict[Tuple[int, int], torch.Tensor] = (
            self.current_vi_structure.gamma
        )

        L = _compute_cholesky_tensor(batch_size, latent_dim, parent, gamma_dict, device)

        e = torch.randn((batch_size, latent_dim), device=device)

        v = torch.einsum("ijd,jd->id", L, e)
        z = mu + sigma * v
        return z

    def tree_reparameterize(
        self, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        if self.faster_version:
            return self.fast_tree_reparameterize(mu, sigma)
        else:
            return self.classic_tree_reparameterize(mu, sigma)

    def _compute_reconstruction_loss(
        self, recon: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        return -F.mse_loss(recon, x, reduction="sum")

    def _compute_prior_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(z.pow(2) + math.log(2 * math.pi))

    def _compute_entropy(
        self, sigma: torch.Tensor, tree: Optional[TreeStructure]
    ) -> torch.Tensor:
        const = math.log(2 * math.pi * math.e)
        singleton_entropy = 0.5 * torch.sum(const + torch.log(sigma.pow(2) + 1e-8))

        if tree is None or not tree.edge_list:
            return singleton_entropy

        gamma_vals = [
            torch.log(1 - tree.gamma[edge].pow(2) + 1e-8)
            for edge in tree.edge_list
            if edge in tree.gamma
        ]
        edge_correction = (
            0.5 * torch.sum(torch.stack(gamma_vals)) if gamma_vals else 0.0
        )
        return singleton_entropy + edge_correction

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        if (self.current_vi_structure is None) or (
            self.current_vi_structure.adj_matrix.size(0) != batch_size
        ):
            self.current_vi_structure = self._init_chain_tree(num_nodes=x.size(0))

        mu, sigma = self.encode(x)

        # print("== Step 2: Check encoder output ==")
        # print("Mu: mean", mu.mean().item(), "std", mu.std().item())
        # print("Sigma: mean", sigma.mean().item(), "std", sigma.std().item())
        # print("Any NaNs in mu?", torch.isnan(mu).any().item())
        # print("Any NaNs in sigma?", torch.isnan(sigma).any().item())
        # print("Sigma min/max:", sigma.min().item(), sigma.max().item())

        z = self.tree_reparameterize(mu, sigma)

        # print("== Step 3: Check latent z ==")
        # print("Z: mean", z.mean().item(), "std", z.std().item())
        # print("Any NaNs in z?", torch.isnan(z).any().item())

        reconstruction = self.decode(z)

        # print("== Step 4: Check reconstructions ==")
        # print("Reconstructed image: min", reconstruction.min().item(), "max", reconstruction.max().item())
        # print("Any NaNs in recon?", torch.isnan(reconstruction).any().item())
        log_px_z = self._compute_reconstruction_loss(reconstruction, x)
        prior_log_prob = self._compute_prior_log_prob(z)
        entropy = self._compute_entropy(sigma, self.current_vi_structure)

        elbo = log_px_z + prior_log_prob + entropy
        self.current_elbo = -elbo

        self.current_vi_structure = self.vi_optimizer.optimize_tree(
            self.current_vi_structure, batch_size, self.current_elbo
        )

        logvar = torch.log(sigma.pow(2) + 1e-8)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        mean_kl_per_dim = kl_per_dim.mean(dim=0)
        num_active_dims = (mean_kl_per_dim > 0.01).sum().item()

        return {
            "recon": reconstruction,
            "partial_loss": self.current_elbo,
            "num_active_dims": num_active_dims,
        }
