import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, Tuple, Optional

from src.models.treevi.pyg_tree import PyGTreeOptimizer
from src.models.treevi.tree_variational_interface.tree_vi_structure.tree_vi import (
    TreeStructure,
)
from src.models.vae.vae import VAE


class GeoVAE(nn.Module):
    """
    VAE + PyG‐based TreeVI with GNN optimized structure.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        residual_hiddens: int,
        num_residual_layers: int,
        latent_dim: int = 256,
        emb_dim: int = 16,
        hidden_dim_gnn: int = 64,
        num_inst_gnn_layers: int = 2,
        optimizer_config: dict = {},
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        image_size: int = 64,
    ):
        super(GeoVAE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim

        # Vanilla VAE
        self.vanilla_vae = VAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            residual_hiddens=residual_hiddens,
            num_residual_layers=num_residual_layers,
            latent_dim=latent_dim,
            device=device,
            image_size=image_size,
        )

        # Instance-level GNN
        self.inst_convs = nn.ModuleList(
            [GCNConv(latent_dim, latent_dim) for _ in range(num_inst_gnn_layers)]
        )

        # Dimensional GNN
        self.dim_adj_logits = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.dim_embeddings = nn.Parameter(torch.randn(latent_dim, emb_dim))
        self.dim_gnn_1 = GCNConv(emb_dim, hidden_dim_gnn)
        self.dim_gnn_2 = GCNConv(hidden_dim_gnn, 1)

        self.tree_optimizer = PyGTreeOptimizer.GeoVAEOptimizer(
            latent_dim=latent_dim, hidden_dim=hidden_dim_gnn, device=device
        )

        self.current_inst_edge_index = None
        self.current_gamma_dict = {}
        

    def _init_chain_tree_edge_index(self, num_nodes: int) -> torch.LongTensor:
        """
        Create a simple chain of nodes for num_nodes. Returns bidirected edge_index of size: [2, 2*(num_nodes-1)].
        """
        src = []
        dst = []
        for i in range(num_nodes - 1):
            src += [i, i + 1]
            dst += [i + 1, i]
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        return edge_index

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.vanilla_vae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vanilla_vae.decode(z)

    def compute_instance_gnn(
        self, mu: torch.Tensor, edge_index: torch.LongTensor
    ) -> torch.Tensor:
        """
        Passes mu:[B,D] through several GCNConv layers according to edge_index: [2, E].
        Returns v_inst: [B,D].
        """
        h = mu
        for conv in self.inst_convs:
            h = F.relu(conv(h, edge_index))
        return h

    def compute_dim_gnn(self) -> torch.Tensor:
        """
        Create a graph of latent dimensions: dim_adj_logits → binarne edge_index_dim.
        Passes dim_embeddings: [D, emb_dim] through 2 layers of GCNConv and returns dim_importance: [D].
        """
        D = self.latent_dim

        adj_logits = self.dim_adj_logits + self.dim_adj_logits.t()
        adj_logits.fill_diagonal_(0.0)
        adj_probs = torch.sigmoid(adj_logits)

        mask = (adj_probs > 0.5).float()
        src, dst = torch.nonzero(mask, as_tuple=True)
        if src.numel() == 0:
            temp = torch.ones((D, D), device=self.device)
            temp.fill_diagonal_(0.0)
            src, dst = torch.where(temp > 0)
        edge_index_dim = torch.stack([src, dst], dim=0)

        x_dim = self.dim_embeddings
        x_dim = F.relu(self.dim_gnn_1(x_dim, edge_index_dim))
        x_dim = self.dim_gnn_2(x_dim, edge_index_dim)
        dim_importance = x_dim.squeeze(1)

        return dim_importance

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)
        mu, sigma = self.encode(x)

        if (self.current_inst_edge_index is None) or (
            self.current_inst_edge_index.size(1) != 2 * (B - 1)
        ):
            self.current_inst_edge_index = self._init_chain_tree_edge_index(B)
        edge_index = self.current_inst_edge_index

        v_inst = self.compute_instance_gnn(mu, edge_index)
        dim_importance = self.compute_dim_gnn()

        v = v_inst * dim_importance.view(1, -1)
        
        eps = torch.randn_like(mu)
        z = mu + sigma * (eps * v)
        
        #z = mu + sigma * v
        recon = self.decode(z)

        log_px_z = -F.mse_loss(recon, x, reduction="sum")
        prior_log_prob = -0.5 * torch.sum(z.pow(2) + math.log(2 * math.pi))
        entropy = 0.5 * torch.sum(1 + torch.log(sigma.pow(2) + 1e-8))
        elbo = log_px_z + prior_log_prob + entropy

        logvar = torch.log(sigma.pow(2) + 1e-8)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        mean_kl_per_dim = kl_per_dim.mean(dim=0)
        num_active_dims = (mean_kl_per_dim > 0.01).sum().item()

        new_edge_index, new_gamma_dict = self.tree_optimizer(
            current_edge_index=self.current_inst_edge_index,
            mu=mu,
            gamma_dict=self.current_gamma_dict,
        )
        self.current_inst_edge_index = new_edge_index
        self.current_gamma_dict = new_gamma_dict

        return {
            "recon": recon,
            "elbo": elbo,
            "num_active_dims": num_active_dims,
            "dim_importance": dim_importance,
            "v_inst": v_inst,
        }
