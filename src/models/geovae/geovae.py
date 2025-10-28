import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE
from typing import Dict, Tuple
from enum import Enum

from src.models.geovae.geovae_optimizer import GeoVAEOptimizer
from src.models.vae.vae import VAE


class GraphConvType(Enum):
    GCN = 1
    GAT = 2
    GST = 3  


class GeoVAE(nn.Module):
    """
    GeoVAE: Variational Autoencoder with instance-level and dimension-level GNN
    that can use different convolution types (GCN, GAT, GraphSAGE).
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
        graph_conv_type: GraphConvType = GraphConvType.GCN,
        num_dim_gnn_layers: int = 2
    ):
        super(GeoVAE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.graph_conv_type = graph_conv_type

        self.vanilla_vae = VAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            residual_hiddens=residual_hiddens,
            num_residual_layers=num_residual_layers,
            latent_dim=latent_dim,
            device=device,
            image_size=image_size,
        )

        if graph_conv_type == GraphConvType.GCN:
            ConvLayer = GCNConv
        elif graph_conv_type == GraphConvType.GAT:
            ConvLayer = lambda in_c, out_c: GATConv(in_c, out_c, heads=1, concat=False)
        elif graph_conv_type == GraphConvType.GST:
            ConvLayer = lambda in_c, out_c: GraphSAGE(in_c, out_c, num_layers=1) 
        else:
            raise ValueError(f"Unsupported graph_conv_type: {graph_conv_type}")

        self.inst_convs = nn.ModuleList(
            [ConvLayer(latent_dim, latent_dim) for _ in range(num_inst_gnn_layers)]
        )

        self.dim_adj_logits = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.dim_embeddings = nn.Parameter(torch.randn(latent_dim, emb_dim))

        dim_layers = []
        in_c = emb_dim
        for i in range(num_dim_gnn_layers):
            out_c = hidden_dim_gnn if i < num_dim_gnn_layers - 1 else 1
            dim_layers.append(ConvLayer(in_c, out_c))
            in_c = out_c
        self.dim_gnn_layers = nn.ModuleList(dim_layers)


        self.tree_optimizer = GeoVAEOptimizer(
            latent_dim=latent_dim,
            hidden_dim=optimizer_config.get("hidden_dim", hidden_dim_gnn),
            num_gnn_layers=optimizer_config.get("num_gnn_layers", 2),
            graph_conv_type=optimizer_config.get("graph_conv_type", graph_conv_type.name),
            device=device,
        )

        self.current_inst_edge_index = None
        self.current_gamma_dict = {}

    def _init_chain_tree_edge_index(self, num_nodes: int) -> torch.LongTensor:
        src, dst = [], []
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
        h = mu
        for conv in self.inst_convs:
            h = F.leaky_relu(conv(h, edge_index))
        return h

    def compute_dim_gnn(self, mask_threshold: float = 0.6) -> torch.Tensor:
        D = self.latent_dim
        adj_logits = self.dim_adj_logits + self.dim_adj_logits.t()
        adj_logits.fill_diagonal_(0.0)
        adj_probs = torch.sigmoid(adj_logits)

        mask = (adj_probs > mask_threshold).float()
        src, dst = torch.nonzero(mask, as_tuple=True)
        if src.numel() == 0:
            temp = torch.ones((D, D), device=self.device)
            temp.fill_diagonal_(0.0)
            src, dst = torch.where(temp > 0)
        edge_index_dim = torch.stack([src, dst], dim=0)

        x_dim = self.dim_embeddings
        for i, conv in enumerate(self.dim_gnn_layers):
            x_dim = conv(x_dim, edge_index_dim)
            if i < len(self.dim_gnn_layers) - 1:
                x_dim = F.leaky_relu(x_dim)
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
