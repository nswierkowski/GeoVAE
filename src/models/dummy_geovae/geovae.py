import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from src.models.vae.vae import VAE


class DummyGeoVAE(nn.Module):
    """
    Baseline model: VAE + simple MLPs instead of GNNs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        residual_hiddens: int,
        num_residual_layers: int,
        latent_dim: int = 256,
        emb_dim: int = 16,
        hidden_dim_mlp: int = 64,
        num_inst_mlp_layers: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        image_size: int = 64,
    ):
        super(DummyGeoVAE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim

        # Vanilla VAE backbone
        self.vanilla_vae = VAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            residual_hiddens=residual_hiddens,
            num_residual_layers=num_residual_layers,
            latent_dim=latent_dim,
            device=device,
            image_size=image_size,
        )

        inst_layers = []
        in_dim = latent_dim
        for _ in range(num_inst_mlp_layers):
            inst_layers.append(nn.Linear(in_dim, hidden_dim_mlp))
            inst_layers.append(nn.ReLU())
            in_dim = hidden_dim_mlp
        inst_layers.append(nn.Linear(in_dim, latent_dim))
        self.inst_mlp = nn.Sequential(*inst_layers)

        self.dim_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim_mlp),
            nn.ReLU(),
            nn.Linear(hidden_dim_mlp, 1)
        )

        self.dim_embeddings = nn.Parameter(torch.randn(latent_dim, emb_dim))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.vanilla_vae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vanilla_vae.decode(z)

    def compute_instance_mlp(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Passes mu:[B,D] through simple MLP.
        Returns v_inst: [B,D].
        """
        return self.inst_mlp(mu)

    def compute_dim_mlp(self) -> torch.Tensor:
        """
        Each latent dim embedding goes through MLP → importance score.
        Returns dim_importance: [D].
        """
        x_dim = self.dim_embeddings
        x_dim = self.dim_mlp(x_dim)  # [D,1]
        return x_dim.squeeze(1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, sigma = self.encode(x)

        v_inst = self.compute_instance_mlp(mu)                 # [B,D]
        dim_importance = self.compute_dim_mlp()                # [D]
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

        return {
            "recon": recon,
            "elbo": elbo,
            "num_active_dims": num_active_dims,
            "dim_importance": dim_importance,
            "v_inst": v_inst,
        }
