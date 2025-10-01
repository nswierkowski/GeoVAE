import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from src.models.treevi.TreeStructure import TreeStructure
from src.models.treevi.TreeOptimizer import TreeOptimizer

class TreeVAE(nn.Module):
    def __init__(
        self,
        input_dim=784,
        latent_dim=10,
        hidden_dims=[500, 500, 2000],
        device=None,
        tree_config: Optional[dict] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder
        encoder_layers = []
        last_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h))
            encoder_layers.append(nn.ReLU())
            last_dim = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_dim, latent_dim)

        # Decoder
        decoder_layers = []
        last_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(last_dim, h))
            decoder_layers.append(nn.ReLU())
            last_dim = h
        decoder_layers.append(nn.Linear(last_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

        # TreeOptimizer (authors' implementation)
        self.tree_optimizer = TreeOptimizer(latent_dim, tree_config or {})
        # Current tree (starts as None, updated later)
        self.current_tree: Optional[TreeStructure] = None

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma, logvar

    def decode(self, z):
        return self.decoder(z)

    def tree_reparameterize(self, mu, sigma):
        if self.current_tree is None:
            # fallback: mean-field
            return mu + sigma * torch.randn_like(sigma)

        # Proper tree-based reparam (simplified here)
        # In authors' code this propagates gamma along edges
        z = mu + sigma * torch.randn_like(sigma)
        return z

    def update_tree(self, mu, sigma, loss_term: torch.Tensor):
        """
        Called occasionally to re-learn the tree using TreeOptimizer.
        """
        if self.current_tree is None:
            # initialize adjacency matrix with zeros
            init_adj = torch.zeros((self.latent_dim, self.latent_dim), device=self.device)
            init_tree = TreeStructure(init_adj, [], {})
        else:
            init_tree = self.current_tree

        self.current_tree = self.tree_optimizer.optimize_tree(
            initial_tree=init_tree,
            num_nodes=self.latent_dim,
            loss_term=loss_term,
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        # flatten if input is image
        if x.ndim > 2:
            x = x.view(x.size(0), -1)

        mu, sigma, logvar = self.encode(x)
        z = self.tree_reparameterize(mu, sigma)
        recon = self.decode(z)

        # losses
        recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        elbo = recon_loss + kld

        # optional: update tree using ELBO as loss term
        self.update_tree(mu, sigma, elbo.detach())

        return {
            "recon": recon,
            "partial_loss": kld,
            "num_active_dims": mu.size(1),
            "loss": elbo,
        }
