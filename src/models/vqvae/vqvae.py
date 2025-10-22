import torch
import torch.nn as nn
from typing import Tuple
from src.models.baseline.encoder import Encoder
from src.models.baseline.decoder import Decoder
from src.models.vqvae.quantizer import Quantizer


class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        residual_hiddens: int,
        num_residual_layers: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        image_size: int = 64,
    ):
        """
        VQVAE built on top of a VAE-style encoder/decoder backbone.
        Args:
            input_dim (int): Input channels (e.g., 1 for grayscale, 3 for RGB).
            hidden_dim (int): Hidden channels in encoder/decoder.
            residual_hiddens (int): Residual block hidden channels.
            num_residual_layers (int): Number of residual blocks.
            num_embeddings (int): Size of the codebook.
            embedding_dim (int): Dimensionality of embedding vectors.
            commitment_cost (float): Weight for commitment loss.
            device (str): "cuda" or "cpu".
            image_size (int): Height/Width of the input images.
        """
        super(VQVAE, self).__init__()
        self.device = device
        self.image_size = image_size

        # Encoder (same as in VAE)
        self.encoder = Encoder(
            in_channels=input_dim,
            num_hiddens=hidden_dim,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=residual_hiddens,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_dim, image_size, image_size)
            feat = self.encoder(dummy)
            self._feature_shape = feat.shape[1:]   

        self._pre_vq_conv = nn.Conv2d(
            in_channels=self._feature_shape[0],
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1
        )

        self._vq_layer = Quantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )

        # Decoder (same as in VAE)
        self.decoder = Decoder(
            in_channels=embedding_dim,
            num_hiddens=hidden_dim,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=residual_hiddens,
            out_channels=input_dim,
        )

        self.to(device)

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encoder(x)
        z = self._pre_vq_conv(z)

        quantized, quant_loss, encodings, perplexity = self._vq_layer(z)

        recon = self.decoder(quantized)

        return {
            "recon": recon,
            "partial_loss": quant_loss,
            "num_active_dims": perplexity,  
            "encodings": encodings,
        }
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z = self._pre_vq_conv(z)
        return z, None

