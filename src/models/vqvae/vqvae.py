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
        device: str,
    ):
        """
        Vector Quantized Variational Autoencoder (VQVAE) for 64x64 RGB images.
        Args:
            input_dim (int): Number of input channels (3 for RGB).
            hidden_dim (int): Number of hidden channels in encoder/decoder.
            residual_hiddens (int): Hidden channels in residual blocks.
            num_residual_layers (int): Number of residual blocks.
            num_embeddings (int): Number of embeddings in the codebook.
            embedding_dim (int): Dimension of each embedding vector.
            commitment_cost (float): Commitment cost for the VQ layer.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        super(VQVAE, self).__init__()

        self.device = device
        # Encoder
        self.encoder = Encoder(
            in_channels=input_dim,
            num_hiddens=hidden_dim,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=residual_hiddens,
        )
        self._pre_vq_conv = nn.Conv2d(
            in_channels=hidden_dim, out_channels=embedding_dim, kernel_size=1, stride=1
        )

        # Codebook (Quantization Layer)
        self._vq_layer = Quantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )

        # Decoder
        self.decoder = Decoder(
            in_channels=embedding_dim,
            num_hiddens=hidden_dim,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=residual_hiddens,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the VQVAE.
        Args:
            x (torch.Tensor): Input tensor - image of shape [Batch, Channels, Height, Width].
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - Reconstructed tensor of shape [Batch, Channels, Height, Width].
                - Loss value for the quantization (commitment + codebook).
                - One hot encoded indices of the closest embeddings. [B*H*W, num_embeddings]
                - Perplexity of the quantization.
        """
        # Encoding
        z = self.encoder(x)
        z = self._pre_vq_conv(z)

        # Quantization
        quantized, quant_loss, encodings, perplexity = self._vq_layer(z)

        # Decode quantized representation
        reconstructed = self.decoder(quantized)

        return {
            "recon": reconstructed,
            "partial_loss": quant_loss,
            "num_active_dims": perplexity,
            "encodings": encodings,
        }
