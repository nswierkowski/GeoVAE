from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        """
        Vector Quantization layer for VQ-VAE.
        Args:
            num_embeddings (int): Number of embeddings in the codebook.
            embedding_dim (int): Dimension of each embedding vector.
            commitment_cost (float): Weight for the commitment loss term.
        """
        super(Quantizer, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost

        self._embeddings = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embeddings.weight.data.uniform_(
            -1.0 / self._num_embeddings, 1.0 / self._num_embeddings
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the quantization layer.
        Args:
            x (torch.Tensor): Input tensor - image of shape [Batch, Channels, Height, Width].
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - Quantized tensor of shape [Batch, Channels, Height, Width].
                - "Loss value for the quantization (commitment + codebook)".
                - One hot encoded indices of the closest embeddings. [B*H*W, num_embeddings]
                - Perplexity of the quantization.
        """
        # Permute input to [Batch, Height, Width, Channels], contiguous is for memory efficiency
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape

        # Flatten the input to [Batch * Height * Width, Channels]
        x_flattened = x.view(-1, self._embedding_dim)

        # Calculates L2 norm between input and codebook embeddings
        # distances = torch.cdist(x_flattened, self.embeddings.weight) - computationally expensive
        distances = (
            torch.sum(x_flattened**2, dim=1, keepdim=True)
            + torch.sum(self._embeddings.weight**2, dim=1)
            - 2 * torch.matmul(x_flattened, self._embeddings.weight.t())
        )

        # Find the closest embedding for each input vector (lowest value of each row)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=x.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Based on indicies of the lowest distances select the codebook embeddings
        # The alternative is to make a one-hot encoding of chosen codebook embeddings and then multiply with the codebook
        quantized = torch.matmul(encodings, self._embeddings.weight)

        # Reshape quantized to permuted input shape
        quantized = quantized.view(x_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        quant_loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # Straight through estimator trick
        # This allows gradients to flow through the quantized values
        quantized = x + (quantized - x).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Calculate perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, quant_loss, encodings, perplexity.item()
