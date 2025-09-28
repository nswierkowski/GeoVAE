import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A residual block that consists of two convolutional layers with ReLU activations.
    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of output channels for the second convolutional layer.
        num_residual_hiddens (int): Number of output channels for the first convolutional layer.
    """

    def __init__(self, in_channels: int, num_hiddens: int, num_residual_hiddens: int):
        super(ResidualBlock, self).__init__()

        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._block(x)


class ResidualStack(nn.Module):
    """
    A stack of residual blocks.
    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of output channels for the second convolutional layer in each block.
        num_residual_layers (int): Number of residual blocks in the stack.
        num_residual_hiddens (int): Number of output channels for the first convolutional layer in each block.
    """

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ):
        super(ResidualStack, self).__init__()

        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                ResidualBlock(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(self._num_residual_layers)
            ]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)

        return F.relu(x)
