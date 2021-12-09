import torch
from torch import nn, Tensor
from torch.nn import Sequential

from src.modules.residual_block import ResidualBlock


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, num_hidden_units: int = 256):
        super().__init__()
        self.model = Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_hidden_units // 2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Conv2d(
                in_channels=num_hidden_units // 2,
                out_channels=num_hidden_units,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            ResidualBlock(num_hidden_units, num_hidden_units),
            ResidualBlock(num_hidden_units, num_hidden_units)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
