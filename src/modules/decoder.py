import torch
from torch import nn, Tensor

from src.modules.residual_block import ResidualBlock


class Decoder(nn.Module):
    def __init__(self, num_hidden_units: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            ResidualBlock(num_hidden_units, num_hidden_units),
            ResidualBlock(num_hidden_units, num_hidden_units),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=num_hidden_units,
                out_channels=num_hidden_units // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=num_hidden_units // 2,
                out_channels=3,
                kernel_size=2,
                stride=2,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
