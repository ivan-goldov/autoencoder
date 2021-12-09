import torch
from torch import nn, Tensor

from src.modules.decoder import Decoder
from src.modules.encoder import Encoder


class AutoEncoder(nn.Module):
    def __init__(self, in_channels: int = 0, out_channels: int = 0, kernel_size: int = 3, padding: int = 0):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    def save(self, path: str):
        torch.save(self.state_dict(), path)
