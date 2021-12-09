from typing import Dict

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

    def save_checkpoint(self, path: str, epoch: int, opt_state_dict: Dict):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': opt_state_dict,
            'epoch': epoch
        }, path + '/autoencoder_ckp')

    def save_model(self, path: str):
        torch.save(self.state_dict(), path + '/autoencoder')

    @staticmethod
    def load_checkpoint(path: str):
        model = AutoEncoder()
        optimizer = torch.optim.Adam(model.parameters())
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return model, optimizer, epoch

    @staticmethod
    def load_model(path: str):
        model = AutoEncoder()
        model.load_state_dict(torch.load(path))
        return model
