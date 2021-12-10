from typing import Dict, Tuple

import torch
from torch import nn, Tensor

from src.modules.decoder import Decoder
from src.modules.encoder import Encoder
from os.path import join, normpath


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def get_encoder(self) -> nn.Module:
        return self.encoder

    def get_decoder(self) -> nn.Module:
        return self.decoder

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    def save_checkpoint(self, path: str, epoch: int, opt_state_dict: Dict):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': opt_state_dict,
            'epoch': epoch
        }, join(normpath(path), 'autoencoder_ckp'))

    def save_model(self, path: str):
        torch.save(self.state_dict(), join(normpath(path), 'autoencoder'))

    @staticmethod
    def load_checkpoint(path: str) -> Tuple[nn.Module, torch.optim.Adam, int]:
        model = AutoEncoder()
        optimizer = torch.optim.Adam(model.parameters())
        checkpoint = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return model, optimizer, epoch

    @staticmethod
    def load_model(path: str) -> nn.Module:
        model = AutoEncoder()
        model.load_state_dict(torch.load(
            normpath(path),
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ))
        return model
