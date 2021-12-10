from os.path import join, normpath

import torch
from torch import nn, Tensor


class Classifier(nn.Module):
    def __init__(self, in_channels: int = 256, n_classes: int = 10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3),
            nn.Flatten(),
            nn.Linear(36, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def save(self, path: str):
        torch.save(self.state_dict(), join(normpath(path), 'classifier'))

    @staticmethod
    def load(path: str) -> nn.Module:
        model = Classifier()
        model.load_state_dict(torch.load(
            normpath(path),
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ))
        return model
