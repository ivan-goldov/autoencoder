from os.path import join, normpath

import torch
from torch import nn, Tensor


class Classifier(nn.Module):
    def __init__(self, encoder: nn.Module, in_channels: int = 256, n_classes: int = 10):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(576, n_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(self.encoder(x))

    def save(self, path: str):
        torch.save(self.state_dict(), join(normpath(path), 'classifier'))

    @staticmethod
    def load(classifier_path: str, encoder: nn.Module) -> nn.Module:
        model = Classifier(encoder)
        model.load_state_dict(torch.load(
            normpath(classifier_path),
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ))
        return model
