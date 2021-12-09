import torch
from torch import nn, Tensor


class Classifier(nn.Module):
    def __init__(self, in_features: int = 10, n_classes: int = 10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def save(self, path: str):
        torch.save(self.state_dict(), path + '/classifier')

    @staticmethod
    def load(path: str):
        model = Classifier()
        model.load_state_dict(torch.load(path))
        return model
