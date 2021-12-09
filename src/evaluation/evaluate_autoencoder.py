from typing import Optional

import torch
import torchvision.utils
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_processing.show_image import show_image


def evaluate_autoencoder(
        model: nn.Module,
        test_loader: DataLoader,
        compare_images: bool = False,
        wandb_login: Optional[str] = None
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss(reduction='mean')

    with tqdm(total=len(test_loader), desc='evaluation') as bar:
        total_loss = 0
        for batch in test_loader:
            img = batch['img'].to(device)
            reconstructed = model(img)
            total_loss += criterion(reconstructed, img).item()

        bar.update(1)

    if wandb_login:
        wandb.log({'evaluate_autoencoder_loss': total_loss})

    if compare_images:
        show_image(torchvision.utils.make_grid(reconstructed, img))
