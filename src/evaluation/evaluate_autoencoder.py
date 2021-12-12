from argparse import ArgumentParser
from typing import Optional

import torch
import torchvision.utils
import wandb
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data_processing.cifar10_dataset import Cifar10Dataset
from src.data_processing.show_image import show_image
from src.modules.autoencoder import AutoEncoder


def evaluate_autoencoder(
        model: nn.Module,
        test_data: Optional[Dataset] = None,  # will be evaluated on cifar10 if no data given
        test_batch_size: int = 64,
        wandb_login: Optional[str] = None
):
    with torch.no_grad():
        if test_data is None:
            test_data = Cifar10Dataset('test')

        test_loader = DataLoader(test_data, batch_size=test_batch_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        criterion = nn.MSELoss(reduction='mean')

        with tqdm(total=len(test_loader), desc='evaluation') as bar:
            total_loss = 0
            for batch in test_loader:
                img = batch[0].to(device)
                reconstructed = model(img)
                total_loss += criterion(reconstructed, img).item()
                bar.update(1)

        total_loss /= len(test_loader)

        print(f'Validation loss: {total_loss}')

        if wandb_login:
            wandb.init(project='autoencoder', entity=wandb_login)
            wandb.log({'validation_autoencoder_loss': total_loss})

        show_image(torchvision.utils.make_grid(img[:5]))
        show_image(torchvision.utils.make_grid(reconstructed[:5]))


def main():
    parser = ArgumentParser()
    parser.add_argument('autoencoder_path', help='path to saved autoencoder model', type=str)
    parser.add_argument('--wandb_login', help='wandb login to log loss', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    autoencoder = AutoEncoder().load_model(args.autoencoder_path)
    evaluate_autoencoder(
        model=autoencoder,
        test_batch_size=args.batch_size,
        wandb_login=args.wandb_login
    )


if __name__ == '__main__':
    main()
