from argparse import ArgumentParser
from typing import Optional

import torch.cuda
import wandb as wandb
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from src.data_processing.image_dataset import Cifar10Dataset
from src.modules.autoencoder import AutoEncoder


def train_autoencoder(
        epochs: int,
        lr: float,
        train_batch_size: int,
        test_batch_size: int,
        wandb_login: Optional[str],
        save_path: Optional[str],
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, test_data = Cifar10Dataset('train'), Cifar10Dataset('test')
    train_loader = DataLoader(train_data, batch_size=train_batch_size)
    test_loader = DataLoader(test_data, batch_size=test_batch_size)
    autoencoder = AutoEncoder()
    autoencoder.to(device)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')

    if wandb_login:
        wandb.init(project='autoencoder', entity=wandb_login)
        wandb.config = {
            'lr': lr,
            'epochs': epochs,
            'train_batch_size': train_batch_size,
            'test_batch_size': test_batch_size,
        }

    with tqdm(total=epochs, desc="epoch") as bar:
        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()

                img = batch['img'].to(device)
                reconstructed = autoencoder(img)
                loss = criterion(reconstructed, img)

                loss.backward()
                optimizer.step()
                print(loss.item)

            bar.update(1)

            if save_path:
                autoencoder.save(save_path)

            if wandb_login:
                wandb.log({'loss': loss.item})


def main():
    parser = ArgumentParser()
    parser.add_argument(name='epochs', type=int, default=200)
    parser.add_argument(name='lr', type=float, default=3e-4)
    parser.add_argument(name='train_batch_size', type=int, default=16)
    parser.add_argument(name='test_batch_size', type=int, default=16)
    parser.add_argument(name='wandb_login', help='login for wandb to log process', type=Optional[str], default=None)
    parser.add_argument(name='save_path', help='path to save model', type=Optional[str], default=None)
    args = parser.parse_args()
    train_autoencoder(
        epochs=args['epochs'],
        lr=args['lr'],
        train_batch_size=args['train_batch_size'],
        test_batch_size=args['test_batch_size'],
        wandb_login=args['wandb_login'],
        save_path=args['save_path']
    )


if __name__ == '__main__':
    main()
