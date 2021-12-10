from argparse import ArgumentParser
from typing import Optional

import torch.cuda
import torchvision
import wandb as wandb
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from src.evaluation.evaluate_autoencoder import evaluate_autoencoder
from src.modules.autoencoder import AutoEncoder
from src.training.add_training_arguments import add_training_arguments


def train_autoencoder(
        epochs: int = 50,
        lr: float = 3e-4,
        train_batch_size: int = 64,
        test_batch_size: int = 16,
        wandb_login: Optional[str] = None,
        save_path: Optional[str] = None,
        seed: int = 0,
):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train_data, test_data = Cifar10Dataset('train'), Cifar10Dataset('test')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5, )])
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=256, num_workers=16)
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

    with tqdm(total=epochs, desc='training') as bar:
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()

                img = batch[0].to(device)
                reconstructed = autoencoder(img)
                loss = criterion(reconstructed, img)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            print(f'Last epoch loss: {epoch_loss}')

            bar.update(1)

            evaluate_autoencoder(
                model=autoencoder,
                test_data=test_data,
                test_batch_size=test_batch_size,
                wandb_login=wandb_login
            )

            if save_path:
                autoencoder.save_checkpoint(save_path, epoch, optimizer.state_dict())

            if wandb_login:
                wandb.log({'autoencoder_loss': epoch_loss})


def main():
    parser = ArgumentParser()
    parser = add_training_arguments(parser)
    args = parser.parse_args()
    train_autoencoder(
        epochs=args.epochs,
        lr=args.lr,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        wandb_login=args.wandb_login,
        save_path=args.save_path,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
