from argparse import ArgumentParser
from typing import Optional

import torch.cuda
import wandb as wandb
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from src.data_processing.image_dataset import Cifar10Dataset
from src.evaluation.evaluate_autoencoder import evaluate_autoencoder
from src.modules.autoencoder import AutoEncoder
from src.training.add_training_arguments import add_training_arguments


def train_autoencoder(
        epochs: int,
        lr: float,
        train_batch_size: int,
        test_batch_size: int,
        to_evaluate: bool,
        wandb_login: Optional[str],
        save_path: Optional[str],
        seed: int,
):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, test_data = Cifar10Dataset('train'), Cifar10Dataset('test')
    train_loader = DataLoader(train_data, batch_size=train_batch_size)
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
            for batch in train_loader:
                optimizer.zero_grad()

                img = batch['img'].to(device)
                reconstructed = autoencoder(img)
                loss = criterion(reconstructed, img)

                loss.backward()
                optimizer.step()
                print(loss.item())

            bar.update(1)

            if save_path:
                autoencoder.save(save_path, epoch, optimizer.state_dict())

            if wandb_login:
                wandb.log({'autoencoder_loss': loss.item()})

    if to_evaluate:
        evaluate_autoencoder(
            model=autoencoder,
            test_loader=DataLoader(test_data, batch_size=test_batch_size),
            compare_images=True,
            wandb_login=wandb_login
        )


def main():
    parser = ArgumentParser()
    parser = add_training_arguments(parser)
    args = parser.parse_args()
    train_autoencoder(
        epochs=args['epochs'],
        lr=args['lr'],
        train_batch_size=args['train_batch_size'],
        test_batch_size=args['test_batch_size'],
        to_evaluate=args['to_evaluate'],
        wandb_login=args['wandb_login'],
        save_path=args['save_path'],
        seed=args['seed']
    )


if __name__ == '__main__':
    main()
