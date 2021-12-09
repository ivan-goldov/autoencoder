from argparse import ArgumentParser

import torch.cuda
from torch import nn
from torch.utils.data import DataLoader

from src.data_processing.image_dataset import Cifar10Dataset
from src.modules.autoencoder import AutoEncoder


def train_autoencoder(
        epochs: int = 100,
        lr: float = 3e-4,
        train_batch_sie: int = 16,
        test_batch_size: int = 16,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, test_data = Cifar10Dataset('train'), Cifar10Dataset('test')
    train_loader = DataLoader(train_data, batch_size=train_batch_sie)
    test_loader = DataLoader(test_data, batch_size=test_batch_size)
    autoencoder = AutoEncoder()
    autoencoder.to(device)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')

    for batch in train_loader:
        optimizer.zero_grad()

        img = batch['img'].to(device)
        reconstructed = autoencoder(img)
        loss = criterion(reconstructed, img)

        loss.backward()
        optimizer.step()
        print(loss.item)


def main():
    parser = ArgumentParser()
    parser.add_argument(name='')


if __name__ == '__main__':
    train_autoencoder()
