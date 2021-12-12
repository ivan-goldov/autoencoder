from argparse import ArgumentParser
from functools import partial
from typing import Optional

import torch
import torchvision
import wandb
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from src.data_processing.cifar10_dataset import Cifar10Dataset
from src.evaluation.evaluate_classifier import evaluate_classifier
from src.modules.autoencoder import AutoEncoder
from src.modules.classifier import Classifier
from src.training.add_training_arguments import add_training_arguments


def train_classifier(
        encoder: nn.Module,
        epochs: int = 100,
        lr: float = 3e-4,
        train_batch_size: int = 512,
        test_batch_size: int = 64,
        hidden_size: int = 256,
        wandb_login: Optional[str] = None,
        save_path: Optional[str] = None,
        seed: int = 0,
):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier = Classifier(encoder, in_channels=hidden_size)
    classifier.to(device)

    train_loader = DataLoader(Cifar10Dataset('train'), batch_size=train_batch_size)
    # train_loader = DataLoader(
    # torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform),
    #                            batch_size=512)

    if wandb_login:
        wandb.init(project='autoencoder', entity=wandb_login)
        wandb.config = {
            'epochs': epochs,
            'lr': lr,
            'train_batch_size': train_batch_size,
            'test_batch_size': test_batch_size,
            'save_path': save_path
        }

    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss()

    metrics = [
        ('accuracy', accuracy_score),
        ('f1', partial(f1_score, average='macro')),
    ]

    with tqdm(total=epochs, desc='training') as bar:
        for epoch in range(epochs):
            epoch_loss = 0
            predictions = []
            targets = []
            for batch in train_loader:
                optimizer.zero_grad()

                img, labels = batch[0].to(device), batch[1].to(device)
                outputs = classifier(img)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                predictions.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
                targets.extend(labels.detach().cpu().numpy())

            evaluate_classifier(classifier, metrics=metrics, wandb_login=wandb_login)

            epoch_loss /= len(train_loader)
            bar.update(1)
            print(epoch_loss)

            if wandb_login:
                wandb.log({'classifier_loss': epoch_loss})
                for name, metric in metrics:
                    wandb.log({'training ' + name: metric(targets, predictions)})

            if save_path:
                classifier.save(save_path)


def main():
    parser = ArgumentParser()
    parser.add_argument('autoencoder_model_path', help='path for autoencoder model', type=str)
    parser = add_training_arguments(parser)
    args = parser.parse_args()
    # autoencoder = AutoEncoder.load_model(args.autoencoder_model_path)
    autoencoder = AutoEncoder()
    encoder = autoencoder.get_encoder()
    train_classifier(
        encoder=encoder,
        epochs=args.epochs,
        lr=args.lr,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        hidden_size=args.hidden_size,
        wandb_login=args.wandb_login,
        save_path=args.save_path,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
