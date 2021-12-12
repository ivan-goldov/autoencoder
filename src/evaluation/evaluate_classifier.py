from argparse import ArgumentParser
from typing import Optional, List, Tuple, Callable

import torch
import wandb
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data_processing.cifar10_dataset import Cifar10Dataset
from src.modules.autoencoder import AutoEncoder
from src.modules.classifier import Classifier


def evaluate_classifier(
        classifier: nn.Module,
        metrics: List[Tuple[str, Callable]],
        test_data: Optional[Dataset] = None,
        test_batch_size: int = 64,
        wandb_login: Optional[str] = None,
):
    with torch.no_grad():
        if not test_data:
            test_data = Cifar10Dataset('test')

        test_loader = DataLoader(test_data, batch_size=test_batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier.to(device)

        criterion = nn.CrossEntropyLoss()

        predictions = []
        targets = []

        with tqdm(total=len(test_loader), desc='evaluation') as bar:
            total_loss = 0
            for batch in test_loader:
                img, labels = batch[0].to(device), batch[1].to(device)
                outputs = classifier(img)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                predictions.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
                targets.extend(labels.detach().cpu().numpy())

                bar.update(1)

        if wandb_login:
            wandb.log({'evaluate_classifier_loss': total_loss})
            for name, metrics in metrics:
                wandb.log({'validation_' + name: metrics(targets, predictions)})


def main():
    parser = ArgumentParser()
    parser.add_argument('autoencoder_path', help='path to saved autoencoder model', type=str)
    parser.add_argument('classifier_path', help='path to saved classifier model', type=str)
    parser.add_argument('--wandb_login', help='wandb login to log loss and metrics', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    autoencoder = AutoEncoder.load_model(args.autoencoder_path)
    evaluate_classifier(
        classifier=Classifier.load(args.classifier_path, encoder=autoencoder.get_encoder()),
        metrics=[('accuracy', accuracy_score)],
        test_data=Cifar10Dataset('test'),
        test_batch_size=args.batch_size,
        wandb_login=args.wandb_login
    )


if __name__ == '__main__':
    main()
