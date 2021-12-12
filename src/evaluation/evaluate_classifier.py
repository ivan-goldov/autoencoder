from argparse import ArgumentParser
from functools import partial
from typing import Optional

import torch
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data_processing.cifar10_dataset import Cifar10Dataset
from src.modules.autoencoder import AutoEncoder
from src.modules.classifier import Classifier


def evaluate_classifier(
        classifier: nn.Module,
        test_data: Optional[Dataset] = None,
        test_batch_size: int = 64,
        wandb_login: Optional[str] = None
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

        # with tqdm(total=len(test_loader), desc='evaluation') as bar:
        total_loss = 0
        for batch in test_loader:
            img, labels = batch[0].to(device), batch[1].to(device)
            outputs = classifier(img)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
            targets.extend(labels.detach().cpu().numpy())

            # bar.update(1)

        scores = [
            ('accuracy', accuracy_score),
            ('f1', partial(f1_score, average='macro')),
        ]

        for name, score in scores:
            print(f'{name}: {score(targets, predictions)}')

        # show_image(torchvision.utils.make_grid(img[:5]))
        # print(' '.join(test_data.label_to_str([predictions[j]]) for j in range(5)))

        if wandb_login:
            wandb.log({'evaluate_classifier_loss': total_loss})
            for name, score in scores:
                wandb.log({name: score(targets, predictions)})


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
        test_data=Cifar10Dataset('test'),
        test_batch_size=args.batch_size,
        wandb_login=args.wandb_login
    )


if __name__ == '__main__':
    main()
