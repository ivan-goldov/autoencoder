from argparse import ArgumentParser
from typing import Optional

import torch
import torchvision
import wandb
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

from src.data_processing.image_dataset import Cifar10Dataset
from src.data_processing.show_image import show_image
from src.modules.autoencoder import AutoEncoder
from src.modules.classifier import Classifier


def evaluate_classifier(
        classifier: nn.Module,
        encoder: nn.Module,
        test_data: Optional[Dataset] = None,  # will be evaluated on cifar10 if no data given
        test_batch_size: int = 16,
        wandb_login: Optional[str] = None
):
    with torch.no_grad():
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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
                img, labels = batch['img'].to(device), batch['label'].to(device)
                hidden_representation = encoder(img)
                outputs = classifier(hidden_representation)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                predictions.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
                targets.extend(labels.detach().cpu().numpy())

            bar.update(1)

        scores = [
            accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
        ]

        for score in scores:
            print(f'{score.__name__}: {accuracy_score(targets, predictions)}')

        show_image(torchvision.utils.make_grid(img[:5]))
        print(' '.join(classes[predictions[j]] for j in range(5)))

        if wandb_login:
            wandb.log({'evaluate_classifier_loss': total_loss})
            for score in scores:
                wandb.log(f'{score.__name__}: {accuracy_score(targets, predictions)}')


def main():
    parser = ArgumentParser()
    parser.add_argument('autoencoder_path', help='path to saved autoencoder model', type=str)
    parser.add_argument('classifier_path', help='path to saved classifier model', type=str)
    parser.add_argument('--wandb_login', help='wandb login to log loss and metrics', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    print(args)
    autoencoder = AutoEncoder().load_model(args.autoencoder_path)
    evaluate_classifier(
        classifier=Classifier().load(args.classifier_path),
        encoder=autoencoder.get_encoder(),
        test_batch_size=args.batch_size,
        wandb_login=args.wandb_login
    )


if __name__ == '__main__':
    main()
