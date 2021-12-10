from typing import Optional

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

from src.data_processing.image_dataset import Cifar10Dataset


def evaluate_classifier(
        classifier: nn.Module,
        encoder: nn.Module,
        test_data: Optional[Dataset] = None,  # will be evaluated on cifar10 if no data given
        test_batch_size: int = 16,
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

        with tqdm(total=len(test_loader), desc='evaluation') as bar:
            total_loss = 0
            for batch in test_loader:
                img, labels = batch['img'].to(device), batch['labels'].to(device)
                hidden_representation = encoder(img)
                outputs = classifier(hidden_representation)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                predictions.append(torch.argmax(outputs).item())
                targets.append(labels.item())

            bar.update(1)

        scores = [
            accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
        ]

        for score in scores:
            print(f'{score.__name__}: {accuracy_score(targets, predictions)}')

        if wandb_login:
            wandb.log({'evaluate_classifier_loss': total_loss})
            for score in scores:
                wandb.log(f'{score.__name__}: {accuracy_score(targets, predictions)}')
