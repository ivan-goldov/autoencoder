from typing import Optional

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score


def evaluate_classifier(
        classifier: nn.Module,
        encoder: nn.Module,
        test_loader: DataLoader,
        wandb_login: Optional[str] = None
):
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
