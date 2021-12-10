from typing import Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from src.data_processing.cifar_processing import get_train_data, get_test_data


class Cifar10Dataset(Dataset):
    def __init__(self, split: str):
        super().__init__()
        if split == 'train':
            self.data = get_train_data('../../data/cifar-10-batches-py')
        elif split == 'test':
            self.data = get_test_data("../../data/cifar-10-batches-py")
        else:
            raise ValueError(
                'Incorrect data split, available options are: train, test.'
            )
        self.split = split

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, int]]:
        """
        All images represented as 3072 vector where first 1024 integers are red, 1024 green, 1024 blue,
        so they are converted to torch.Tensor with shape (3, 32, 32)
        """

        vector, label = self.data['data'][index], self.data['labels'][index]
        vector = vector.astype(np.float32) / 255
        tensor = torch.from_numpy(vector).resize_(3, 32, 32)

        if self.split == 'train':
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=45),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            tensor = transform(tensor)
        return {'img': tensor, 'label': label}

    def __len__(self):
        return len(self.data['data'])
