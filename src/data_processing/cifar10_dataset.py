from typing import Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from src.data_processing.cifar_processing import get_train_data, get_test_data


class Cifar10Dataset(Dataset):
    __label_to_str = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }

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

    def __getitem__(self, index: int) -> Dict[int, Union[torch.Tensor, int]]:
        """
        All images represented as 3072 vector where first 1024 integers are red, 1024 green, 1024 blue,
        so they are converted to torch.Tensor with shape (3, 32, 32)
        """

        vector, label = self.data[0][index], self.data[1][index]
        vector = vector.astype(np.float32) / 255
        tensor = torch.from_numpy(vector).resize_(3, 32, 32)

        transform = transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                         (0.24703233, 0.24348505, 0.26158768))
        tensor = transform(tensor)
        return {0: tensor, 1: label}

    @classmethod
    def label_to_str(cls, label: int) -> str:
        return cls.__label_to_str[label]

    def __len__(self):
        return len(self.data[0])
