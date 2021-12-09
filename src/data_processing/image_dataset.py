from typing import Dict, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from cifar_processing import get_train_data, get_test_data
from src.modules.autoencoder import AutoEncoder


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
        return len(self.data)

# def main():
#     t = Cifar10Dataset("train")
#     dataloader = DataLoader(dataset=t, batch_size=1)
#     batch = next(iter(dataloader))['img']
#     # plt.imshow(t.permute(1, 2, 0))
#     # plt.show()
#     model = AutoEncoder()
#     # model = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, stride=2)
#     res = model.encode(batch)
#     # plt.imshow(res.permute(1, 2, 0))
#     print(res.shape)
#     res2 = model.decode(res)
#     print(res2.shape)
#
#
# if __name__ == '__main__':
#     main()
