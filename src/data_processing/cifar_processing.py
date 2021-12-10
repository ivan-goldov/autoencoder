import pickle
from os import listdir
from os.path import join
from typing import Dict, Any


def unpickle(file_path: str) -> Dict[Any, Any]:
    with open(file_path, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def get_train_data(folder_path: str) -> Dict[int, Any]:
    train_paths = [
        join(folder_path, f) for f in listdir(folder_path) if f.startswith('data_batch')
    ]

    train_dicts = [unpickle(f) for f in train_paths]
    convenient_train_dict = {0: [], 1: []}
    for d in train_dicts:
        convenient_train_dict[0].extend(d[b'data'])
        convenient_train_dict[0].extend(d[b'labels'])
    return convenient_train_dict


def get_test_data(folder_path: str) -> Dict[int, Any]:
    test_batch = join(folder_path, 'test_batch')
    test_dict = unpickle(test_batch)
    convenient_test_dict = {0: test_dict[b'data'], 1: test_dict[b'labels']}
    return convenient_test_dict
