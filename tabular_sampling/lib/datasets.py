import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from naslib.utils.utils import AttrDict, Cutout
from torchvision import datasets as dset, transforms as transforms

from tabular_sampling.lib import constants as constants
from tabular_sampling.lib.aug_lib import TrivialAugment

"""
Adapted in large part from the original NASBench-201 code repository at 
https://github.com/D-X-Y/AutoDL-Projects/tree/bc4c4692589e8ee7d6bab02603e69f8e5bd05edc
"""


def get_dataloaders(dataset: constants.Datasets, batch_size: int, cutout: int = -1, split: bool = True,
                    resize: int = 0, trivial_augment=False, datadir: Path = None):
    datapath = get_default_datadir() if datadir is None else datadir
    train_data, test_data, min_shape, class_num = get_datasets(dataset, datapath, cutout=cutout, resize=resize,
                                                    trivial_augment=trivial_augment)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    train_transform = train_data.transform
    test_transform = test_data.transform

    ValLoaders = {"test": test_loader}
    if split:
        ## Split original training data into a training and a validation set, use test data as a test set
        assert dataset is constants.Datasets.cifar10, f"Only Cifar-10 supports validation set splits, cannot split " \
                                                      f"{dataset.value[0]}"
        split_info = load_splits(path=datapath / "cifar-split.json")
        assert len(train_data) == len(split_info.train) + len(split_info.valid), \
            f"invalid length : {len(train_data)} vs {len(split_info.train)} + {len(split_info.valid)}"
        valid_data = deepcopy(train_data)
        valid_data.transform = test_transform
        # data loader
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(split_info.train),
            num_workers=0,
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(split_info.valid),
            num_workers=0,
            pin_memory=True,
        )
        ValLoaders["train"] = train_loader
        ValLoaders["valid"] = valid_loader
    else:
        # data loader
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        ValLoaders["train"] = train_loader

    return ValLoaders, min_shape


def load_splits(path: Path):
    assert path.exists(), "Can not find {:}".format(path)
    # Reading data back
    with open(path, "r") as f:
        data = json.load(f)
    splits = {k: np.array(v[1], dtype=int) for k, v in data.items()}
    del data
    return AttrDict(splits)


def get_default_datadir() -> Path:
    return Path(__file__).parent.parent.parent / "data"


class TrivialAugmentTransform(torch.nn.Module):
    def __init__(self):
        self._apply_op = TrivialAugment()
        super(TrivialAugmentTransform, self).__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self._apply_op(img)


def get_datasets(name: constants.Datasets, root: Path, cutout: int, resize: int, trivial_augment=False):
    if not isinstance(name, constants.Datasets):
        raise TypeError(f"A dataset name should be an instance of {constants.Datasets}, was given {type(name)}.")

    dataset_fns = {
        constants.Datasets.cifar10: dset.CIFAR10,
        constants.Datasets.fashionMNIST: dset.FashionMNIST
    }

    name_str, image_size, nchannels, nclasses, mean, std, train_size, test_size = name.value

    if name not in dataset_fns:
        raise NotImplementedError(f"Pre-processing for dataset {name_str} has not yet been implemented.")

    # Data Augmentation
    lists = [TrivialAugmentTransform()] if trivial_augment else []
    lists += [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)]
    lists += [transforms.Resize(resize)] if resize else []
    lists += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    if cutout > 0 and not trivial_augment:  # Trivial Augment already contains Cutout
        lists += [Cutout(cutout)]
    train_transform = transforms.Compose(lists)
    test_transform = transforms.Compose(
        ([transforms.Resize(resize)] if resize else []) + [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    min_shape = (1, nchannels, image_size, image_size)
    train_data = dataset_fns[name](root, train=True, transform=train_transform, download=True)
    test_data = dataset_fns[name](root, train=False, transform=test_transform, download=True)
    assert len(train_data) == train_size, f"Invalid dataset configuration, expected {train_size} images, got " \
                                          f"{len(train_data)} for dataset {name_str}."
    assert len(test_data) == test_size, f"Invalid dataset configuration, expected {test_size} images, got " \
                                        f"{len(test_data)} for dataset {name_str}."

    return train_data, test_data, min_shape, nclasses