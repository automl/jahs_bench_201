import json
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import Callable, Sequence, Optional

import numpy as np
import torch
from jahs_bench.tabular.lib.naslib.utils.utils import AttrDict, Cutout
from torchvision import datasets as dset, transforms as transforms

from jahs_bench.tabular.lib.core import constants as constants
from jahs_bench.tabular.lib.core.aug_lib import TrivialAugment

from icgen.vision_dataset import ICVisionDataset

"""
Adapted in large part from the original NASBench-201 code repository at
https://github.com/D-X-Y/AutoDL-Projects/tree/bc4c4692589e8ee7d6bab02603e69f8e5bd05edc
"""


def get_dataloaders(dataset: constants.Datasets, batch_size: int, cutout: int = -1, split: bool = True,
                    resolution: int = 1., trivial_augment=False, datadir: Path = None):

    if not isinstance(dataset, constants.Datasets):
        raise TypeError(f"A dataset name should be an instance of {constants.Datasets}, was given {type(dataset)}.")

    if resolution <= 0. or resolution > 1.:
        raise ValueError(f"Invalid image resolution scaling: {resolution}. Should be a value between 0. and 1.")

    dataset_fns = {**{
        constants.Datasets.cifar10: dset.CIFAR10,
        # constants.Datasets.fashionMNIST: dset.FashionMNIST,
    }, **{d: partial(load_icgen_dataset, name=d.name) for d in constants.icgen_datasets}}

    name_str, image_size, nchannels, nclasses, mean, std, train_size, test_size = dataset.value
    crop_size = image_size
    padding_size = max(0, int(4 * resolution))
    image_size = int(resolution * image_size)

    datadir = get_default_datadir() if datadir is None else datadir
    if dataset in constants.icgen_datasets:
        # Ugly hack. This data would automatically be read by ICVisionDataset but we need the mean/std values before
        # that object is initialized.
        crop_size = image_size
        datadir = datadir / "downsampled" / str(image_size)
        with open(datadir / dataset.name / "info.json") as fp:
            meta = json.load(fp)
        mean = [m / 255 for m in meta["mean_pixel_value_per_channel"]]
        std = [m / 255 for m in meta["mean_std_pixel_value_per_channel"]]

    if dataset not in dataset_fns:
        raise NotImplementedError(f"Pre-processing for dataset {name_str} has not yet been implemented.")

    # Data Augmentation
    lists = [TrivialAugmentTransform()] if trivial_augment else []
    lists += [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size, padding=padding_size)]
    # ICGen datasets have been resized already so we avoid downsampling twice.
    lists += [transforms.Resize(image_size)] if resolution < 1. and dataset not in constants.icgen_datasets else []
    lists += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    if cutout > 0 and not trivial_augment:  # Trivial Augment already contains Cutout
        lists += [Cutout(cutout)]

    train_transform = transforms.Compose(lists)
    test_transform = transforms.Compose(
        ([transforms.Resize(image_size)] if resolution < 1. and dataset not in constants.icgen_datasets else [])
        + [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    min_shape = (1, nchannels, image_size, image_size)
    train_data = dataset_fns[dataset](root=datadir, train=True, transform=train_transform, download=False)
    test_data = dataset_fns[dataset](root=datadir, train=False, transform=test_transform, download=False)

    assert len(train_data) == train_size, f"Invalid dataset configuration, expected {train_size} images, got " \
                                          f"{len(train_data)} for dataset {name_str}."
    assert len(test_data) == test_size, f"Invalid dataset configuration, expected {test_size} images, got " \
                                        f"{len(test_data)} for dataset {name_str}."

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_transform = test_data.transform

    loaders = {"test": test_loader}
    if split:
        ## Split original training data into a training and a validation set, use test data as a test set
        splitdir = datadir / dataset.name if dataset in constants.icgen_datasets else datadir
        split_info = load_splits(path=splitdir / f"{dataset.name}-validation-split.json")
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
        loaders["train"] = train_loader
        loaders["valid"] = valid_loader
    else:
        # data loader
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        loaders["train"] = train_loader

    return loaders, min_shape


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


def load_icgen_dataset(name: str, root: Path, train: bool = True,
                       transform: Optional[Sequence[Callable]] = None,
                       target_transform: Optional[Sequence[Callable]] = None,
                       download: bool = False) -> ICVisionDataset:
    """ Load an ICVisionDataset. This function serves as a wrapper around the underlying ICVisionDataset initializer
    in order to provide an interface compatible with calls to most Torchvision.Dataset classes. The 'download'
    parameter has been provided for compatibility only, the actual dataset should be downloaded and prepared in advance.
    """

    if download:
        raise UserWarning("The parameter 'download' has been provided for compatibility purposes only. The actual "
              "dataset should be downloaded and prepared in advance using ICGen.")

    dataset = ICVisionDataset(dataset=name, root=root,
                              split="train" if train else "test",
                              transform=transform, target_transform=target_transform)
    return dataset
