import logging
import time
from pathlib import Path
import json
import enum
from typing import Dict, Union, Any, Optional, Tuple
import ConfigSpace
import torch
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from copy import deepcopy
from naslib.utils import utils as naslib_utils, logging as naslib_logging
from naslib.utils.utils import AttrDict, Cutout
from naslib.search_spaces.core.graph import Graph
from .custom_nasb201_code import CosineAnnealingLR

def _query_config(config: Union[ConfigSpace.Configuration, Dict], param: str, default: Optional[Any] = None) -> Any:
    """ Query the given 'config' object for the parameter named 'param'. If the parameter is not found, returns default
    if given else None. This is necessary entirely because, as of the writing of this code, the 'default' argument of
     ConfigSpace.Configuration.get(), ConfigSpace v. 0.4.19, does not work as intended. """

    config = config.get_dictionary() if isinstance(config, ConfigSpace.Configuration) else config
    return config.get(param, default)


def _init_adam(model, config: Union[ConfigSpace.Configuration, Dict]):
    lr, weight_decay = _query_config(config, "learning_rate"), _query_config(config, "weight_decay")
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optim


def _init_adamw(model, config: Union[ConfigSpace.Configuration, Dict]):
    lr, weight_decay = _query_config(config, "learning_rate"), _query_config(config, "weight_decay")
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optim


def _init_sgd(model, config: Union[ConfigSpace.Configuration, Dict]):
    lr, momentum, weight_decay = _query_config(config, "learning_rate"), _query_config(config, "momentum", 0.9), \
                                 _query_config(config, "weight_decay")
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    return optim


@enum.unique
class optimizers(enum.Enum):
    def __init__(self, constructor):
        self.constructor = constructor

    def construct(self, *args, **kwargs):
        return self.constructor(*args, **kwargs)

    SGD = _init_sgd,
    Adam = _init_adam,
    AdamW = _init_adamw,


class Checkpointer(object):
    """ Essentially a stateful-function factory for checkpointing model training at discrete intervals of time and
    epochs. Initialized with references to all necessary objects and data for checkpointing and called by specifying
    the elapsed runtime and number of epochs. However, it can also be used to load existing checkpoints. """

    def __init__(self, model: Graph, optimizer: torch.optim.Optimizer, scheduler: CosineAnnealingLR,
                 interval_seconds: int, interval_epochs: int, outdir: Path, taskid: int, model_idx: int, logger=None,
                 map_location=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.interval_seconds = interval_seconds
        self.interval_epochs = interval_epochs
        self.outdir = outdir
        self.taskid = taskid
        self.model_idx = model_idx
        self.runtime = 0.
        self.elapsed_epochs = -1
        self.logger = logger
        assert outdir.exists() and outdir.is_dir(), \
            f"The Checkpointer does not assume responsibility for creating the relevant directory tree for the " \
            f"output file. This must be done before passing in the directory. Specified directory does not exist or " \
            f"is not a directory: {outdir}"
        naslib_logging.log_first_n(
            logging.DEBUG,
            f"Successfully initialized checkpointer for taskid {taskid}, model index {model_idx}.",
            3,
            name=logger.name
        )
        self._load_latest(map_location)


    @property
    def output_file_basename(self):
        return f"{self.taskid}_{self.model_idx}"


    def __call__(self, runtime: float, elapsed_epochs: int, force_checkpoint: bool = False):
        if (runtime - self.runtime) >= self.interval_seconds \
            or (elapsed_epochs - self.elapsed_epochs) >= self.interval_epochs or force_checkpoint:
            output_file = self.outdir / f"{self.output_file_basename}_{runtime}.pt"
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "epochs": elapsed_epochs,
                },
                output_file
            )
            self.runtime = runtime
            self.elapsed_epochs = elapsed_epochs
            self.logger.debug(f"Checkpointed model training at f{str(output_file)}")

    def _load_latest(self, map_location: Optional[Any] = None):
        """ Attempts to load a previously saved checkpoint in order to resume model training. If a checkpoint is
        successfully loaded, the model, optimizer and scheduler state dictionaries are appropriately loaded and the
        relevant values of runtime and elapsed_epochs are updated, else, the state dictionaries and other object
        attributes are left untouched. """

        def extract_runtime(f: Path) -> float:
            return float(f.stem.split("_")[2])

        latest = max(
            self.outdir.rglob(f"{self.output_file_basename}_*.pt"),
            key=lambda f: extract_runtime(f),
            default=None
        )

        if latest is None:
            self.logger.info(f"No valid checkpoints for taskid {self.taskid}, model index {self.model_idx} found.")
            return

        state_dicts = torch.load(latest, map_location=map_location)
        self.model.load_state_dict(state_dicts["model_state_dict"])
        self.optimizer.load_state_dict(state_dicts["optimizer_state_dict"])
        self.scheduler.load_state_dict(state_dicts["scheduler_state_dict"])
        self.elapsed_epochs = state_dicts["epochs"]
        self.runtime = extract_runtime(latest)
        self.logger.info(f"Loaded existing checkpoint from {str(latest)}")



"""
Adapted in large part from the original NASBench-201 code repository at 
https://github.com/D-X-Y/AutoDL-Projects/tree/bc4c4692589e8ee7d6bab02603e69f8e5bd05edc
"""

Dataset2Class = {
    "cifar10": 10,
    "cifar100": 100,
    # "imagenet-1k-s": 1000,
    # "imagenet-1k": 1000,
    # "ImageNet16": 1000,
    # "ImageNet16-150": 150,
    # "ImageNet16-120": 120,
    # "ImageNet16-200": 200,
}


def load_splits(path: Path):
    assert path.exists(), "Can not find {:}".format(path)
    # Reading data back
    with open(path, "r") as f:
        data = json.load(f)
    splits = {k: np.array(v[1], dtype=int) for k, v in data.items()}
    del(data)
    return AttrDict(splits)


def get_dataloaders(dataset, batch_size, cutout: float = -1., split: bool = True, resize: int = 0):
    datapath = naslib_utils.get_project_root() / "data"
    train_data, test_data, xshape, class_num = get_datasets(dataset, datapath, cutout=cutout, resize=resize)
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
        assert dataset == "cifar10"
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

    return ValLoaders, train_transform, test_transform


def get_datasets(name, root, cutout, resize):

    if name == "cifar10":
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == "cifar100":
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    # elif name.startswith("imagenet-1k"):
    #     mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # elif name.startswith("ImageNet16"):
    #     mean = [x / 255 for x in [122.68, 116.66, 104.01]]
    #     std = [x / 255 for x in [63.22, 61.26, 65.09]]
    else:
        raise TypeError("Unknown dataset : {:}".format(name))

    # Data Argumentation
    if name == "cifar10" or name == "cifar100":
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),] + \
                ([transforms.Resize(resize)] if resize else []) + \
                [transforms.ToTensor(), transforms.Normalize(mean, std),]
        if cutout > 0:
            lists += [Cutout(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            ([transforms.Resize(resize)] if resize else []) + [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        xshape = (1, 3, 32, 32)
    # elif name.startswith("ImageNet16"):
    #     lists = [
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(16, padding=2),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std),
    #     ]
    #     if cutout > 0:
    #         lists += [CUTOUT(cutout)]
    #     train_transform = transforms.Compose(lists)
    #     test_transform = transforms.Compose(
    #         [transforms.ToTensor(), transforms.Normalize(mean, std)]
    #     )
    #     xshape = (1, 3, 16, 16)
    # elif name == "tiered":
    #     lists = [
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(80, padding=4),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std),
    #     ]
    #     if cutout > 0:
    #         lists += [CUTOUT(cutout)]
    #     train_transform = transforms.Compose(lists)
    #     test_transform = transforms.Compose(
    #         [
    #             transforms.CenterCrop(80),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean, std),
    #         ]
    #     )
    #     xshape = (1, 3, 32, 32)
    # elif name.startswith("imagenet-1k"):
    #     normalize = transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #     )
    #     if name == "imagenet-1k":
    #         xlists = [transforms.RandomResizedCrop(224)]
    #         xlists.append(
    #             transforms.ColorJitter(
    #                 brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
    #             )
    #         )
    #         xlists.append(Lighting(0.1))
    #     elif name == "imagenet-1k-s":
    #         xlists = [transforms.RandomResizedCrop(224, scale=(0.2, 1.0))]
    #     else:
    #         raise ValueError("invalid name : {:}".format(name))
    #     xlists.append(transforms.RandomHorizontalFlip(p=0.5))
    #     xlists.append(transforms.ToTensor())
    #     xlists.append(normalize)
    #     train_transform = transforms.Compose(xlists)
    #     test_transform = transforms.Compose(
    #         [
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             normalize,
    #         ]
    #     )
    #     xshape = (1, 3, 224, 224)
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if name == "cifar10":
        train_data = dset.CIFAR10(
            root, train=True, transform=train_transform, download=True
        )
        test_data = dset.CIFAR10(
            root, train=False, transform=test_transform, download=True
        )
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == "cifar100":
        train_data = dset.CIFAR100(
            root, train=True, transform=train_transform, download=True
        )
        test_data = dset.CIFAR100(
            root, train=False, transform=test_transform, download=True
        )
        assert len(train_data) == 50000 and len(test_data) == 10000
    # elif name.startswith("imagenet-1k"):
    #     train_data = dset.ImageFolder(osp.join(root, "train"), train_transform)
    #     test_data = dset.ImageFolder(osp.join(root, "val"), test_transform)
    #     assert (
    #         len(train_data) == 1281167 and len(test_data) == 50000
    #     ), "invalid number of images : {:} & {:} vs {:} & {:}".format(
    #         len(train_data), len(test_data), 1281167, 50000
    #     )
    # elif name == "ImageNet16":
    #     train_data = ImageNet16(root, True, train_transform)
    #     test_data = ImageNet16(root, False, test_transform)
    #     assert len(train_data) == 1281167 and len(test_data) == 50000
    # elif name == "ImageNet16-120":
    #     train_data = ImageNet16(root, True, train_transform, 120)
    #     test_data = ImageNet16(root, False, test_transform, 120)
    #     assert len(train_data) == 151700 and len(test_data) == 6000
    # elif name == "ImageNet16-150":
    #     train_data = ImageNet16(root, True, train_transform, 150)
    #     test_data = ImageNet16(root, False, test_transform, 150)
    #     assert len(train_data) == 190272 and len(test_data) == 7500
    # elif name == "ImageNet16-200":
    #     train_data = ImageNet16(root, True, train_transform, 200)
    #     test_data = ImageNet16(root, False, test_transform, 200)
    #     assert len(train_data) == 254775 and len(test_data) == 10000
    else:
        raise TypeError("Unknown dataset : {:}".format(name))

    class_num = Dataset2Class[name]
    return train_data, test_data, xshape, class_num
