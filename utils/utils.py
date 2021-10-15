import logging
from pathlib import Path
import enum
import json
import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Union, Any, Optional, Iterable, Callable, List

import ConfigSpace
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from naslib.search_spaces.core.graph import Graph
from naslib.utils import utils as naslib_utils
from naslib.utils.utils import AttrDict, Cutout, AverageMeter
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


class DirectoryTree(object):
    """ A very simply class that translates a base directory into various relevant output directories. Useful for
    maintaining consistency across the code.
    The following directory structure is used by DirectoryTree:

    <base_dir>
    |--> <task_id>/
    |    |--> metrics/
    |    |    |--> <elapsed_runtime>.pkl.gz    <-- these are metric DataFrame files
    |    |--> models/
    |    |    |--> <model_idx>/
    |    |    |    |--> metrics/
    |    |    |    |    |--> <elapsed_runtime>.pkl.gz    <-- these are metric DataFrame files
    |    |    |    |--> checkpoints/
    |    |    |    |    |--> <elapsed_runtime>.pt    <-- these are checkpoint files
    """

    def __init__(self, basedir: Path, taskid: Optional[int], model_idx: Optional[int] = None):
        self.basedir = basedir.resolve()
        assert self.basedir.exists() and self.basedir.is_dir(), \
            f"The base directory and its parent directory tree must be created beforehand. Given base directory " \
            f"either does not exist or is not a directory: {str(self.basedir)}"
        self.taskid = taskid
        self.model_idx = model_idx

    @property
    def taskid(self) -> int:
        return self._taskid

    @taskid.setter
    def taskid(self, new_id: Optional[int]):
        if new_id is not None:
            assert isinstance(new_id, int), f"Task ID must be an integer, was given {type(new_id)}"
            self._taskid = new_id
            self.task_dir.mkdir(exist_ok=True, parents=True)
            self.task_metrics_dir.mkdir(exist_ok=True, parents=True)
        else:
            self._taskid = None

    @property
    def model_idx(self) -> int:
        return self._model_idx

    @model_idx.setter
    def model_idx(self, new_index: Optional[int]):
        if new_index is not None:
            assert isinstance(new_index, int), f"Model index must be an integer, was given {type(new_index)}"
            self._model_idx = new_index
            self.model_dir.mkdir(exist_ok=True, parents=True)
            self.model_metrics_dir.mkdir(exist_ok=True, parents=True)
            self.model_checkpoints_dir.mkdir(exist_ok=True, parents=True)
        else:
            self._model_idx = None

    @property
    def task_dir(self) -> Path:
        return self.basedir / str(self.taskid)

    @property
    def task_metrics_dir(self) -> Path:
        return self.task_dir / "metrics"

    @property
    def model_dir(self) -> Path:
        assert self.model_idx is not None, \
            "A valid model index needs to be set before the relevant model level subtree can be accessed."
        return self.task_dir / "models" / str(self.model_idx)

    @property
    def model_metrics_dir(self) -> Path:
        return self.model_dir / "metrics"

    @property
    def model_checkpoints_dir(self) -> Path:
        return self.model_dir / "checkpoints"


class Checkpointer(object):
    """
    Essentially a stateful-function factory for checkpointing model training at discrete intervals of time and epochs.
    Initialized with references to all necessary objects and data for checkpointing and called by specifying the
    elapsed runtime and number of epochs. However, it can also be used to load existing checkpoints. Consult
    DirectoryTree for a breakdown of how the file structure is organized.
    """

    def __init__(self, model: Graph, optimizer: torch.optim.Optimizer, scheduler: CosineAnnealingLR,
                 interval_seconds: int, interval_epochs: int, dir_tree: DirectoryTree, logger: logging.Logger = None,
                 map_location=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.interval_seconds = interval_seconds
        self.interval_epochs = interval_epochs
        self.dir_tree = dir_tree
        self.runtime = 0.
        self.elapsed_epochs = -1
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        logger.debug(f"Successfully initialized checkpointer.")
        self._load_latest(map_location)

    def __call__(self, runtime: float, elapsed_epochs: int, force_checkpoint: bool = False):
        if (runtime - self.runtime) >= self.interval_seconds \
                or (elapsed_epochs - self.elapsed_epochs) >= self.interval_epochs or force_checkpoint:
            output_file = self.dir_tree.model_checkpoints_dir / f"{runtime}.pt"
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "epochs": elapsed_epochs,
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
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
            return float(f.stem)

        latest = max(
            self.dir_tree.model_checkpoints_dir.rglob("*.pt"),
            key=lambda f: extract_runtime(f),
            default=None
        )

        if latest is None:
            self.logger.info(f"No valid checkpoints found at {self.dir_tree.model_checkpoints_dir}.")
            return

        state_dicts = torch.load(latest, map_location=map_location)
        self.model.load_state_dict(state_dicts["model_state_dict"])
        self.optimizer.load_state_dict(state_dicts["optimizer_state_dict"])
        self.scheduler.load_state_dict(state_dicts["scheduler_state_dict"])
        torch.set_rng_state(state_dicts["torch_rng_state"])
        np.random.set_state(state_dicts["numpy_rng_state"])
        random.setstate(state_dicts["python_rng_state"])
        self.elapsed_epochs = state_dicts["epochs"]
        self.runtime = extract_runtime(latest)
        self.logger.info(f"Loaded existing checkpoint from {str(latest)}")


class MetricLogger(object):
    """
    Holds a set of metrics, information about where they are to be logged, the frequency at which they should be
    logged as well as some functionality to convert any given set of metrics into a pandas DataFrame. Consult
    DirectoryTree for a breakdown of how the file structure is organized.
    """

    @enum.unique
    class MetricSet(enum.Enum):
        """ Defines uniquely recognized sets of metrics that are handled slightly differently from the norm. """
        model = enum.auto()
        task = enum.auto()
        default = enum.auto()

    def __init__(self, dir_tree: DirectoryTree, metrics: dict, log_interval: Optional[int] = None,
                 set_type: MetricSet = MetricSet.default, logger: logging.Logger = None):
        self.dir_tree = dir_tree
        self.metrics = metrics
        self.log_interval = log_interval
        self.set_type = set_type
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.elapsed_runtime = 0.
        self.resume_latest_saved_metrics()

    @classmethod
    def _nested_dict_to_df(cls, nested: Union[dict, Any]) -> pd.DataFrame:
        """ Given a nested dict of dicts, use each level of the nested keys as column index levels in a pandas
        MultiIndex and convert the entire data structure into a pandas DataFrame. Note that any item that is not a dict
        will be considered to be a leaf node in the nested dict structure, therefore, any iterables will be directly
        converted into columns. Care should be taken to make sure that such iterables are of an appropriate length.
        Single data items at each level will only be copied over to as many indices as the length of the longest
        iterable at that level. """

        assert isinstance(nested, dict), f"Expected input to be a possibly nested dictionary, received {type(nested)}."

        index = None
        new_df = {}
        singles = {}
        for k, v in nested.items():
            if isinstance(v, dict):
                new_df[k] = cls._nested_dict_to_df(v)
            elif isinstance(v, Iterable):
                new_df[k] = pd.Series(v)
            else:
                singles[k] = v
                continue
            if index is None:
                index = new_df[k].index
                # new_df = pd.DataFrame(index=vdf.index)
        if new_df:
            new_df = pd.concat(new_df, axis=1)  # Create new level in columns MultiIndex or promote Index to MultiIndex
            new_df = new_df.assign(**singles)
        else:
            new_df = pd.DataFrame(singles, index=[0])

        return new_df

    @classmethod
    def _df_to_nested_dict(cls, df: pd.DataFrame) -> dict:
        """ Given a DataFrame, attempts to convert it into a nested dict. This is intended to be the inverse of
        _nested_dict_to_df() and therefore only intended to work with DataFrame objects that were originally created by
        that function. """

        metrics_dict = AttrDict()
        for k in df.columns.unique(0):
            if isinstance(df[k], pd.Series):
                metrics_dict[k] = df[k].to_list()
            else:
                metrics_dict[k] = cls._df_to_nested_dict(df[k])

        return metrics_dict

    @classmethod
    def _handle_model_metric_dict(cls, metrics: Dict) -> pd.DataFrame:
        df = cls._nested_dict_to_df(metrics)
        assert isinstance(df, pd.DataFrame), f"This function should only be used with a DataFrame, not {type(df)}"
        assert isinstance(df.index, pd.Index), \
            f"Expected the model-level metrics DataFrame index to be an Index, was {type(df.index)}"
        assert isinstance(df.columns, pd.MultiIndex), \
            f"Expected the model-level metrics DataFrame columns to be a MultiIndex, not {type(df.columns)}"
        assert df.columns.nlevels == 2, \
            f"Expected the model-level metrics DataFrame to have exactly 2 levels in the columns MultiIndex, got " \
            f"{df.columns.nlevels}"

        df.index = df.index.set_names(["Epoch"]) + 1
        df.columns.set_names(["MetricType", "MetricName"], inplace=True)
        return df

    @classmethod
    def _handle_task_metric_dict(cls, metrics: Dict) -> pd.DataFrame:
        model_configs = pd.DataFrame(metrics["model_config"]).to_dict()
        model_configs = {k: list(v.values()) for k, v in model_configs.items()} # Convert list of dicts to dict of lists
        df = cls._nested_dict_to_df(metrics | {"model_config": model_configs})

        assert isinstance(df, pd.DataFrame), f"This function should only be used with a DataFrame, not {type(df)}"
        assert isinstance(df.index, pd.Index), \
            f"Expected the task-level metrics DataFrame index to be an Index, was {type(df.index)}"
        assert isinstance(df.columns, pd.MultiIndex), \
            f"Expected the task-level metrics DataFrame columns to be a MultiIndex, not {type(df.columns)}"
        assert df.columns.nlevels == 2, \
            f"Expected the task-level metrics DataFrame to have exactly 2 levels in the columns MultiIndex, got " \
            f"{df.columns.nlevels}"

        return df

    @property
    def _metric_set_dict_handlers(self):
        return {
            MetricLogger.MetricSet.model: self._handle_model_metric_dict,
            MetricLogger.MetricSet.task: self._handle_task_metric_dict,
            MetricLogger.MetricSet.default: self._nested_dict_to_df
        }

    @classmethod
    def _handle_task_metric_df(cls, df: pd.DataFrame) -> dict:
        metric_dict = AttrDict({
            "model_idx": df["model_idx"].values.reshape(-1).tolist(),
            "model_config": list(df["model_config"].transpose().to_dict().values()),
            "global_seed": df["global_seed"].values.reshape(-1).tolist(),
        })

        return metric_dict

    @property
    def _metric_set_df_handlers(self):
        return {
            MetricLogger.MetricSet.model: self._df_to_nested_dict,
            MetricLogger.MetricSet.task: self._handle_task_metric_df,
            MetricLogger.MetricSet.default: self._df_to_nested_dict
        }

    def _generate_df(self) -> pd.DataFrame:
        """ Converts the currently stored dict of metrics into a Pandas DataFrame ready for being saved to disk. """

        # metric_df = self._nested_dict_to_df(self.metrics)
        metric_df = self._metric_set_dict_handlers[self.set_type](self.metrics)
        return metric_df

    @property
    def _metric_set_to_log_directory(self):
        return {
            MetricLogger.MetricSet.model: "model_metrics_dir",
            MetricLogger.MetricSet.task: "task_metrics_dir",
            MetricLogger.MetricSet.default: "basedir"
        }

    @property
    def log_directory(self):
        return getattr(self.dir_tree, self._metric_set_to_log_directory[self.set_type])

    def log(self, elapsed_runtime: float, force: bool = False, where: Optional[Path] = None):
        if (self.log_interval is not None and elapsed_runtime - self.elapsed_runtime > self.log_interval) or force:
            df = self._generate_df()
            pth = where if where is not None else self.log_directory
            df.to_pickle(pth / f"{elapsed_runtime}.pkl.gz")
            self.elapsed_runtime = elapsed_runtime
            self.logger.info(f"Logged metrics at {str(pth)}")

    def resume_latest_saved_metrics(self, where: Optional[Path] = None):
        """ Used in conjunction with the corresponding method of Checkpointer to resume model training and evaluation
        from where it was left off. """

        def extract_runtime(f: Path) -> float:
            return float(f.name.rstrip(".pkl.gz"))

        pth = where if where is not None else self.log_directory
        latest = max(
            pth.rglob("*.pkl.gz"),
            key=lambda f: extract_runtime(f),
            default=None
        )

        if latest is None:
            self.logger.info(f"No valid pre-existing metric DataFrames found at {str(pth)}.")
            return

        metric_df = pd.read_pickle(latest)
        self.metrics |= self._metric_set_df_handlers[self.set_type](metric_df)
        self.elapsed_runtime = extract_runtime(latest)


def get_common_metrics(extra_metrics: Optional[List[str]] = None, template: Callable = AverageMeter) -> AttrDict:
    """ Convenience function for generating a dictionary with the most commonly used metrics as keys. The initial value
    of these keys can optionally be set by passing an object factory to the parameter 'template', which defaults to
    naslib.utils.utils.AverageMeter. """
    metrics = AttrDict(
        duration=template(),
        forward_duration=template(),
        data_load_duration=template(),
        loss=template(),
        acc=template(),
        **({m: template() for m in extra_metrics} if extra_metrics else {})
    )
    return metrics


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
    del (data)
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
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), ] + \
                ([transforms.Resize(resize)] if resize else []) + \
                [transforms.ToTensor(), transforms.Normalize(mean, std), ]
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
