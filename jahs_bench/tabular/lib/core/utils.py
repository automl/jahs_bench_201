import enum
import logging
import random
from itertools import repeat, cycle
from pathlib import Path
from typing import Dict, Union, Any, Optional, Iterable, Callable, List, Tuple, Hashable

import ConfigSpace
import jahs_bench.tabular.lib.naslib.utils.utils as naslib_utils
import numpy as np
import pandas as pd
import torch

from jahs_bench.tabular.lib.naslib.search_spaces.core.graph import Graph
from jahs_bench.tabular.lib.naslib.utils.utils import AttrDict, AverageMeter

import jahs_bench.tabular.lib.core.constants as constants
from jahs_bench.tabular.search_space import NASB201HPOSearchSpace
from jahs_bench.tabular.lib.core.custom_nasb201_code import CosineAnnealingLR


def get_training_config_help() -> dict:
    return {k: v["help"] for k, v in constants.training_config.items()}


def _query_config(config: Union[ConfigSpace.Configuration, Dict], param: str, default: Optional[Any] = None) -> Any:
    """ Query the given 'config' object for the parameter named 'param'. If the parameter is not found, returns default
    if given else None. This is necessary entirely because, as of the writing of this code, the 'default' argument of
     ConfigSpace.Configuration.get(), ConfigSpace v. 0.4.19, does not work as intended. """

    config = config.get_dictionary() if isinstance(config, ConfigSpace.Configuration) else config
    return config.get(param, default)


def _init_adam(model, config: Union[ConfigSpace.Configuration, Dict]):
    lr, weight_decay = _query_config(config, "LearningRate"), _query_config(config, "WeightDecay")
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optim


def _init_adamw(model, config: Union[ConfigSpace.Configuration, Dict]):
    lr, weight_decay = _query_config(config, "LearningRate"), _query_config(config, "WeightDecay")
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optim


def _init_sgd(model, config: Union[ConfigSpace.Configuration, Dict]):
    lr, momentum, weight_decay = _query_config(config, "LearningRate"), _query_config(config, "Momentum", 0.9), \
                                 _query_config(config, "WeightDecay")
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    return optim


@enum.unique
class Optimizers(enum.Enum):
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

    def __init__(self, basedir: Path, taskid: Optional[int] = None, model_idx: Optional[int] = None,
                 read_only: bool = False):
        """
        The DirectoryTree enables a fixed, pre-determined directory sub-tree to be made and easily traversed at the
        given base directory, 'basedir'. 'taskid' and 'model_idx' are optional integers that can be changed and
        reassigned any time in order to generate the respective branch of the tree. Setting 'read_only' to True allows
        the tree to assume that no new directories need to be created if they don't exist and raise an Error if such an
        attempt to access such directory is made.
        """
        self.basedir = basedir.resolve()
        assert self.basedir.exists() and self.basedir.is_dir(), \
            f"The base directory and its parent directory tree must be created beforehand. Given base directory " \
            f"either does not exist or is not a directory: {str(self.basedir)}"
        self.read_only = read_only
        # Setting these to int values also creates the appropriate directories if read_only is False
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
            if self.read_only:
                return
            if not self.task_dir.exists():
                self.task_dir.mkdir(parents=True)
            if not self.task_metrics_dir.exists():
                self.task_metrics_dir.mkdir(parents=True)
            if not self.task_models_dir.exists():
                self.task_models_dir.mkdir(parents=True)
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
            if self.read_only:
                return
            subdirs = [self.model_dir, self.model_metrics_dir, self.model_checkpoints_dir, self.model_tensorboard_dir]
            for subdir in subdirs:
                if not subdir.exists():
                    subdir.mkdir(parents=True)
        else:
            self._model_idx = None

    @property
    def task_dir(self) -> Path:
        return self.basedir / str(self.taskid)

    @property
    def task_config_file(self) -> Path:
        return self.task_dir / "task_config.json"

    @property
    def task_metrics_dir(self) -> Path:
        return self.task_dir / "metrics"

    @property
    def task_models_dir(self) -> Path:
        return self.task_dir / "models"

    @property
    def model_dir(self) -> Path:
        assert self.model_idx is not None, \
            "A valid model index needs to be set before the relevant model level subtree can be accessed."
        return self.task_models_dir / str(self.model_idx)

    @property
    def model_metrics_dir(self) -> Path:
        return self.model_dir / "metrics"

    @property
    def model_checkpoints_dir(self) -> Path:
        return self.model_dir / "checkpoints"

    @property
    def model_tensorboard_dir(self) -> Path:
        return self.model_dir / "tensorboard_logs"

    @property
    def model_error_description_file(self) -> Path:
        return self.model_dir / f"error_description.json"

    @property
    def existing_tasks(self) -> Optional[List[Path]]:
        return None if not self.basedir.exists() else [d for d in self.basedir.iterdir() if d.is_dir()]

    @property
    def existing_models(self) -> List[Path]:
        return None if not self.task_models_dir.exists() else [d for d in self.task_models_dir.iterdir() if d.is_dir()]


class SynchroTimer:
    """ A very simply class whose sole purpose is to provide a share-able object and interface so that multiple other
    pieces of code (such as Checkpointer and MetricLogger instances) can consult the same timing related logic. It
    handles the preparation of a signal based on the configuration of time/epoch frequency. Listeners that have been
    registered with a SynchroTimer object can then query it anytime to know if the signal has been prepared. This is
    by no means intended to be used for multi-process signal passing! """

    previous_timestamp: float
    last_epoch: int
    ping_interval_time: float
    ping_interval_epochs: float
    _signal: bool
    _listeners: Dict[Any, bool]

    def __init__(self, ping_interval_time: float = None, ping_interval_epochs: float = None):
        """ Setting either of the two interval types to None disables updates to the signal logic based on that
        property. It is thus possible to create a SynchroTimer object which can only generate signals at arbitrary
        intervals when manually asked to do so (see SynchroTimer.update()). """
        self.ping_interval_time = ping_interval_time
        self.ping_interval_epochs = ping_interval_epochs
        self.previous_timestamp = 0.
        self.last_epoch = -1
        self._listeners = {}
        self._signal = False

    def adjust(self, previous_timestamp: Optional[float] = None, last_epoch: Optional[int] = None):
        """ Sets the elapsed time and epochs without causing a signal to be generated. """
        self.previous_timestamp = previous_timestamp
        self.last_epoch = last_epoch

    def register_new_listener(self) -> Hashable:
        """ Allows SynchroTimer to know how many access points are listening to its signals and, in turn, ensure that
        each listener receives each unique instance of a signal only once. """
        id = len(self._listeners.keys()) + 1
        self._listeners[id] = self._signal
        return id

    def _update_signals_(self):
        """ Prepares signals for all _listeners. """
        for l in self._listeners.keys():
            self._listeners[l] = self._signal

    def update(self, timestamp: Optional[float] = None, curr_epoch: Optional[int] = None, force: Optional[bool] = False):
        """ Given an input of time/epochs or an overriding flag 'force', determines whether a signal should be
        prepared for the listeners or not. Even if ping_interval_time or ping_interval_epoch was set to None,
        the time and epochs can still be specified here to keep track of them. In such a case, however, they will not
        cause the timer's signal to be set. """

        time_signal = False if self.ping_interval_time is None \
            else (timestamp - self.previous_timestamp) >= self.ping_interval_time

        epoch_signal = False if self.ping_interval_epochs is None \
            else (curr_epoch - self.last_epoch) >= self.ping_interval_epochs

        self._signal = time_signal or epoch_signal or force
        if self._signal:
            self._update_signals_()
            self.previous_timestamp = timestamp
            self.last_epoch = curr_epoch

    def ping(self, id: Hashable):
        """ Any listener can call this function along with its registered ID to query the signal. """

        if id not in self._listeners:
            raise KeyError(f"Unrecognized listener ID {id}.")
        signal = self._listeners[id]

        # Either the signal was already False and remains unchanged, or was True and has now been consumed.
        self._listeners[id] = False
        return signal

class Checkpointer(object):
    """
    Essentially a stateful-function factory for checkpointing model training at discrete intervals of time and epochs.
    Initialized with references to all necessary objects and data for checkpointing and called by specifying the
    elapsed runtime and number of epochs. However, it can also be used to load existing checkpoints. Consult
    DirectoryTree for a breakdown of how the file structure is organized. A SynchroTimer object needs to be passed to
    the initializer argument 'timer' in order to enable logging functionality. Without it, a Checkpointer object can
    only be used in 'read-only' mode.
    """

    def __init__(self, model: Graph, optimizer: torch.optim.Optimizer, scheduler: CosineAnnealingLR,
                 dir_tree: DirectoryTree, timer: Optional[SynchroTimer] = None, logger: logging.Logger = None,
                 map_location=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dir_tree = dir_tree
        self.timer = timer
        self.runtime = 0. if self.timer is None else self.timer.previous_timestamp
        self.last_epoch = -1 if self.timer is None else self.timer.last_epoch
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        logger.debug(f"Successfully initialized checkpointer.")
        self._load_latest(map_location)

    def __call__(self, force_checkpoint: bool = False):
        if self.timer is None:
            raise RuntimeError("The log functionality of Checkpointer is enabled only when a valid timer has been set.")
        if self._signal or force_checkpoint:
            runtime = self.timer.previous_timestamp
            last_epoch = self.timer.last_epoch
            output_file = self.dir_tree.model_checkpoints_dir / f"{runtime}.pt"
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "epochs": last_epoch,
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
                },
                output_file
            )
            self.runtime = runtime
            self.last_epoch = last_epoch
            self.logger.debug(f"Checkpointed model training at f{str(output_file)}")

    @property
    def timer(self) -> SynchroTimer:
        return self._timer

    @timer.setter
    def timer(self, new_timer: SynchroTimer):
        if new_timer is None:
            self._timer = None
        elif not isinstance(new_timer, SynchroTimer):
            raise ValueError(f"Checkpointer requires a valid instance of SynchroTimer, cannot set timer to "
                             f"{type(new_timer)}.")
        else:
            self._timer = new_timer
            self._timer_id = self._timer.register_new_listener()

    @property
    def _signal(self):
        return False if self.timer is None else self.timer.ping(self._timer_id)

    @classmethod
    def _extract_runtime_from_filename(cls, f: Path) -> float:
        return float(f.name.rstrip(".pt"))

    @classmethod
    def _get_sorted_chkpt_paths(cls, pth: Path, ascending: bool = True) -> List[Path]:
        latest = sorted(
            pth.rglob("*.pt"),
            key=lambda f: cls._extract_runtime_from_filename(f),
            reverse=not ascending
        )
        return latest

    @classmethod
    def _load_checkpoint(cls, pths: Union[Path, List[Path]], safe_load: Optional[bool] = True,
                         map_location: Optional[Any] = None, get_path: bool = False) -> \
            Union[Optional[dict], Tuple[Optional[dict], Path]]:
        """ Attempts to load a checkpoint from either a single Path or a list of Paths, attempting each one in the
        order they occur. If 'safe_load' is set to False, the first checkpoint in the list that is readable from disk
        is loaded. It is set to True by default - i.e. if the first checkpoint cannot be loaded, an error is raised.
        'map_location' has been provided for compatibility with GPUs. """

        if isinstance(pths, Path):
            pths = [pths]

        state_dicts = None
        pth = None
        for pth in pths:
            try:
                state_dicts = torch.load(pth, map_location=map_location)
            except Exception as e:
                if safe_load:
                    raise RuntimeError(f"Could not safely load checkpoint at {pth}") from e
                self.logger.info(f"Found possibly corrupt checkpoint at {pth}, reverting to an earlier checkpoint.")
            else:
                break

        return (state_dicts, pth) if get_path else state_dicts

    def _load_latest(self, safe_load: Optional[bool] = True, map_location: Optional[Any] = None):
        """ Attempts to load a previously saved checkpoint in order to resume model training. If a checkpoint is
        successfully loaded, the model, optimizer and scheduler state dictionaries are appropriately loaded and the
        relevant values of runtime and last_epoch are updated, else, the state dictionaries and other object
        attributes are left untouched.  If 'safe_load' is set to False, the most recent checkpoint that is readable
        from disk is loaded. Be careful with this setting - it is possible for the loaded checkpoint data and metrics
        data to be out of sync in such a scenario. It is set to True by default - i.e. if the latest checkpoint cannot
        be loaded, a RuntimeError is raised. 'map_location' has been provided for compatibility with GPUs. """

        latest = self._get_sorted_chkpt_paths(self.dir_tree.model_checkpoints_dir, ascending=False)
        state_dicts, pth = self._load_checkpoint(pths=latest, safe_load=safe_load, map_location=map_location,
                                                 get_path=True)

        if state_dicts is None:
            self.logger.info(f"No valid checkpoints found at {self.dir_tree.model_checkpoints_dir}.")
            return

        # state_dicts = torch.load(latest, map_location=map_location)
        self.model.load_state_dict(state_dicts["model_state_dict"])
        self.optimizer.load_state_dict(state_dicts["optimizer_state_dict"])
        self.scheduler.load_state_dict(state_dicts["scheduler_state_dict"])
        torch_rng_state = state_dicts["torch_rng_state"]
        # Fix needed for compatibilty with cuda, not yet tested for repeatability!!
        if "cpu" not in str(torch_rng_state.device.type):
            torch_rng_state = torch_rng_state.cpu()
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(state_dicts["numpy_rng_state"])
        random.setstate(state_dicts["python_rng_state"])
        self.last_epoch = state_dicts["epochs"]
        self.runtime = self._extract_runtime_from_filename(pth)
        self.logger.info(f"Loaded existing checkpoint from {str(pth)}")


class MetricLogger(object):
    """
    Holds a set of metrics, information about where they are to be logged, the frequency at which they should be
    logged as well as some functionality to convert any given set of metrics into a pandas DataFrame. Consult
    DirectoryTree for a breakdown of how the file structure is organized. A SynchroTimer object needs to be passed to
    the initializer argument 'timer' in order to enable logging functionality. Without it, a MetricLogger object can
    only be used in 'read-only' mode.
    """

    @enum.unique
    class MetricSet(enum.Enum):
        """ Defines uniquely recognized sets of metrics that are handled slightly differently from the norm. """
        model = enum.auto()
        task = enum.auto()
        default = enum.auto()

    def __init__(self, dir_tree: DirectoryTree, metrics: Dict, timer: Optional[SynchroTimer] = None,
                 set_type: MetricSet = MetricSet.default, logger: logging.Logger = None, where: Path = None,
                 safe_load: bool = True):
        """
        Setup a MetricLogger instance.

        :param dir_tree: DirectoryTree
            An instance of DirectoryTree that has already been initialized with the parameters appropriate for usage
            with this object, i.e. base directory, taskid and model index depending on which level of the tree this
            MetricLogger is intended to interface with.
        :param metrics: dict-like
            A dict-like object that will store the actual metric data. This object can be initialized with keys and no
            data before being passed to the MetricLogger, in which case the data will be filled in by loading it from
            disk.
        :param timer: SynchroTimer
            Used to coordinate the timing of saving metrics to disk with other parts of the code.
        :param set_type: Enum
            One of MetricLogger.MetricSet to indicate which level of the directory this logger will be interfacing with.
        :param logger: logging.Logger
            Used to generate log messages. Optional.
        :param where: Path
            An optional Path to a directory containing *.pkl.gz files, which should be compatible pickled Pandas
            DataFrames used to initialize the metrics dict. If None, the directory is read from the 'dir_tree'.
        :param safe_load: bool
            If True (default), failing to load the latest metrics log raises a RuntimeError. Otherwise, the next most
            recent readable metrics log is used instead.
        """
        self.dir_tree = dir_tree
        self.metrics = metrics
        self.timer = timer
        self.set_type = set_type
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.elapsed_runtime = 0. if self.timer is None else self.timer.previous_timestamp
        self.resume_latest_saved_metrics(where=where, safe_load=safe_load)

    @property
    def timer(self) -> SynchroTimer:
        return self._timer

    @timer.setter
    def timer(self, new_timer: SynchroTimer):
        if new_timer is None:
            self._timer = None
        elif not isinstance(new_timer, SynchroTimer):
            raise ValueError(f"Checkpointer requires a valid instance of SynchroTimer, cannot set timer to "
                             f"{type(new_timer)}.")
        else:
            self._timer = new_timer
            self._timer_id = self._timer.register_new_listener()

    @property
    def _signal(self):
        return False if self.timer is None else self.timer.ping(self._timer_id)

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
            f"Expected the model-level metrics DataFrame index to be an Index, not {type(df.index)}"
        assert isinstance(df.columns, pd.MultiIndex), \
            f"Expected the model-level metrics DataFrame columns to be a MultiIndex, not {type(df.columns)}"
        assert df.columns.nlevels == len(constants.metricdf_column_level_names), \
            f"Expected the model-level metrics DataFrame to have exactly {len(constants.metricdf_column_level_names)} levels " \
            f"in the columns MultiIndex, got {df.columns.nlevels}. Consult tabular_sampling.lib.constants for more " \
            f"details about the expected structure."

        df.index = df.index.set_names([constants.MetricDFIndexLevels.epoch.value]) + 1
        df.columns.set_names(constants.metricdf_column_level_names, inplace=True)
        return df

    @classmethod
    def _handle_task_metric_dict(cls, metrics: Dict) -> pd.DataFrame:
        model_configs = pd.DataFrame(metrics["model_config"]).to_dict()
        model_configs = {k: list(v.values()) for k, v in
                         model_configs.items()}  # Convert list of dicts to dict of lists
        df = cls._nested_dict_to_df({**metrics, **{"model_config": model_configs}})

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
        # Handle model configs specially since it's the only column index that uses a MultiIndex and needs to be
        # converted to a list of dicts format.
        model_configs = list(df["model_config"].transpose().to_dict().values())
        # The following is necessary to undo the automatic conversion of nested dicts to MultiIndex columns.
        other_cols = df.columns.unique(0).difference(["model_config"])
        metric_dict = cls._df_to_nested_dict(df[other_cols].droplevel(1, axis=1))
        metric_dict["model_config"] = model_configs

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
        # noinspection PyArgumentList
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

    def log(self, force: bool = False, where: Optional[Path] = None):
        if self.timer is None:
            raise RuntimeError("The log functionality of MetricLogger is enabled only when a valid timer has been set.")
        if self._signal or force:
            elapsed_runtime = self.timer.previous_timestamp
            df = self._generate_df()
            pth = where if where is not None else self.log_directory
            df.to_pickle(pth / f"{elapsed_runtime}.pkl.gz")
            self.elapsed_runtime = elapsed_runtime
            self.logger.info(f"Logged metrics at {str(pth)}")

    @classmethod
    def _extract_runtime_from_filename(cls, f: Path) -> float:
        return float(f.name.rstrip(".pkl.gz"))

    @classmethod
    def _get_sorted_metric_paths(cls, pth: Path, ascending: bool = True) -> List[Path]:
        latest = sorted(
            pth.rglob("*.pkl.gz"),
            key=lambda f: cls._extract_runtime_from_filename(f),
            reverse=not ascending
        )
        return latest

    @classmethod
    def _load_metrics_log(cls, pths: Union[Path, List[Path]],
                          safe_load: Optional[bool] = True, get_path: bool = False) -> \
            Union[Optional[pd.DataFrame], Tuple[Optional[pd.DataFrame], Path]]:
        """ Attempts to load a metrics log from either a single Path or a list of Paths, attempting each one in the
        order they occur. If 'safe_load' is set to False, the first metrics log in the list that is readable from disk
        is loaded. It is set to True by default - i.e. if the first metrics log cannot be loaded, a RuntimeError is
        raised. """

        if isinstance(pths, Path):
            pths = [pths]

        metric_df = None
        pth = None
        for pth in pths:
            try:
                metric_df = pd.read_pickle(pth)
            except Exception as e:
                if safe_load:
                    raise RuntimeError(f"Could not safely load metric DataFrame at {pth}") from e

                self.logger.info(f"Found possibly corrupt metric DataFrame at {pth}, reverting to an earlier "
                                 f"checkpoint.")
            else:
                break

        return (metric_df, pth) if get_path else metric_df

    def resume_latest_saved_metrics(self, where: Optional[Path] = None, safe_load: Optional[bool] = True):
        """ Used in conjunction with the corresponding method of Checkpointer to resume model training and evaluation
        from where it was left off. If 'safe_load' is set to False, the most recent metric data that is readable from
        disk is loaded. Be careful with this setting - it is possible for the loaded checkpoint data and metrics data
        to be out of sync in such a scenario. It is set to True by default - i.e. if the latest metrics DataFrame
        cannot be loaded, an error is raised."""

        pth = where if where is not None else self.log_directory
        latest = self._get_sorted_metric_paths(pth=pth, ascending=False)
        metric_df, dfpth = self._load_metrics_log(pths=latest, safe_load=safe_load, get_path=True)

        if metric_df is None:
            self.logger.info(f"No valid pre-existing metric DataFrames found at {str(pth)}.")
            return

        # noinspection PyArgumentList
        loaded_metrics = self._metric_set_df_handlers[self.set_type](metric_df)
        for k in self.metrics.keys():
            self.metrics[k] = loaded_metrics[k]

        self.elapsed_runtime = self._extract_runtime_from_filename(dfpth)


def attrdict_factory(metrics: Optional[List[str]] = None, template: Callable = AverageMeter) -> AttrDict:
    """ Convenience function for generating an AttrDict object with arbitrary keys initialized using a factory
    function, essentially an AttrDict extension to collections.defaultdict. 'metrics' is a list of keys (usually
    strings) and the initial value of these keys can optionally be set by passing an object factory to the parameter
    'template', which defaults to naslib.utils.utils.AverageMeter. """

    metrics = AttrDict({m: template() for m in metrics} if metrics else {})
    return metrics


def default_global_seed_gen(rng: Optional[np.random.RandomState] = None, global_seed: Optional[int] = None) \
        -> Iterable[int]:
    if global_seed is not None:
        return global_seed if isinstance(global_seed, Iterable) else repeat(global_seed)
    elif rng is not None:
        def seeds():
            while True:
                yield rng.randint(0, 2 ** 32 - 1)

        return seeds()
    else:
        raise ValueError("Cannot generate sequence of global seeds when both 'rng' and 'global_seed' are None.")


# TODO: Update documentation
def adapt_search_space(original_space: Union[NASB201HPOSearchSpace, ConfigSpace.ConfigurationSpace],
                       portfolio: Optional[pd.DataFrame] = None, taskid: Optional[int] = None,
                       opts: Optional[Union[Dict[str, Any], List[str]]] = None, suffix: Optional[str] = "_custom") -> \
        Union[Tuple[ConfigSpace.ConfigurationSpace, bool], bool]:
    """ Given a NASB201HPOSearchSpace object and a valid configuration, restricts the respective configuration space
    object and consequently the overall search space by setting the corresponding parameters to constant values. Such
    a configuration may be provided either by means of a portfolio file along with the relevant taskid or a dictionary
    'opts' mapping strings to values of any type, or both in which case values in 'opts' take precedence. In both
    cases, the configuration is read as a dictionary and the function attempts to autonomously restrict the search
    space such that for each key *k*, the corresponding parameter is set to the constant value *v*. An optional
    string 'suffix' may be provided to modify the name of the ConfigSpace object. Default: '_custom'. If None, the
    name remains unchanged. The name also remains unchanged in the case when the given space was not modified at all,
    most likely because no valid keys were present in the configuration dictionary or the dictionary was empty. In
    such a case, False is returned. Otherwise, True is returned. """

    new_consts = {}
    if portfolio is None and opts is None:
        return False
    elif portfolio is not None:
        if taskid is None:
            raise ValueError(f"When a portfolio is given, an integer taskid must also be provided. Was given {taskid}")
        else:
            new_consts = portfolio.loc[taskid % portfolio.index.size, :].to_dict()

    # Convert opts from list of string pairs to a mapping from string to string.
    if isinstance(opts, List):
        i = iter(opts)
        opts = {k: v for k, v in zip(i, i)}
    new_consts = {**new_consts, **opts}

    if hasattr(original_space, "config_space"):
        config_space = original_space.config_space
        flag_cs_attr = True
    else:
        config_space = original_space
        flag_cs_attr = False

    known_params = {p.name: p for p in config_space.get_hyperparameters()}

    def param_interpretor(param, value):
        known_config_space_value_types = {
            ConfigSpace.UniformIntegerHyperparameter: int,
            ConfigSpace.UniformFloatHyperparameter: float,
            ConfigSpace.CategoricalHyperparameter: lambda x: type(param.choices[0])(x),
            ConfigSpace.OrdinalHyperparameter: lambda x: type(param.sequence[0])(x),
        }
        return known_config_space_value_types[type(param)](value)

    modified = False

    for arg, val in new_consts.items():
        if arg in known_params:
            old_param = known_params[arg]
            new_val = param_interpretor(old_param, val)
            if isinstance(new_val, bool):
                new_val = int(new_val)  # Because ConfigSpace doesn't allow Boolean constants.
            known_params[arg] = ConfigSpace.Constant(arg, new_val,
                                                     meta=dict(old_param.meta, **dict(constant_overwrite=True)))
            modified = True

    if modified:
        new_config_space = ConfigSpace.ConfigurationSpace(f"{config_space.name}{suffix if suffix is not None else ''}")
        new_config_space.add_hyperparameters(known_params.values())
        if flag_cs_attr:
            original_space.config_space = new_config_space
        else:
            return new_config_space, modified

    return modified


def default_random_sampler(search_space: NASB201HPOSearchSpace, global_seed_gen: Iterable[int],
                           rng: np.random.RandomState, **kwargs) -> Tuple[NASB201HPOSearchSpace, dict, int]:
    """ Given a compatible search space, a global seed generator for fixing global seeds before model training and a
    local RNG for controlling sampling from the search space, generates samples from the search space, their
    configuration dictionary and the corresponding global seed. In this process, the global seed is automatically set.
    The extra keyword arguments are complete ignored and have been provided only to allow function signature
    overloading.
    """
    while True:
        curr_global_seed = next(global_seed_gen)
        naslib_utils.set_seed(curr_global_seed)
        model: NASB201HPOSearchSpace = search_space.clone()
        model.sample_random_architecture(rng=rng)
        model_config = model.config.get_dictionary()
        yield model, model_config, curr_global_seed


def model_sampler(search_space: NASB201HPOSearchSpace, taskid: int, global_seed_gen: Iterable[int],
                  rng: np.random.RandomState, portfolio_pth: Optional[Path] = None, opts: Optional[dict] = None,
                  cycle_models: bool = False, **kwargs) -> Iterable[Tuple[NASB201HPOSearchSpace, dict, int]]:
    """ Generates an iterable over tuples of initialized models, their configuration dictionary, and the global int
    seed used to initialize the global RNG state of PyTorch, NumPy and Random for that model by parsing a given search
    space, taking into account an optional portfolio of configurations. 'global_seed_gen' is an iterable of integers
    used to generate the int seeds for each model and 'rng' is used for consistent random sampling. An optional
    dictionary 'opts' mapping parameter names to constant values can be used to override the corresponding parameter
    values in the portfolio and the search space. If the flag 'cycle_models' is set, then in the case when a portfolio
    of fixed model configs for each task is given, the configs are cycled infinitely, thereby allowing the same configs
    to be evaluated under multiple global seeds. This flag has no effect when the portfolio does not contains
    model-specific configs or when no portfolio is given. Warning: When using the fixed sampling mode by specifying
    each individual model config, the behaviour of specifying additional 'opts' is undefined. """

    taskid_lvl = constants.MetricDFIndexLevels.taskid.value
    use_fixed_sampler = False
    if opts is None:
        opts = {}
    elif isinstance(opts, list):
        i = iter(opts)
        opts = {k: v for k, v in zip(i, i)}

    if portfolio_pth is not None:
        # Accommodate for a portfolio file modifying the search space.
        portfolio = pd.read_pickle(portfolio_pth)
        if isinstance(portfolio.index, pd.MultiIndex):
            # Format 2 of the portfolio: We need to sample only the models specified in the portfolio
            portfolio = portfolio.xs(taskid % portfolio.index.unique(taskid_lvl).size, axis=0, level=taskid_lvl)
            if "opts" in portfolio.columns.levels[0]:
                # These values should be constant across all sampled models
                opts = {**portfolio["opts"].iloc[0].to_dict(), **opts}
            use_fixed_sampler = True
            portfolio = portfolio["model_config"]
        else:
            # Format 1 of the portfolio: We need to fix one or more parameters in the search space for this task
            opts = {**portfolio.loc[taskid % portfolio.index.unique(taskid_lvl).size, :].to_dict(), **opts}

    # Use the search space modifiers that have now already subsumed the portfolio
    adapt_search_space(search_space, opts=opts, suffix=f"_task_{taskid}")

    if use_fixed_sampler:
        # The portfolio now consists of only the model configs that we want to iterate over within this task.
        model_inds = cycle(portfolio.index.values) if cycle_models else portfolio.index.values

        def sampler():
            for idx, curr_global_seed in zip(model_inds, global_seed_gen):
                model_config = portfolio.loc[idx, :].to_dict()
                model = search_space.clone()
                model.clear()
                model.config = ConfigSpace.Configuration(
                    model.config_space,
                    {**model.config_space.get_default_configuration().get_dictionary(), **model_config}
                )
                model._construct_graph()
                model_config = model.config.get_dictionary()
                yield model, model_config, curr_global_seed

        return sampler()
    else:
        return default_random_sampler(search_space=search_space, global_seed_gen=global_seed_gen, rng=rng)


