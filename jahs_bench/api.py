import logging
from enum import Enum, unique, auto
from pathlib import Path
from typing import Optional, Union, Sequence, Tuple, Iterable

import numpy as np
import pandas as pd

from jahs_bench.lib.core.configspace import joint_config_space
from jahs_bench.surrogate.model import XGBSurrogate
import jahs_bench.download

## Requires installation of the optional "data_creation" components and dependencies
try:
    from jahs_bench.tabular.sampling import run_task
    from jahs_bench.tabular.lib.core.constants import Datasets as _Tasks
    from jahs_bench.tabular.lib.core.utils import DirectoryTree, MetricLogger, \
        AttrDict
    import shutil
except ImportError as e:
    data_creation_available = False
else:
    data_creation_available = True

_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)

@unique
class BenchmarkTypes(Enum):
    Surrogate = "surrogate"
    Table = "table"
    Live = "live"


@unique
class BenchmarkTasks(Enum):
    CIFAR10 = "cifar10"
    ColorectalHistology = "colorectal_histology"
    FashionMNIST = "fashion_mnist"


class Benchmark:
    _call_fn = None
    _surrogates = None
    _table = None
    __known_metrics = ("FLOPS", "latency", "runtime", "size_MB", "test-acc", "train-acc",
                       "valid-acc")

    def __init__(self, task: Union[str, BenchmarkTasks],
                 kind: Union[str, BenchmarkTypes] = BenchmarkTypes.Surrogate,
                 download: bool = True, save_dir: Union[str, Path] = "jahs_bench_data",
                 metrics: Optional[Iterable[str]] = None, lazy: bool = False):
        """
        Public facing API for accessing JAHS-Bench, capable of querying a single
        configuration at a time on any known task in three different modes: surrogate,
        tabular and live.

        task: name of task
            A member of the BenchmarkTasks enum or a string corresponding to the value of
            such a member indicating the specific task that the benchmark should be
            queried on.
        kind: type of query
            A member of the BenchmarkTypes enum or a string corresponding to the value of
            such a member indicating the type of query that should be run. "surrogate"
            implies that a corresponding surrogate model will be used to predict the
            performance metrics. "tabular" implies that the tabular dataset of
            performance data will be queried. "live" implies that the configuration will
            be used to generate and run a fresh neural network pipeline on the given task
            and the generated metrics will be recorded and returned.
        download: bool
            A flag indicating whether or not the requisite data (tabular, surrogate
            model, task data) should be downloaded. If False, the relevant data is
            expected to exist under "save_dir".
        save_dir: Path-like
            The (absolute or relative) path to a directory where the data required for
            the benchmark to run will be read from (and saved to if download=True).
        metrics: optional sequence of strings
            An optional sequence of strings indicating which performance metrics need to
            be predicted by the benchmark. This is especially useful for the surrogate
            benchmark, as specifying only a subset of the available metrics drastically
            reduces the memory requirements of the surrogate model.
        lazy: bool  # TODO: Finalize
            (Experimental) A flag to enable lazy loading of the surrogate models (future:
            extend to tabular and live), thereby loading each metric's surrogate model
            once per query and one at a time, in order to reduce the instantaneous memory
            requirements of the surrogate benchmark.
        """

        if isinstance(task, str):
            try:
                task = BenchmarkTasks(task.lower())
            except ValueError as e:
                valid = [x.value for x in BenchmarkTasks]
                raise ValueError(f"Invalid/Unknown value of parameter 'task': '{task}'. "
                                 f"Must be one of {valid}.") from e

        if isinstance(kind, str):
            try:
                kind = BenchmarkTypes(kind)
            except ValueError as e:
                valid = [x.value for x in BenchmarkTypes]
                raise ValueError(f"Invalid/Unknown value of parameter 'kind': '{kind}'. "
                                 f"Must be one of {valid}.") from e

        self.kind = kind
        self.task = task
        self.metrics = tuple(metrics) if metrics is not None else self.__known_metrics
        self.save_dir = Path(save_dir)
        self.surrogate_dir = self.save_dir / "assembled_surrogates"
        self.table_dir = self.save_dir / "metric_data"
        self.task_dir = self.save_dir / "tasks"
        self._lazy = lazy

        if download and kind is BenchmarkTypes.Surrogate:
            if not self.surrogate_dir.exists():
                jahs_bench.download.download_surrogates(self.save_dir)

        if download and kind is BenchmarkTypes.Table:
            if not self.table_dir.exists():
                jahs_bench.download.download_metrics(self.save_dir)

        if download and kind is BenchmarkTypes.Live:
            # TODO: Implement
            # if not self.table_dir.exists():
            #     jahs_bench.download.download_tasks(self.save_dir)
            pass

        loaders = {
            BenchmarkTypes.Surrogate: self._load_surrogate,
            BenchmarkTypes.Table: self._load_table,
            BenchmarkTypes.Live: self._load_live,
        }

        # Setup the benchmark
        loaders[kind]()

    def _load_surrogate(self):
        assert self.surrogate_dir.exists() and self.surrogate_dir.is_dir()

        model_path = self.surrogate_dir / self.task.value
        outputs = [p.name for p in model_path.iterdir()
                   if p.is_dir() and p.name in self.metrics]

        self._surrogates = {}

        for o in outputs:
            pth = model_path / str(o)
            self._surrogates[o] = XGBSurrogate.load(pth) if not self._lazy else \
                _LazySurrogate(model_pth=pth)

        self._call_fn = self._benchmark_surrogate

    def _load_table(self):
        assert self.save_dir.exists() and self.save_dir.is_dir()

        table_path = self.save_dir / self.task.value
        table_names = ["train_set.pkl.gz", "valid_set.pkl.gz", "test_set.pkl.gz"]
        tables = [pd.read_pickle(table_path / n) for n in table_names]
        table = pd.concat(tables, axis=0)
        del tables

        # level_0_cols = ["features", "labels"]
        features: list = joint_config_space.get_hyperparameter_names() + ["epoch"]

        if table["features"].columns.intersection(features).size != len(features):
            raise ValueError(f"The given performance datasets at {table_path} could not "
                             f"be resolved against the known search space consisting of "
                             f"the parameters {features}")

        self._features = table["features"].columns
        self._labels: pd.Index = table["labels"].columns
        self._table_features = table.loc[:, "features"]
        self._table_labels = table.loc[:, "labels"]
        self._table_features.rename_axis("Sample ID", axis=0, inplace=True)
        self._table_features = self._table_features.reset_index()
        self._call_fn = self._benchmark_tabular

    def _load_live(self):
        self._call_fn = self._benchmark_live

    def __call__(self, config: dict, nepochs: Optional[int] = 200,
                 full_trajectory: bool = False, **kwargs):
        return self._call_fn(config=config, nepochs=nepochs,
                             full_trajectory=full_trajectory, **kwargs)

    def _benchmark_surrogate(self, config: dict, nepochs: Optional[int] = 200,
                             full_trajectory: bool = False, **kwargs) -> dict:
        assert nepochs > 0

        if full_trajectory:
            features = pd.DataFrame([config] * nepochs)
            epochs = list(range(1, nepochs+1))
            features = features.assign(epoch=epochs)
        else:
            features = pd.Series(config).to_frame().transpose()
            epochs = [nepochs]
            features.loc[:, "epoch"] = nepochs

        outputs = []
        for model in self._surrogates.values():
            outputs.append(model.predict(features))

        outputs: pd.DataFrame = pd.concat(outputs, axis=1)
        # outputs = outputs.reindex(index=epochs)
        outputs.index = epochs
        return outputs.to_dict(orient="index")

    def _benchmark_tabular(self, config: dict, nepochs: Optional[int] = 200,
                           full_trajectory: bool = False, **kwargs) -> dict:
        assert nepochs > 1
        assert self._table_features is not None and self._table_labels is not None,\
            "No performance dataset has been loaded into memory - a tabular query " \
            "cannot be made."

        query = config.copy()
        if "epoch" in query:
            query.pop("epoch")

        query_df = pd.DataFrame(query, index=list(range(1, nepochs+1)) \
                                if full_trajectory else [nepochs])
        query_df.rename_axis("epoch", axis=0, inplace=True)
        query_df = query_df.reset_index()

        check = self._features.difference(query_df.columns)
        if check.size != 0:
            raise ValueError(f"The given query has missing parameters: {check.tolist()}")

        idx = pd.merge(self._table_features, query_df, how="inner")["Sample ID"]

        if idx.size == 0:
            raise KeyError(f"Could not find any entries for the config {config} at "
                           f"{nepochs} epochs.") from e
        elif full_trajectory:
            # Return the full trajectory, but only for the first instance of this config
            # that was found.
            result = self._table_labels.loc[idx, :].iloc[:nepochs]
        else:
            # Return only the first result that was found
            result = self._table_labels.loc[idx, :].iloc[:1]

        return result.to_dict(orient="index")

    def _benchmark_live(self, config: dict, nepochs: Optional[int] = 200,
                        full_trajectory: bool = False, *,
                        train_config: Optional[dict] = None,
                        worker_dir: Optional[Path] = None, clean_tmp_files : bool = True,
                        **kwargs) -> dict:
        """
        Simple wrapper around the base benchmark data generation capabilities offered by
        tabular_sampling.distributed_nas_sampling.run_task(). Providing 'train_config'
        and 'kwargs' dicts can be used to access the full range of customizations offered
        by 'run_task()' - consults its documentation if needed. `worker_dir` can be used
        to specify a custom directory where temporary files generated during model
        training are stored. If it is not provided, a directory will be created at
        '/tmp/jahs_bench'. Disable the flag `clean_tmp_files` to retain these
        temporary files after the function call.
        """

        if not data_creation_available:
            raise RuntimeError(
                f"Cannot train the given configuration since the required modules have "
                f"not been installed. Optional component 'data_creation' of "
                f"jahs_bench must be installed in order to perform a live training "
                f"of the given configuration. Alternatively, try to query on either the "
                f"surrogate model or the performance dataset table.")

        if train_config is None:
            train_config = dict(epochs=nepochs, batch_size=256, use_grad_clipping=False,
                                split=True, warmup_epochs=0, disable_checkpointing=True,
                                checkpoint_interval_seconds=3600,
                                checkpoint_interval_epochs=50)

        basedir = (Path("/tmp") if worker_dir is None else worker_dir) / "jahs_bench"
        basedir.mkdir(exist_ok=True)
        task = _Tasks[self.task.value]

        args = {**dict(basedir=basedir, taskid=0, train_config=AttrDict(train_config),
                       dataset=task, datadir=self.task_dir, local_seed=None, global_seed=None,
                       debug=False, generate_sampling_profile=False, nsamples=1,
                       portfolio_pth=None, cycle_portfolio=False, opts=config), **kwargs}
        run_task(**args)

        dtree = DirectoryTree(basedir=basedir, taskid=0, model_idx=1, read_only=True)
        metric_pth = MetricLogger._get_sorted_metric_paths(pth=dtree.model_metrics_dir)
        metric_pth = metric_pth[-1]
        df: pd.DataFrame = pd.read_pickle(metric_pth)

        if full_trajectory:
            result = df.to_dict(orient="index")
        else:
            # It is possible that the model training diverged, so we cannot directly use
            # the value of 'nepochs' provided by the user.
            nepochs = df.index.max()
            result = df.loc[nepochs].to_dict(orient="index")

        if clean_tmp_files:
            shutil.rmtree(dtree.basedir, ignore_errors=True)

        return result

    def sample_config(self,
                      random_state: Optional[Union[int, np.random.RandomState]] = None,
                      ) -> dict:
        """ For a tabular benchmark, return a random configuration from the set of
        configurations recorded in the currently loaded dataset. Otherwise, randomly
        sample a configuration from the full search space and return it. """

        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        if self.kind is BenchmarkTypes.Table:
            assert self._table_features is not None, \
                "Cannot extract random sample - the tabular benchmark could not be " \
                "properly initialized."
            index = random_state.choice(self._table_features.index)
            row = self._table_features.loc[index].drop("Sample ID")
            config = row.to_dict()
        else:
            joint_config_space.random = random_state
            config = joint_config_space.sample_configuration().get_dictionary()
            nepochs = random_state.randint(1, 200)
            config['epoch'] = nepochs

        return config


class _LazySurrogate(XGBSurrogate):
    """ A wrapper around the XGBSurrogate object that defers the actual read of the
    surrogate model from the disk to the moment at which the model is actually used for a
    prediction, thus reducing overall memory requirements at the expense of more disk
    reads. """

    def __init__(self, model_pth: Union[str, Path], **kwargs):
        self._model_pth = Path(model_pth)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        model = XGBSurrogate.load(self._model_pth)
        res = model.predict(features)
        del model
        return res

    def fit(self, *args, **kwargs):
        raise RuntimeError("Lazy loading of surrogates is enabled. This mode only "
                           "supports the predict() functionality.")

    def dump(self, *args, **kwargs):
        raise RuntimeError("Lazy loading of surrogates is enabled. This mode only "
                           "supports the predict() functionality.")

    @classmethod
    def load(cls, outdir: Path) -> None:
        raise RuntimeError("Lazy loading of surrogates is enabled. This mode only "
                           "supports the predict() functionality.")



if __name__ == "__main__":
    b = Benchmark(task="cifar10", kind="surrogate")
    config = b.sample_config()
    print(f"Config: {config}")

    result = b(config, nepochs=config["epoch"])
    print(f"Result: {result}")
