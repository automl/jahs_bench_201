import logging
from enum import Enum, unique, auto
from pathlib import Path
from typing import Optional, Union, Sequence, Tuple

import numpy as np
import pandas as pd

from jahs_bench_201.lib.core.configspace import joint_config_space
from jahs_bench_201.surrogate.model import XGBSurrogate
import jahs_bench_201.download

## Requires installation of the optional "data_creation" components and dependencies
try:
    from jahs_bench_201.tabular.distributed_nas_sampling import run_task
    from jahs_bench_201.tabular.lib.core.constants import Datasets
    from jahs_bench_201.tabular.lib.core.utils import DirectoryTree, MetricLogger, \
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

    def __init__(self, task: Union[str, BenchmarkTasks],
                 kind: Union[str, BenchmarkTypes],
                 download: bool = True, save_dir: Optional[Union[str, Path]] = None):
        """ Load up the benchmark for querying. """

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

        if download and kind == "surrogate":
            save_dir = Path(save_dir)
            surrogate_dir = save_dir / "assembled_surrogates"
            if not surrogate_dir.exists():
                jahs_bench_201.download.download_surrogates(save_dir)

        if download and kind == "table":
            save_dir = Path(save_dir)
            metric_data_dir = save_dir / "metric_data"
            if not metric_data_dir.exists():
                jahs_bench_201.download.download_metrics(save_dir)

        loaders = {
            BenchmarkTypes.Surrogate: self._load_surrogate,
            BenchmarkTypes.Table: self._load_table,
            BenchmarkTypes.Live: self._load_live,
        }

        # Download the data and setup the benchmark
        loaders[kind]()

    def _load_surrogate(self, model_path: Optional[Union[str, Path]] = None,
                        outputs: Optional[Sequence[str]] = None):
        assert model_path.exists() and model_path.is_dir()

        if outputs is None:
            outputs = [p.name for p in model_path.iterdir() if p.is_dir()]

        self._surrogates = {}
        for o in outputs:
            self._surrogates[o] = XGBSurrogate.load(model_path / str(o))
        self._call_fn = self._benchmark_surrogate

    def _load_table(self, table_path: Optional[Union[str, Path]] = None,
                    outputs: Optional[Sequence[str]] = None):
        assert table_path.exists() and table_path.is_file()

        table = pd.read_pickle(table_path)
        level_0_cols = ["features", "labels"]
        features: list = joint_config_space.get_hyperparameter_names() + ["epoch"]

        if table["features"].columns.intersection(features).size != len(features):
            raise ValueError(f"The given performance dataset at {table_path} could not "
                             f"be resolved against the known search space consisting of "
                             f"the parameters {features}")
        features = table["features"].columns
        labels: pd.Index = table["labels"].columns

        if outputs is not None:
            # Attempt to convert the sequence of outputs into a list
            outputs = list(outputs) if not isinstance(outputs, list) \
                else [outputs] if isinstance(outputs, str) else outputs

            if labels.intersection(outputs).size != len(outputs):
                raise ValueError(f"The set of outputs specified for the performance "
                                 f"dataset {outputs} must be a subset of all available "
                                 f"outputs: {labels.tolist()}.")

            # Drop all unnecessary outputs
            table.drop([("labels", l) for l in labels.difference(outputs)], axis=1,
                       inplace=True)

        # TODO: Deal with the issue of the index being possibly non-unique, since there
        #  are no guarantees that a configuration wasn't sampled twice.
        # Make the DataFrame indexable by configurations
        # table.set_index(table[["features"]].columns.tolist(), inplace=True)
        # table.index.names = features.tolist()
        # table = table.droplevel(0, axis=1)
        self._table = table
        self._call_fn = self._benchmark_tabular

    def __call__(self, config: dict, nepochs: Optional[int] = 200,
                 full_trajectory: bool = False, **kwargs):
        return self._call_fn(config=config, nepochs=nepochs)

    def _benchmark_surrogate(self, config: dict, nepochs: Optional[int] = 200,
                             full_trajectory: bool = False,) -> dict:
        features = pd.Series(config).to_frame().transpose()
        features.loc[:, "epoch"] = nepochs

        outputs = {}
        for o, model in self._surrogates.items():
            outputs[o] = model.predict(features)

        outputs: pd.DataFrame = pd.concat(outputs, axis=1)
        return outputs.to_dict()

    # TODO: Return only the first hit of a query when multiple instances of a config are
    #  present
    def _benchmark_tabular(self, config: dict, nepochs: Optional[int] = 200,
                           full_trajectory: bool = False,
                           suppress_keyerror: bool = False) -> dict:
        raise NotImplementedError("The functionality for directly querying the tabular "
                                  "performance dataset is still under construction.")
        assert self._table is not None, "No performance dataset has been loaded into " \
                                        "memory - a tabular query cannot be made."
        query = config.copy()
        query["epoch"] = nepochs
        query = tuple((query[k] for k in self._table.index.names))
        try:
            output = self._table.loc[query].to_dict(orient="index")
            output = list(output.values())[0]
        except KeyError as e:
            _log.debug(f"Registered a key-error while querying the performance dataset "
                       f"for the configuration: {config} at {nepochs} epochs. The "
                       f"constructed query was: {query}.")
            if suppress_keyerror:
                output = {}
            else:
                raise KeyError(f"Could not find any entries for the config {config} at "
                               f"{nepochs} epochs.") from e

        return output

    def _benchmark_live(self, config: dict, nepochs: Optional[int] = 200,
                        # datadir: Union[str, Path],
                        # batch_size: Optional[int] = 256,
                        # use_splits: Optional[bool] = True,
                        train_config: Optional[dict] = None,
                        worker_dir: Optional[Path] = None, clean_tmp_files : bool = True,
                        full_trajectory: bool = False,
                        **kwargs) -> dict:
        """
        Simple wrapper around the base benchmark data generation capabilities offered by
        tabular_sampling.distributed_nas_sampling.run_task(). Providing 'train_config'
        and 'kwargs' dicts can be used to access the full range of customizations offered
        by 'run_task()' - consults its documentation if needed. `worker_dir` can be used
        to specify a custom directory where temporary files generated during model
        training are stored. If it is not provided, a directory will be created at
        '/tmp/jahs_bench_201'. Disable the flag `clean_tmp_files` to retain these
        temporary files after the function call.
        """

        if not data_creation_available:
            raise RuntimeError(
                f"Cannot train the given configuration since the required modules have "
                f"not been installed. Optional component 'data_creation' of "
                f"jahs_bench_201 must be installed in order to perform a live training "
                f"of the given configuration. Alternatively, try to query on either the "
                f"surrogate model or the performance dataset table.")

        if isinstance(datadir, str):
            datadir = Path(datadir)

        if train_config is None:
            train_config = dict(epochs=nepochs, batch_size=256, use_grad_clipping=False,
                                split=True, warmup_epochs=0, disable_checkpointing=True,
                                checkpoint_interval_seconds=3600,
                                checkpoint_interval_epochs=50)

        basedir = (Path("/tmp") if worker_dir is None else worker_dir) / "jahs_bench_201"
        basedir.mkdir(exist_ok=True)
        task = Datasets[self.task.value]

        args = {**dict(basedir=basedir, taskid=0, train_config=AttrDict(train_config),
                       dataset=task, datadir=datadir, local_seed=None, global_seed=None,
                       debug=False, generate_sampling_profile=False, nsamples=1,
                       portfolio_pth=None, cycle_portfolio=False, opts=config), **kwargs}
        run_task(**args)

        dtree = DirectoryTree(basedir=basedir, taskid=0, model_idx=1, read_only=True)
        metric_pth = MetricLogger._get_sorted_metric_paths(pth=dtree.model_metrics_dir)
        metric_pth = metric_pth[-1]
        df = pd.read_pickle(metric_pth)

        # It is possible that the model training diverged, so we cannot directly use the
        # value of 'nepochs' provided by the user.
        nepochs = df.index.max()
        latest = df.loc[nepochs]
        if clean_tmp_files:
            shutil.rmtree(dtree.basedir, ignore_errors=True)

        return latest.to_dict()

    def random_sample(self,
                      random_state: Optional[Union[int, np.random.RandomState]] = None,
                      **kwargs) -> Tuple[dict, dict]:
        """ Randomly query the benchmark for a configuration. If a tabular benchmark has
        been loaded, a sample from the set of known configurations is queried. Otherwise,
        a random configuration is sampled from the search space and queried on the
        surrogate benchmark or trained live, as the case may be. An optional seed for
        initializing an RNG or a pre-initialized RNG may be passed in `random_state` for
        reproducible queries. """

        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        if self._table is not None:
            index = random_state.choice(self._table.index)
            row = self._table.loc[index]
            config = row["features"].to_dict()
            result = row["labels"].to_dict()
            # TODO: Reinstate the functionality to query the table itself for consistency
            #  once the issue of non-unique indices has been fixed
            # query = random_state.choice(self._table.index)
            # query = {self._table.index.names[i]: query[i] for i in range(len(query))}

            # Quite convoluted and redundant, but this helps maintain consistency.
            # nepochs = query.pop("epoch")
            # output = self(config=query, nepochs=nepochs)
            # return {**query, **{"epoch": nepochs}}, output
        else:
            joint_config_space.random = random_state
            config = joint_config_space.sample_configuration().get_dictionary()
            nepochs = random_state.randint(1, 200)
            result = self(config=config, nepochs=nepochs, **kwargs)
            config["epoch"] = nepochs

        return config, result


if __name__ == "__main__":
    b = Benchmark(task="cifar10", kind="surrogate")
    config, result = b.random_sample()
    print(f"Config: {config}")
    print(f"Result: {result}")
