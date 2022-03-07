import logging

import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from typing import Optional, Union
from functools import partial

from tabular_sampling.distributed_nas_sampling import run_task
from tabular_sampling.lib.core.constants import Datasets
from tabular_sampling.lib.core.utils import DirectoryTree, MetricLogger, AttrDict

from tabular_sampling.surrogate.xgb import XGBSurrogate

_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)

def _map_dataset(dataset: str) -> Datasets:
    try:
        ds = [d for d in Datasets.__members__.values() if d.value[0] == dataset][0]
    except IndexError:
        raise ValueError(f"Invalid dataset name {dataset}, must be one of "
                         f"{[d.value[0] for d in Datasets.__members__.values()]}.")
    else:
        return ds


class Benchmark:
    def __init__(self, use_surrogate: bool = True, model_path: Optional[Union[str, Path]] = True):
        if use_surrogate:
            if isinstance(model_path, str):
                datadir = Path(datadir)

            assert model_path is not None, "A path to a directory where a surrogate model was saved must be given " \
                                           "when 'use_surrogate=True' is used."
            assert model_path.exists() and model_path.is_dir()
            self.surrogate = XGBSurrogate.load(model_path)
            self._call_fn = partial(Benchmark._benchmark_surrogate, surrogate=self.surrogate)
        else:
            self.surrogate = None
            self._call_fn = Benchmark._benchmark_live


    def __call__(self, config: dict, dataset: str = Datasets.cifar10.value[0],
                 datadir: Optional[Union[str, Path]] = None, nepochs: Optional[int] = 200,
                 batch_size: Optional[int] = 256, use_splits: Optional[bool] = True,
                 train_config: Optional[dict] = None, **kwargs):
        return self._call_fn(config=config, dataset=dataset, datadir=datadir, nepochs=nepochs, batch_size=batch_size,
                             use_splits=use_splits, train_config=train_config, **kwargs)

    @staticmethod
    def _benchmark_surrogate(surrogate: XGBSurrogate, config: dict, dataset: str, datadir: Union[str, Path],
                             nepochs: Optional[int] = 200, batch_size: Optional[int] = 256,
                             use_splits: Optional[bool] = True, train_config: Optional[dict] = None, **kwargs) -> dict:
        features = pd.Series(config).to_frame().transpose()
        features.loc[:, "epoch"] = nepochs

        outputs: np.ndarray = surrogate.predict(features)
        outputs = outputs.reshape(-1, surrogate.label_headers.size)

        return {k: outputs[0][i] for i, k in enumerate(surrogate.label_headers.values)}

    @staticmethod
    def _benchmark_live(config: dict, dataset: str, datadir: Union[str, Path], nepochs: Optional[int] = 200,
                        batch_size: Optional[int] = 256, use_splits: Optional[bool] = True,
                        train_config: Optional[dict] = None, worker_dir: Optional[Path] = None,
                        clean_tmp_files : bool = True, **kwargs) -> dict:
        """ Simple wrapper around the base benchmark data generation capabilities offered by
        tabular_sampling.distributed_nas_samplig.run_task(). Providing 'train_config' and 'kwargs' dicts can be used to
        access the full range of customizations offered by 'run_task()' - consults its documentation if needed. This
        script requires access to /tmp in order to write temporary data. """

        if isinstance(datadir, str):
            datadir = Path(datadir)

        if train_config is None:
            train_config = dict(epochs=nepochs, batch_size=batch_size, use_grad_clipping=False, split=use_splits,
                                warmup_epochs=0, disable_checkpointing=True, checkpoint_interval_seconds=3600,
                                checkpoint_interval_epochs=50)

        basedir = (Path("/tmp") if worker_dir is None else worker_dir) / "neps_bench"
        basedir.mkdir(exist_ok=True)
        dataset = _map_dataset(dataset)

        args = {**dict(basedir=basedir, taskid=0, train_config=AttrDict(train_config), dataset=dataset, datadir=datadir,
                       local_seed=None, global_seed=None, debug=False, generate_sampling_profile=False, nsamples=1,
                       portfolio_pth=None, cycle_portfolio=False, opts=config), **kwargs}
        run_task(**args)

        dtree = DirectoryTree(basedir=basedir, taskid=0, model_idx=1, read_only=True)
        metric_pth = MetricLogger._get_sorted_metric_paths(pth=dtree.model_metrics_dir)[-1]
        df = pd.read_pickle(metric_pth)

        # Model metrics dataframe does not have a MultiIndex index - it's simply the epochs and their metrics!
        nepochs = df.index.max()
        latest = df.loc[nepochs]
        if clean_tmp_files:
            shutil.rmtree(dtree.basedir, ignore_errors=True)

        return latest.to_dict()


if __name__ == "__main__":
    from tabular_sampling.search_space.configspace import joint_config_space
    config = joint_config_space.get_default_configuration().get_dictionary()

    b = Benchmark(use_surrogate=False)
    res = b(config=config, dataset="Cifar-10", datadir="/home/archit/thesis/datasets", nepochs=3)
    print(res)

    b = Benchmark(use_surrogate=True, model_path=Path("/home/archit/thesis/experiments/test/surrogates/full_data"))
    res = b(config=config, nepochs=200)
    print(res)
