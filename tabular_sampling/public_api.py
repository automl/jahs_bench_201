import logging
import pandas as pd
from pathlib import Path
from typing import Optional

from distributed_nas_sampling import run_task
from tabular_sampling.lib.constants import training_config, Datasets, MetricDFIndexLevels
from tabular_sampling.lib.utils import DirectoryTree, MetricLogger, AttrDict
from tabular_sampling.lib.postprocessing.metric_df_ops import load_metric_df

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


def benchmark(config: dict, dataset: str, datadir: Path, nepochs: Optional[int] = 200, batch_size: Optional[int] = 256,
              use_splits: Optional[bool] = True, train_config: Optional[dict] = None, **kwargs) -> dict:
    """ Simple wrapper around the base benchmark data generation capabilities offered by
    tabular_sampling.distributed_nas_samplig.run_task(). Providing 'train_config' and 'kwargs' dicts can be used to
    access the full range of customizations offered by 'run_task()' - consults its documentation if needed. This script
    requires access to /tmp in order to write temporary data. """

    if train_config is None:
        train_config = dict(epochs=nepochs, batch_size=batch_size, use_grad_clipping=False, split=use_splits,
                            warmup_epochs=0, disable_checkpointing=True, checkpoint_interval_seconds=3600,
                            checkpoint_interval_epochs=50)

    basedir = Path("/tmp") / "tabular_sampling"
    basedir.mkdir(exist_ok=True)
    dataset = _map_dataset(dataset)

    args = dict(basedir=basedir, taskid=0, train_config=AttrDict(train_config), dataset=dataset, datadir=datadir,
                local_seed=None, global_seed=None, debug=False, generate_sampling_profile=False, nsamples=1,
                portfolio_pth=None, cycle_portfolio=False, opts=config) | kwargs
    run_task(**args)

    dtree = DirectoryTree(basedir=basedir, taskid=0, model_idx=1, read_only=True)
    metric_pth = MetricLogger._get_latest_metric_path(pth=dtree.model_metrics_dir)
    df = pd.read_pickle(metric_pth)
    nepochs = df.index.unique(MetricDFIndexLevels.epoch.value).max()
    latest = df.xs(nepochs, level=MetricDFIndexLevels.epoch.value)

    return latest.to_dict()
