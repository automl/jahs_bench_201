"""
A small script to extract the best performing models from all available configurations
generated after an HPO run using the NEPS framework.
"""
import argparse
import functools
import logging
from pathlib import Path
import shutil
import yaml
from typing import Sequence, Iterator, Dict, Tuple, Optional
from numpy import inf

_log = logging.getLogger(__name__)

def get_loss(config_dir: Path) -> float:
    """ Given the path to a particular configuration's working directory, extract the
    configuration's recorded loss and return it. """

    result_file = config_dir / "result.yaml"
    with open(result_file) as fp:
        result = yaml.safe_load(fp)

    # If the configuration evaluation resulted in some error, set the loss to infinity
    return inf if isinstance(result, str) else result["loss"]

def iterate_configs(metric_dir: Path, max_configs: Optional[int] = None) -> Iterator[Path]:
    """ Given the working directory corresponding to a single metric, iterate over all
    the individual configurations' working directories. """

    subdir = metric_dir / "results"
    config_dirs = subdir.iterdir() if max_configs is None \
        else [subdir / f"config_{i}" for i in range(1, max_configs + 1)]
    for config_dir in config_dirs:
        yield config_dir

def iterate_metrics(root_dir: Path, common_suffix_dirs: Optional[Sequence[str]] = None) \
        -> Iterator[Tuple[str, Path]]:
    """ Given a root directory within which, after descending down an optional sequence
    of sub-directories common to each directory tree, the working directories for each
    metric can be found, return a generator to iterate over these working directories. """

    for subdir in root_dir.iterdir():
        metric_dir = subdir
        metric_name = subdir.name
        if common_suffix_dirs is not None:
            metric_dir = functools.reduce(lambda a, x: a / x, common_suffix_dirs,
                                          metric_dir)
        yield metric_name, metric_dir

def identify_best_configs(root_dir: Path, common_suffix_dirs: Optional[
                          Sequence[str]] = None, max_configs: Optional[int] = None) \
        -> Dict[str, Tuple[Path, float]]:
    """ Given a root directory containing all the metric directories for a single
    dataset's surrogates (after optionally further descending down a common set of
    sub-directories), returns a dictionary mapping each metric to the working directory
    of the best performing model configuration and its loss value. When 'max_configs' is
    given, only the first 'max_configs' configurations are considered. """

    best_configs = {}
    for metric, metric_dir in iterate_metrics(root_dir, common_suffix_dirs):
        best_config = None
        min_loss = None
        _log.info(f"Identifying best configuration for metric {metric}.")
        for config_dir in iterate_configs(metric_dir, max_configs):
            loss = get_loss(config_dir)
            if best_config is None:
                min_loss = loss
                best_config = config_dir
            elif loss < min_loss:
                min_loss = loss
                best_config = config_dir
        best_configs[metric] = (best_config, min_loss)
        _log.info(f"Best configuration for metric {metric} found at {best_config} with "
                  f"loss {min_loss}.")

    return best_configs

def assemble_surrogate(best_configs: Dict[str, Tuple[Path, float]], final_dir: Path) \
        -> Path:
    """ Given a mapping from metric names to a tuple containing the path to the best
    configuration and its loss for that metric, copies all the models into the new
    directory tree that can be used by the JAHS-Bench-201 public API to load a surrogate
    model. The path to this directory is returned. """

    final_dir.mkdir(exist_ok=True, parents=False)
    for metric, (config_dir, loss) in best_configs.items():
        model_dir = config_dir / "xgb_model"
        pth: Path = shutil.copytree(model_dir, final_dir / metric)
        _log.info(f"Inserted best configuration for metric '{metric}' at {pth}")

    return final_dir

def main(final_dir: Path, root_dir: Path, max_configs: Optional[int] = None,
         common_suffix_dirs: Optional[Sequence[str]] = None):
    """ Executes the main program control. """

    assert root_dir.exists() and root_dir.is_dir()
    _log.info(f"Identifying the best configuration for each metric from those present "
              f"in {root_dir}.")
    best_configs = identify_best_configs(root_dir, common_suffix_dirs, max_configs)

    _log.info(f"Assembling the individual models into one directory at {final_dir}.")
    final_model_dir = assemble_surrogate(best_configs, final_dir)

    _log.info(f"Assembled the final model at: {final_model_dir}")

def parse_cli():
    parser = argparse.ArgumentParser(
        "Assemble the best configurations found for each metric after HPO using NEPS "
        "into a single ensemble of surrogate models usable by the public API. "
    )
    parser.add_argument("--final_dir", type=Path,
                        help="The final directory where the assembled ensemble should be "
                             "stored.")
    parser.add_argument("--root_dir", type=Path,
                        help="Path to the working directory where, for a single task, "
                             "all the metrics' evaluated configurations are stored.")
    parser.add_argument("--max_configs", type=int, default=None,
                        help="Maximum number of configs to consider when choosing the "
                             "best performing config. When given, only the first "
                             "'max-configs' configs are considered, otherwise all "
                             "available configs are considered.")
    parser.add_argument("--common_suffix_dirs", type=str, default=None,
                        nargs=argparse.REMAINDER,
                        help="An optional sequence of strings defining further "
                             "sub-directories beyond 'root_dir/<metric_name/' to be "
                             "traversed for each metric before the directory containing "
                             "configuration specific working directories is reached.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_cli()
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                        datefmt="%m/%d %H:%M:%S")
    _log.setLevel(logging.INFO)
    main(**vars(args))
