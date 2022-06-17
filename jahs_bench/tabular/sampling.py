"""
Main script for sampling from a specified joint NAS+HPO search space, training and evaluating the sampled
configurations and recording the metrics when run on a server. Intended for use with a SLURM script such that one job
launches multiple parallel python processes (tasks), each running any specified number of logical CPUs. Each task then
evaluates a number of configurations (models). The data is generated and stored using the following directory structure
template:
<base_dir>    <-- Possibly a job-level directory
|    |--> <task_idx>/
|    |    |--> metrics/
|    |    |    |--> <task-level metrics, if any>
|    |    |--> models /
|    |    |--> <model_idx>/
|    |    |    |--> metrics/
|    |    |    |    |--> <model-level metrics, if any>
|    |    |    |--> checkpoints/
|    |    |    |    |--> <model checkpoints, if any>

The PRNGs are setup in such a way that, for the hard-coded, fixed seed value, the same task, as identified by an
integer task_id, will generate the same sequence of samples from any given search space as well as the same sequence of
global seeds for numpy/torch/random.

"""

import argparse
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Iterable, Sequence, Optional, Union

import numpy as np
import torch

import jahs_bench.tabular.lib.naslib.utils.logging as naslib_logging
import jahs_bench.tabular.lib.naslib.utils.utils as naslib_utils
from jahs_bench.tabular.lib.naslib.utils.utils import AttrDict
from jahs_bench.tabular.lib.core.constants import Datasets, standard_task_metrics
from jahs_bench.tabular.lib.core.constants import training_config as _training_config
from jahs_bench.tabular.lib.core import utils
from jahs_bench.tabular.lib.core import datasets as dataset_lib
from jahs_bench.tabular.lib.core.procs import train
from jahs_bench.tabular.search_space import NASB201HPOSearchSpace

# Randomly generated entropy source, to remain fixed across experiments.
_seed = 79029434164686768057103648623012072794


def get_tranining_config_from_args(args: Sequence[str]) -> AttrDict:
    return AttrDict({k: getattr(args, k) for k in _training_config.keys()})


def argument_parser():
    parser = argparse.ArgumentParser()

    # Unpack the training config into CLI flags.
    for k, v in _training_config.items():
        parser.add_argument(f"--{k}", **v)

    parser.add_argument("--basedir", type=Path, default=Path().cwd(),
                        help="Path to the base directory where all the tasks' output will be stored. Task-specific "
                             "sub-directories will be created here if needed.")
    parser.add_argument("--datadir", type=Path, default=dataset_lib.get_default_datadir(),
                        help="The directory where all datasets are expected to be stored.")
    parser.add_argument("--taskid", type=int,
                        help="An offset from 0 for this task within the current node's allocation of tasks.")
    parser.add_argument("--taskid_base", type=int, default=0,
                        help="An additional offset from 0 to manually shift all task IDs further, useful when sampling "
                             "is intended to continue adding more data beyond a previous job's data.")
    parser.add_argument("--dataset", type=str, choices=list(Datasets.__members__.keys()),
                        help="The name of which dataset is to be used for model training and evaluation. Only one of "
                             "the provided choices can be used.")
    parser.add_argument("--nsamples", type=int, default=100,
                        help="Sets the maximum number of samples to be drawn from the search space for each task. Use "
                             "-1 to set this to unlimited.")
    parser.add_argument("--portfolio", type=Path, default=None,
                        help="Path to a saved pandas DataFrame or Series containing a portfolio of configurations. If "
                             "given, the portfolio is read and used to restrict the search space on a per-task basis. "
                             "Key-value pairs specified as CLI arguments for 'opts' override corresponding portfolio "
                             "values.")
    parser.add_argument("--cycle_portfolio", action="store_true",
                        help="Only appliable when a complete portfolio of task- and mode-wise configurations is given. "
                             "When given, causes the portfolio configurations within each task to cycle infinitely or "
                             "until sampling is stopped due to the limit set down by '--nsamples'.")
    parser.add_argument("--global_seed", type=int, default=None,
                        help="A value for a global seed to be used for all global NumPy and PyTorch random "
                             "operations. This is different from the fixed seed used for reproducible search space "
                             "sampling. If not specified, a random source of entropy is used instead.")
    # parser.add_argument("--standalone-mode", action="store_true",
    #                     help="Switch that enables working in a single-task local setup as opposed to the default "
    #                          "multi-node cluster setup.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (very verbose) logging.")
    parser.add_argument("--generate_sampling_profile", action="store_true",
                        help="When given, does not actually train the sampled models. Instead, samples are simply "
                             "repeatedly drawn at 1s intervals and saved in order to build a profile of expected "
                             "samples.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None,
                        help="A variable number of optional keyword arguments provided as 2-tuples, each potentially "
                             "corresponding to a hyper-parameter in the search space. If a match is found, that "
                             "hyper-parameter is excluded from the search space and fixed to the given value instead.")
    return parser


def run_task(basedir: Path, taskid: int, train_config: AttrDict, dataset: Datasets, datadir,
             local_seed: Optional[int] = None, global_seed: Optional[Union[Iterable[int], int]] = None,
             debug: bool = False, generate_sampling_profile: bool = False, nsamples: int = 0,
             portfolio_pth: Optional[Path] = None, cycle_portfolio: bool = False, logger: logging.Logger = None,
             opts: Optional[Sequence[str]] = None):
    """
    Run the sampling, training and evaluation procedures on a single task.

    :param basedir: Path-like
        The base directory in which the directory tree will be created
    :param taskid: int
        The task ID of the task to be run.
    :param train_config: AttrDict
        A dictionary containing various key-value pairs describing the per-model training configuration. Consult
        'get_training_config_help()' and 'get_training_config_from_args()' for more details.
    :param dataset: str
        The name of a known dataset to be used for model training and evaluation.
    :param datadir: Path-like
        The path to a directory where all the datasets are stored.
    :param local_seed: int
        An integer seed for a locally instantiated and controlled RNG that will be passed around to control this task's
        sampling processes. If None (default), local system entropy is used as a seed instead. Note that for
        reproducibility, a fixed local seed should be passed.
    :param global_seed: Iterable over ints or int, optional
        Used for maintaining reproducibility across model training. Could be either a single integer, used to seed the
        global RNGs of PyTorch, NumPy and Random for every sampled model with the same seed, or an iterable that
        generates a global seed for each sampled model. If None (default), the local RNG is used to automatically
        sample a reproducible sequence of global seeds for each model. Consult 'default_global_seed_gen()' for more
        details.
    :param debug: bool
        A flag that, when set, enables debug mode logging and operation. Default: False.
    :param generate_sampling_profile: bool
        A flag that, when set, indicates that only a "dry run" of the sampling procedure should be done in order to
        generate the expected sampling profile of this task without actually training/evaluating the sampled models.
        Also consult 'nsamples'. Default: False.
    :param nsamples: int
        The maximum number of samples to be drawn from the search space for each task. When set to -1, draws an
        unlimited number of samples. Default: 0.
    :param portfolio_pth: Path-like
        Path to a '.pkl.gz' file containing a portfolio of configurations in the form of a compatible Pandas DataFrame.
        If only task-level configurations are given, the search space is restricted on a task-basis such that the given
        task's sampled configurations always conform to the portfolio's specifications. If both task-level and
        model-level configurations are given, the search space isn't sampled at all and instead the precise
        configurations specified in the portfolio are used to generate models. Also see 'cycle_portfolio'.
    :param cycle_portfolio: bool
        Only applicable when a portfolio is provided (see 'portfolio_pth'). When True, the configurations of each task
        are cycled infinitely or until the limit specified by 'nsamples' is reached. Default: False
    :param opts: Sequence of str
        A sequence of strings read as pairs of values that modify the search space such that the first string in each
        pair corresponds to a known parameter in the relevant config space and the second string corresponds to a
        constant value for that parameter, interpreted and type-cast according to the properties of the parameter in
        the defined config space. Default: None.
    :return:
    """

    rng = np.random.RandomState(np.random.Philox(seed=local_seed, counter=taskid))
    global_seed_gen = utils.default_global_seed_gen(rng, global_seed)
    dir_tree = utils.DirectoryTree(basedir=basedir, taskid=taskid)

    if logger is None:
        logger = naslib_logging.setup_logger(str(dir_tree.task_dir / "log.log"))

    task_metrics = AttrDict(
        utils.attrdict_factory(metrics=standard_task_metrics, template=list))

    # The timer-based interface is necessary to synchronize the model metric logger and checkpointer later.
    tasktimer = utils.SynchroTimer()  # This timer must be set manually, but it still uses model_idx as time
    task_metric_logger = utils.MetricLogger(dir_tree=dir_tree, metrics=task_metrics,
                                            set_type=utils.MetricLogger.MetricSet.task, logger=logger, timer=tasktimer)
    n_known_samples = len(task_metrics.model_idx)
    tasktimer.adjust(previous_timestamp=n_known_samples)

    task_config = {
        "train_config": train_config,
        "dataset": dataset.value[0],
        "local_seed": local_seed,
        "global_seed": global_seed,
        "debug": debug,
        "generate_sampling_profile": generate_sampling_profile,
        "nsamples": nsamples,
        "opts": opts
    }
    with open(dir_tree.task_config_file, "w") as fp:
        json.dump(task_config, fp, default=str)

    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    NASB201HPOSearchSpace.NUM_CLASSES = dataset.value[3]
    search_space = NASB201HPOSearchSpace()
    if dataset.value[2] == 1:  # Check # of channels
        # The dataset is in Grayscale
        search_space.GRAYSCALE = True

    sampler = utils.model_sampler(search_space=search_space, taskid=taskid, global_seed_gen=global_seed_gen, rng=rng,
                                  portfolio_pth=portfolio_pth, opts=opts, cycle_models=cycle_portfolio)

    for model_idx, (model, model_config, curr_global_seed) in enumerate(sampler, start=1):
        if nsamples != -1 and model_idx > nsamples:
            break
        logger.info(f"Sampled new architecture: {model_config} from space {search_space.__class__.__name__}")

        ## Model initialization
        model.parse()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Training model {model_idx} on device {str(device)}")
        transfer_devices = device.type != "cpu"
        if transfer_devices:
            model = model.to(device)

        ## Handle task-level metrics
        if model_idx <= n_known_samples:
            # This particular model has already been sampled once. Verify config and seed.
            assert task_metrics.global_seed[model_idx - 1] == curr_global_seed, \
                f"There is a mismatch between the previously registered global seed used for evaluating the " \
                f"model index {model_idx} and the newly generated seed {curr_global_seed}"
            if not task_metrics.model_config[model_idx - 1] == model_config:
                f"Task {taskid}, model {model_idx}: Model config generation mismatch. Old model config: " \
                f"{task_metrics.model_config[model_idx - 1]}. Newly generated config: {model_config}"
            logger.debug(f"Task {taskid}, model {model_idx} has been previously sampled.")
        else:
            # A new model config has been sampled.
            task_metrics.model_idx.append(model_idx)
            task_metrics.model_config.append(model_config)
            task_metrics.global_seed.append(curr_global_seed)
            task_metrics.size_MB.append(naslib_utils.count_parameters_in_MB(model))

            tasktimer.update(timestamp=model_idx, force=True)
            task_metric_logger.log()
            logger.debug(f"Logged new sample for task {taskid}, model {model_idx}.")

        if generate_sampling_profile:
            # time.sleep(1)
            continue

        ## Actual model training and evaluation
        try:
            dir_tree.model_idx = model_idx

            if dir_tree.model_error_description_file.exists():
                os.remove(dir_tree.model_error_description_file)

            naslib_utils.set_seed(curr_global_seed)
            data_loaders, min_shape = dataset_lib.get_dataloaders(
                dataset=dataset, batch_size=train_config.batch_size, cutout=0, split=train_config.split,
                resolution=model_config.get("Resolution", 0), trivial_augment=model_config.get("TrivialAugment", False),
                datadir=datadir
            )
            validate = "valid" in data_loaders
            train(
                model=model, data_loaders=data_loaders, train_config=train_config, dir_tree=dir_tree, logger=logger,
                validate=validate, transfer_devices=transfer_devices, device=device, debug=debug, min_shape=min_shape
            )
        except Exception as e:
            naslib_logging.log_every_n_seconds(logging.INFO, "Architecture Training failed.", 15, name=logger.name)
            error_description = {
                "exception": traceback.format_exc(),
                "config": model_config,
                "global_seed": curr_global_seed
            }
            with open(dir_tree.model_error_description_file, "w") as fp:
                json.dump(error_description, fp, indent=4)
            if debug:
                raise e
        else:
            ## Clean-up after nominal program execution
            logger.info("Sampled architecture trained successfully.")

            del model  # Release memory
            dir_tree.model_idx = None  # Ensure that previously written data cannot be overwritten

    ## Clean up memory before terminating task
    del task_metrics
    del task_metric_logger
    del dir_tree


if __name__ == "__main__":
    ## Parse CLI
    args = argument_parser().parse_args()

    # Pseudo-RNG should rely on a bit-stream that is largely uncorrelated both within and across tasks
    real_taskid = args.taskid_base + args.taskid
    dataset = Datasets.__members__[args.dataset]  # Value checking has already been performed by ArgumentParser

    run_task(basedir=args.basedir, taskid=real_taskid, train_config=get_tranining_config_from_args(args),
             dataset=dataset, datadir=args.datadir, local_seed=_seed, global_seed=args.global_seed, debug=args.debug,
             generate_sampling_profile=args.generate_sampling_profile, cycle_portfolio=args.cycle_portfolio,
             nsamples=args.nsamples, portfolio_pth=args.portfolio, opts=args.opts)
