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

# TODO: Convert this script into a re-useable library file, with a bare minimum runner script replacement here
# TODO: Extract this entire tree out of the original NASLib repo (see "git subtree") and into its own repo

import argparse
import json
import logging
import time
from itertools import repeat
from pathlib import Path
import numpy as np
import torch

import naslib.utils.logging as naslib_logging
import naslib.utils.utils as naslib_utils
from naslib.utils.utils import AttrDict
from tabular_sampling.search_space import NASB201HPOSearchSpace
from tabular_sampling.lib import utils
from tabular_sampling.lib.count_flops import get_model_flops
from tabular_sampling.lib.procs import train

# Randomly generated entropy source, to remain fixed across experiments.
seed = 79029434164686768057103648623012072794


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=Path, default=Path().cwd(),
                        help="Path to the base directory where all the tasks' output will be stored. Task-specific "
                             "sub-directories will be created here if needed.")
    parser.add_argument("--taskid", type=int,
                        help="An offset from 0 for this task within the current node's allocation of tasks.")
    parser.add_argument("--taskid_base", type=int, default=0,
                        help="An additional offset from 0 to manually shift all task IDs further, useful when sampling "
                             "is intended to continue adding more data beyond a previous job's data.")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of epochs that each sampled architecture should be trained for. Default: 25")
    parser.add_argument("--global_seed", type=int, default=None,
                        help="A value for a global seed to be used for all global NumPy and PyTorch random "
                             "operations. This is different from the fixed seed used for reproducible search space "
                             "sampling. If not specified, a random source of entropy is used instead.")
    parser.add_argument("--standalone-mode", action="store_true",
                        help="Switch that enables working in a single-task local setup as opposed to the default "
                             "multi-node cluster setup.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (very verbose) logging.")
    parser.add_argument("--use_grad_clipping", action="store_true", help="Enable gradient clipping for SGD.")
    parser.add_argument("--split", action="store_true",
                        help="Split training dataset into training and validation sets.")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of samples per mini-batch.")
    parser.add_argument("--warmup_epochs", type=int, default=0.,
                        help="When set to a positive integer, this many epochs are used to warm-start the training.")
    parser.add_argument("--generate_sampling_profile", action="store_true",
                        help="When given, does not actually train the sampled models. Instead, only samples are "
                             "repeatedly drawn at 1s intervals and saved in order to build a profile of expected "
                             "samples.")
    parser.add_argument("--nsamples", type=int, default=100,
                        help="Only used when --generate_sampling_profile is given. Sets the number of samples to be "
                             "drawn from the search space.")
    parser.add_argument("--disable_checkpointing", action="store_true",
                        help="When given, checkpointing of model training is disabled. By default, model training is "
                             "checkpointed either every X seconds or Y epochs, whichever occurs first. Check "
                             "--checkpoint_interval_seconds and --checkpoint_interval_epochs.")
    parser.add_argument("--checkpoint_interval_seconds", type=int, default=1800,
                        help="The time interval between subsequent model training checkpoints, in seconds. Default: "
                             "30 minutes i.e. 1800 seconds.")
    parser.add_argument("--checkpoint_interval_epochs", type=int, default=20,
                        help="The interval between subsequent model training checkpoints, in epochs. Default: "
                             "20 epochs.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None,
                        help="A variable number of optional keyword arguments provided as 2-tuples, each potentially "
                             "corresponding to a hyper-parameter in the search space. If a match is found, that "
                             "hyper-parameter is excluded from the search space and fixed to the given value instead.")
    return parser


if __name__ == "__main__":

    ## Parse CLI
    args = argument_parser().parse_args()
    basedir = args.basedir
    global_seed = args.global_seed

    # Pseudo-RNG should rely on a bit-stream that is largely uncorrelated both within and across tasks
    real_taskid = args.taskid_base + args.taskid
    rng = np.random.RandomState(np.random.Philox(seed=seed, counter=real_taskid))

    dir_tree = utils.DirectoryTree(basedir=basedir, taskid=real_taskid)

    if global_seed is None:
        def seeds():
            while True:
                yield rng.randint(0, 2 ** 32 - 1)


        global_seed = seeds()
    else:
        global_seed = repeat(global_seed)

    logger = naslib_logging.setup_logger(str(dir_tree.task_dir / "log.log"))
    task_metrics = AttrDict({
        "model_idx": [],
        "model_config": [],
        "global_seed": [],
        "size_MB": [],
        "FLOPS": [],
    })
    task_metric_logger = utils.MetricLogger(dir_tree=dir_tree, metrics=task_metrics, log_interval=None,
                                            set_type=utils.MetricLogger.MetricSet.task, logger=logger)
    n_known_samples = len(task_metrics.model_idx)

    with open(dir_tree.task_dir / "task_config.json", "w") as fp:
        json.dump(vars(args), fp, default=str)

    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    naslib_utils.log_args(vars(args))
    search_space = NASB201HPOSearchSpace()

    if utils.adapt_search_space(search_space, args.opts):
        logger.info(f"Modified original search space's hyperparameter config space using constant values. "
                    f"New config space: \n{search_space.config_space}")


    def sampler():
        while True:
            curr_global_seed = next(global_seed)
            naslib_utils.set_seed(curr_global_seed)
            model: NASB201HPOSearchSpace = search_space.clone()
            model.sample_random_architecture(rng=rng)
            model_config = model.config.get_dictionary()
            yield model, model_config, curr_global_seed


    if args.generate_sampling_profile:
        sampled_configs = []

    train_config = AttrDict(vars(args))

    for model_idx, (model, model_config, curr_global_seed) in enumerate(sampler(), start=1):
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
            assert task_metrics.model_config[model_idx - 1] == model_config, \
                f"There is a mismatch between the previously registered model config used for evaluating the " \
                f"model index {model_idx} and the newly generated config {model_config}"
        else:
            # A new model config has been sampled.
            task_metrics.model_idx.append(model_idx)
            task_metrics.model_config.append(model_config)
            task_metrics.global_seed.append(curr_global_seed)
            task_metrics.size_MB.append(naslib_utils.count_parameters_in_MB(model))
            task_metrics.FLOPS.append(get_model_flops(
                model=model,
                input_shape=(train_config.batch_size, 3, model_config["Resolution"] or 32,
                             model_config["Resolution"] or 32),
                transfer_devices=transfer_devices,
                device=device
            ))
            task_metric_logger.log(elapsed_runtime=model_idx, force=True)

        if args.generate_sampling_profile:
            time.sleep(1)

            if model_idx >= args.nsamples:
                break
            else:
                continue

        ## Actual model training and evaluation
        try:
            dir_tree.model_idx = model_idx
            naslib_utils.set_seed(curr_global_seed)
            data_loaders, _, _ = utils.get_dataloaders(dataset="cifar10", batch_size=args.batch_size, cutout=0,
                                                       split=args.split, resize=model_config.get("Resolution", 0),
                                                       trivial_augment=model_config.get("TrivialAugment", False))
            validate = "valid" in data_loaders
            train(
                model=model, data_loaders=data_loaders,
                train_config=train_config, dir_tree=dir_tree,
                logger=logger, validate=validate,
                transfer_devices=transfer_devices, device=device
            )
        except Exception as e:
            naslib_logging.log_every_n_seconds(logging.INFO, "Architecture Training failed.", 15, name=logger.name)
            error_description = {
                "exception": str(e),
                "config": model_config,
                "global_seed": curr_global_seed
            }
            with open(dir_tree.model_dir / f"error_description.json", "w") as fp:
                json.dump(error_description, fp, indent=4)
            if args.debug:
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
