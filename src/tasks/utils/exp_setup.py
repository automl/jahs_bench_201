from pathlib import Path
import argparse
import os

import numpy as np
import random


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


parser = argparse.ArgumentParser(description="Experiment runner")
parser.add_argument(
    "--dataset",
    default="cifar10",
    help="The benchmark dataset to run the experiments.",
    choices=["cifar10", "colorectal_histology", "fashionMNIST"],
)
parser.add_argument(
    "--run_id",
    default="jahs_bench",
    help="run_id",
)
parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="host",
)
parser.add_argument(
    "--fidelity",
    default="Epochs",
    nargs='+',
    help="fidelity.",
    choices=["Epochs", "N", "W", "Resolution", "None"],
)
parser.add_argument(
    "--model_path",
    default=Path(
    __file__).parent.parent.parent.parent / "JAHS-Bench-MF" / "trained_surrogate_models" / "hpo_models"
,
    help="Full path to model dir",
)
parser.add_argument(
    "--working_directory",
    default=os.path.dirname(os.path.realpath(__file__)) + "/../../results/",
    help="Full path to model dir",
)
parser.add_argument("--n_iterations", type=int, default=42, help="n_iterations")
parser.add_argument("--min_budget", type=int, default=12, help="min fidelity value")
parser.add_argument("--max_budget", type=int, default=200, help="max fidelity value")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--use_default_arch", action="store_true", help="Whether to use default arch"
)
parser.add_argument(
    "--use_default_hps", action="store_true", help="Whether to use default hps"
)
parser.add_argument(
    "--eta",
    default=3,
    type=float,
    help="Eta parameter for SH",
)
parser.add_argument(
    "--use_model", action="store_true", help="Whether to use BOHB"
)
parser.add_argument(
    "--no_surrogate", action="store_true", help="Whether to use surrogate benchmark"
)

args = parser.parse_args()
if len(args.fidelity) == 1:
    args.fidelity = args.fidelity[0]
if args.fidelity == "None":
    args.fidelity = None
