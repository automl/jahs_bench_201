""" A collection of constant values that are or may be re-used throughout the rest of the code base. """

from enum import Enum

training_config = {
    "epochs": dict(type=int, default=25,
                   help="Number of epochs that each sampled architecture should be trained for. Default: 25"),
    "batch_size": dict(type=int, default=256, help="Number of samples per mini-batch."),
    "use_grad_clipping": dict(action="store_true", help="Enable gradient clipping for SGD."),
    "split": dict(action="store_true", help="Split training dataset into training and validation sets."),
    "warmup_epochs": dict(type=int, default=0.,
                          help="When set to a positive integer, this many epochs are used to warm-start the training."),
    "disable_checkpointing": dict(
        action="store_true",
        help="When given, checkpointing of model training is disabled. By default, model training is checkpointed "
             "either every X seconds or Y epochs, whichever occurs first. Check --checkpoint_interval_seconds and "
             "--checkpoint_interval_epochs."),
    "checkpoint_interval_seconds": dict(
        type=int, default=1800,
        help="The time interval between subsequent model training checkpoints, in seconds. Default: 30 minutes i.e. "
             "1800 seconds."),
    "checkpoint_interval_epochs": dict(
        type=int, default=20, help="The interval between subsequent model training checkpoints, in epochs. Default: "
                                   "20 epochs.")
}


class Datasets(Enum):
    """ Constants corresponding to datasets, each value is a 4 tuple corresponding to the format:
        name: str,
        image_size: int,
        nchannels: int,
        nclasses: int,
        mean: tuple(int, int, int) or tuple(int),
        std: tuple(int, int, int) or tuple(int),
        expected_train_size: int,
        expected_test_size: int,
    """

    cifar10 = "Cifar-10", 32, 3, 10, [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616], 50_000, 10_000
    fashionMNIST = "FashionMNIST", 28, 1, 10, [0.2860, ], [0.3530, ], 60_000, 10_000

    # For ICGen datasets, the mean and std will be read from the accompanying JSON file and not stored statically.
    # This is an ugly hack, but was necessary at the time. In future iterations, either Cifar10/fashionMNIST will also
    # be read as ICGen datasets or a different approach will be adopted than this Enum.
    uc_merced = "UC-Merced", 32, 3, 21, [0., 0., 0.], [1., 1., 1.], 1_890, 210
    colorectal_histology = "Colorectal Histology", 32, 3, 8, [0., 0., 0.], [1., 1., 1.], 4_504, 496


icgen_datasets = (Datasets.uc_merced, Datasets.colorectal_histology)

## Metric DataFrame constant
class MetricDFIndexLevels(Enum):
    # taskid = "TaskIndex" # intended new, will require lots of modifications to AttrDict member accesses
    # modelid = "ModelIndex" # intended new
    taskid = "taskid" # old, to be replaced
    modelid = "model_idx" # old, to be replaced
    epoch = "Epoch"

metricdf_column_level_names = ["MetricType", "MetricName"]
metricdf_index_levels = list(MetricDFIndexLevels.__members__.keys())

## Metric metadata constants
standard_task_metrics = [MetricDFIndexLevels.modelid.value, "model_config", "global_seed", "size_MB"]
standard_model_dataset_metrics = ["duration", "data_load_duration", "forward_duration", "loss", "acc"]
extra_model_training_metrics = ["backprop_duration"]
standard_model_diagnostic_metrics = ["FLOPS", "latency", "runtime", "cpu_percent", "memory_ram", "memory_swap"]

fidelity_types = {"N": int, "W": int, "Resolution": float}
fidelity_params = tuple(fidelity_types.keys())
# TODO: Move more constants here so as to save the effort of re-typing and re-checking, say, string values, repeatedly.
