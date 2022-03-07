""" A collection of constant values that are or may be re-used throughout the rest of the code base. """

from enum import Enum

training_config = {
    "epochs": dict(type=int, default=25,
                   help="Number of epochs that each sampled architecture should be "
                        "trained for. Default: 25"),
    "batch_size": dict(type=int, default=256, help="Number of samples per mini-batch."),
    "use_grad_clipping": dict(action="store_true",
                              help="Enable gradient clipping for SGD."),
    "split": dict(action="store_true",
                  help="Split training dataset into training and validation sets."),
    "warmup_epochs": dict(type=int, default=0,
                          help="When set to a positive integer, this many epochs are "
                               "used to warm-start the training."),
    "disable_checkpointing": dict(
        action="store_true",
        help="When given, overrides the values of --checkpoint_interval_seconds and "
             "--checkpoint_internal_epochs and disables checkpointing of model weights. "
             "Normally, model weights and meitrcs are checkpointed either every X epochs "
             "or Y seconds. Check --checkpoint_interval_seconds and "
             "--checkpoint_interval_epochs for more information."),
    "checkpoint_interval_seconds": dict(
        type=int, default=None,
        help="The time interval between subsequent model training checkpoints and "
             "metrics, in seconds. When not specified (default), model training "
             "checkpointing at regular intervals of time is turned off. When specified "
             "along with --checkpoint_interval_epochs, both checkpointing intervals are "
             "enabled simultaneously, i.e. the model training will be logged either "
             "every X epochs and Y seconds, depending on which condition is satisfied "
             "first. On the flip side, if neither --checkpoint_interval_epochs nor "
             "--checkpoint_interval_seconds is specified, logging of model training is "
             "disabled entirely. Even if either or both these values are specified, "
             "checkpointing of model weights can be forcefully disabled by the flag "
             "--disable_checkpointing."),
    "checkpoint_interval_epochs": dict(
        type=int, default=None,
        help="The interval between subsequent model training checkpoints and metrics, "
             "in epochs. When not specified (default), model training checkpointing at a "
             "regular number of epochs is turned off. When specified along with "
             "--checkpoint_interval_seconds, both checkpointing intervals are enabled "
             "simultaneously, i.e. the model training will be logged either every X "
             "epochs and Y seconds, depending on which condition is satisfied first. On "
             "the flip side, if neither --checkpoint_interval_epochs nor "
             "--checkpoint_interval_seconds is specified, logging of model training is "
             "disabled entirely. Even if either or both these values are specified, "
             "checkpointing of model weights can be forcefully disabled by the flag "
             "--disable_checkpointing.")
}

datasets = ["cifar10", "colorectal_histology"]

## Metric DataFrame constants
class MetricDFIndexLevels(Enum):
    taskid = "taskid"
    modelid = "model_idx"
    epoch = "Epoch"

metricdf_column_level_names = ["MetricType", "MetricName"]
metricdf_index_levels = list(MetricDFIndexLevels.__members__.keys())

## Metric metadata constants
standard_task_metrics = \
    [MetricDFIndexLevels.modelid.value, "model_config", "global_seed", "size_MB"]
standard_model_dataset_metrics = \
    ["duration", "data_load_duration", "forward_duration", "loss", "acc"]
extra_model_training_metrics = ["backprop_duration"]
standard_model_diagnostic_metrics = \
    ["FLOPS", "latency", "runtime", "cpu_percent", "memory_ram", "memory_swap"]

fidelity_types = {"N": int, "W": int, "Resolution": float}
fidelity_params = tuple(fidelity_types.keys())
