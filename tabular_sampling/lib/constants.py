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
    fashionMNIST = "FashionMNIST", 28, 1, 10, [0.2860,], [0.3530,], 60_000, 10_000
    # TODO: Fix UC-Merced pixel stats, currently they include test-set stats as well.
    uc_merced = "UC-Merced", 256, 3, 21, [123.47803224419, 124.96319675993959, 114.87820437887096], \
                   [44.238157875221745, 41.69850666766839, 39.646115769885235], 60_000, 10_000

# TODO: Move more constants here so as to save the effort of re-typing and re-checking, say, string values, repeatedly.
