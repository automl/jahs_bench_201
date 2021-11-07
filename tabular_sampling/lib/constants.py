""" A collection of constant values that are or may be re-used throughout the rest of the code base. """

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

# TODO: Move more constants here so as to save the effort of re-typing and re-checking, say, string values, repeatedly.
