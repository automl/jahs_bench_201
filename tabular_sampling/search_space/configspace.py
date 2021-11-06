import ConfigSpace as CS

joint_config_space = CS.ConfigurationSpace("NASB201HPO_joint_config_space")
joint_config_space.add_hyperparameters([
    CS.OrdinalHyperparameter("N", sequence=[1, 3, 5], default_value=1,meta=dict(help="Number of cell repetitions")),
    CS.OrdinalHyperparameter("W", sequence=list(range(1, 4)), default_value=1,
                             meta=dict(help="The width of the three channels in the cell. The value of this "
                                            "parameter corresponds to one of three levels:\n1 - (4, 8, 16)\n"
                                            "2 - (8, 16, 32)\n3 - (16, 32, 64)")),
    CS.CategoricalHyperparameter("Op1", choices=list(range(5)), default_value=0,
                                 meta=dict(help="The operation on the first edge of the cell.")),
    CS.CategoricalHyperparameter("Op2", choices=list(range(5)), default_value=0,
                                 meta=dict(help="The operation on the second edge of the cell.")),
    CS.CategoricalHyperparameter("Op3", choices=list(range(5)), default_value=0,
                                 meta=dict(help="The operation on the third edge of the cell.")),
    CS.CategoricalHyperparameter("Op4", choices=list(range(5)), default_value=0,
                                 meta=dict(help="The operation on the fourth edge of the cell.")),
    CS.CategoricalHyperparameter("Op5", choices=list(range(5)), default_value=0,
                                 meta=dict(help="The operation on the fifth edge of the cell.")),
    CS.CategoricalHyperparameter("Op6", choices=list(range(5)), default_value=0,
                                 meta=dict(help="The operation on the sixth edge of the cell.")),
    CS.OrdinalHyperparameter("Resolution", sequence=[8, 16, 32], default_value=32,
                             meta=dict(help="The size of the input image, such that any input image is resized to, "
                                            "say, 16 x 16, during pre-processing if a resolution of 16 is selected.")),
    CS.CategoricalHyperparameter("TrivialAugment", choices= [True, False], default_value=False,
                                 meta=dict(help="Controls whether or not TrivialAugment is used for pre-processing "
                                                "data. If False (default), a set of manually chosen transforms is "
                                                "applied during pre-processing. If True, these are skipped in favor "
                                                "of applying random transforms selected by TrivialAugment.")),
    CS.CategoricalHyperparameter("Swish", choices= [True, False], default_value=False,
                                 meta=dict(help="Controls whether or not the Swish actiation function should be used "
                                                "in the network. When False (default), the ReLU activation is used. "
                                                "When True, all activations are replaced by Swish-1."))
])

# Add Optimizer related HyperParamters
optimizers = CS.CategoricalHyperparameter("Optimizer", choices=["AdamW", "SGD"], default_value="SGD",
                                          meta=dict(help="Which optimizer to use for training this model. SGD refers "
                                                         "to SGD with Nesterov momentum and a momentum of 0.9. AdamW "
                                                         "refers to AdamW with betas fixed to 0.9 and 0.999."))
lr = CS.UniformFloatHyperparameter("LearningRate", lower=1e-5, upper=1e-1, default_value=1e-2, log=True,
                                   meta=dict(help="The learning rate for the optimizer used during model training. "
                                                  "In the case of adaptive learning rate optimizers such as Adam, "
                                                  "this is the initial learning rate."))
weight_decay = CS.UniformFloatHyperparameter("WeightDecay", lower=1e-5, upper=1e-3, default_value=1e-4,
                                             meta=dict(help="Weight decay to be used by the optimizer during model "
                                                            "training."))

joint_config_space.add_hyperparameters([optimizers, lr, weight_decay])
