import ConfigSpace as CS
from jahs_bench_201.lib.core.constants import Activations
from typing import Union
from copy import deepcopy
import numpy as np


def get_fidelity_hypercube(selected_fidelities, eta=3):
    hypercube = []
    for fidelity in selected_fidelities:
        if fidelity not in fidelities:
            raise NotImplementedError(f"No such fidelity {fidelity}")
        _fidelity = fidelities[fidelity]
        if isinstance(_fidelity, CS.OrdinalHyperparameter):
            hypercube.append(_fidelity.sequence)
        else:
            max_SH_iter = -int(np.log(_fidelity.lower / _fidelity.upper) / np.log(eta)) + 1
            budgets = _fidelity.upper * np.power(eta,
                                            -np.linspace(max_SH_iter - 1, 0,
                                                         max_SH_iter))
            hypercube.append(tuple(budgets[-int(eta):].astype(int)))
    return hypercube


def get_diagonal(selected_fidelities, eta=3):
    diagonal = []
    hypercube = get_fidelity_hypercube(selected_fidelities, eta)
    diagonal.append([[_hp[i] for _hp in hypercube] for i, _ in enumerate(hypercube[0])])
    return diagonal


fidelities = dict(
    N=CS.OrdinalHyperparameter("N", sequence=[1, 3, 5], default_value=5,
                               meta=dict(help="Number of cell repetitions")),
    W=CS.OrdinalHyperparameter("W", sequence=[4, 8, 16], default_value=16,
                               meta=dict(
                                   help="The width of the first channel in the cell. Each of the "
                                        "subsequent cell's first channels is twice as wide as the "
                                        "previous cell's, thus, for a value 4 (default) of W, the first "
                                        "channel widths are [4, 8, 16].")),
    Epochs=CS.UniformFloatHyperparameter("Epochs", lower=1, upper=200,
                                         default_value=200, log=False,
                                         meta=dict(help="Number of training epochs.")),
    Resolution=CS.OrdinalHyperparameter("Resolution", sequence=[0.25, 0.5, 1.],
                                        default_value=1.,
                                        meta=dict(
                                            help="The sample resolution of the input images w.r.t. one side of the "
                                                 "actual image size, assuming square images, i.e. for a dataset "
                                                 "with 32x32 images, specifying a value of 0.5 corresponds to "
                                                 "using downscaled images of size 16x16 as inputs."))
)

architecture = [
    CS.CategoricalHyperparameter("Op1", choices=list(range(5)), default_value=0,
                                 meta=dict(
                                     help="The operation on the first edge of the cell.")),
    CS.CategoricalHyperparameter("Op2", choices=list(range(5)), default_value=0,
                                 meta=dict(
                                     help="The operation on the second edge of the cell.")),
    CS.CategoricalHyperparameter("Op3", choices=list(range(5)), default_value=0,
                                 meta=dict(
                                     help="The operation on the third edge of the cell.")),
    CS.CategoricalHyperparameter("Op4", choices=list(range(5)), default_value=0,
                                 meta=dict(
                                     help="The operation on the fourth edge of the cell.")),
    CS.CategoricalHyperparameter("Op5", choices=list(range(5)), default_value=0,
                                 meta=dict(
                                     help="The operation on the fifth edge of the cell.")),
    CS.CategoricalHyperparameter("Op6", choices=list(range(5)), default_value=0,
                                 meta=dict(
                                     help="The operation on the sixth edge of the cell.")),
]

hyperparameters = [
    CS.CategoricalHyperparameter("TrivialAugment", choices=[True, False],
                                 default_value=False, meta=dict(
            help="Controls whether or not TrivialAugment is used for pre-processing "
                 "data. If False (default), a set of manually chosen transforms is "
                 "applied during pre-processing. If True, these are skipped in favor of "
                 "applying random transforms selected by TrivialAugment.")),
    CS.CategoricalHyperparameter("Activation",
                                 choices=list(Activations.__members__.keys()),
                                 default_value="ReLU",
                                 meta=dict(
                                     help="Which activation function is to be used for the network. "
                                          "Default is ReLU.")),

    # Add Optimizer related HyperParamters
    CS.CategoricalHyperparameter("Optimizer", choices=["SGD"],
                                 default_value="SGD", meta=dict(
            help="Which optimizer to use for training this model. "
                 "This is just a placeholder for now, to be used "
                 "properly in future versions.")),
    CS.UniformFloatHyperparameter("LearningRate", lower=1e-3, upper=1e0,
                                  default_value=1e-1, log=True, meta=dict(
            help="The learning rate for the optimizer used during model training. In the "
                 "case of adaptive learning rate optimizers such as Adam, this is the "
                 "initial learning rate.")),
    CS.UniformFloatHyperparameter("WeightDecay", lower=1e-5, upper=1e-2,
                                  default_value=5e-4, log=True,
                                  meta=dict(help="Weight decay to be used by the "
                                                 "optimizer during model training.")),

]


def create_config_space(
    use_default_arch: bool = False,
    use_default_hps: bool = False,
    fidelity: Union[str, list] = "Epochs",
    seed=None
):
    default_config = dict()
    joint_config_space = CS.ConfigurationSpace("jahs_bench_config_space", seed=seed)

    # noinspection PyPep8

    _fidelities = deepcopy(fidelities)
    if fidelity is not None:
        if isinstance(fidelity, list):
            for _f in fidelity:
                assert _f in fidelities, f"Other fidelities than {fidelities.keys()} not supported"
                _fidelities.pop(_f)
        else:
            assert fidelity in fidelities, f"Other fidelities than {fidelities.keys()} not supported"
            _fidelities.pop(fidelity)
    _cs = CS.ConfigurationSpace(seed=seed)
    _cs.add_hyperparameters(list(_fidelities.values()))
    default_config.update(
        _cs.get_default_configuration().get_dictionary()
    )

    if use_default_arch:
        _cs = CS.ConfigurationSpace(seed=seed)
        _cs.add_hyperparameters(architecture)
        default_config.update(
            _cs.sample_configuration().get_dictionary()
        )
    else:
        joint_config_space.add_hyperparameters(architecture)

    if use_default_hps:
        _cs = CS.ConfigurationSpace(seed=seed)
        _cs.add_hyperparameters(hyperparameters)
        default_config.update(
            _cs.sample_configuration().get_dictionary()
        )
    else:
        joint_config_space.add_hyperparameters(hyperparameters)

    return default_config, joint_config_space
