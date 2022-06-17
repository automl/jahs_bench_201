import os
import pickle

import ConfigSpace
import numpy as np
import random
import torch
import torch.nn as nn

from jahs_bench.tabular.lib.naslib.search_spaces.core import primitives as core_ops
from jahs_bench.tabular.lib.naslib.search_spaces.core.graph import Graph, EdgeData
from jahs_bench.tabular.lib.naslib.search_spaces.core.primitives import AbstractPrimitive
from jahs_bench.tabular.lib.naslib.search_spaces.core.query_metrics import Metric

from jahs_bench.tabular.lib.naslib.utils.utils import get_project_root

from .conversions import convert_op_indices_to_naslib, convert_naslib_to_op_indices, convert_naslib_to_str
from .primitives import ResNetBasicblock, ConvBN, StemGrayscale
from .configspace import joint_config_space
from .constants import OP_NAMES, Activations


class NASB201HPOSearchSpace(Graph):
    """
    Implementation of the nasbench 201 + HPO search space.
    """

    OPTIMIZER_SCOPE = [
        "stage_1",
        "stage_2",
        "stage_3",
    ]

    QUERYABLE = False

    # When this is True, newly created graphs (class instances) are immediately compiled into PyTorch models
    COMPILE_ON_CREATE = True

    GRAYSCALE = False # TODO: Verify implementation
    NUM_CELL_STAGES = 3 # Cannot be less than 1, abbreviated as S TODO: Finish and verify implementation
    KNOWN_CHANNEL_WIDTHS = {
        1: (4, 8, 16),
        2: (8, 16, 32),
        3: (16, 32, 64),
    }

    config_space = joint_config_space
    _config = joint_config_space.get_default_configuration()

    def __init__(self):
        # We re-define the way SearchSpaces are typically initialized by NASLib in order to accommodate for the greater
        # dynamic flexibility in graph design required by HPO, which may include fidelity parameters. Thus, the
        # initializer now only sets a number of fixed parameters that provide some general information about the
        # search space and calls a separate function to instantiate a graph. Thus, the graph can be cleared and
        # re-constructed using different parameters any time by calling the relevant function again.

        super().__init__()
        self.num_classes = self.NUM_CLASSES if hasattr(self, 'NUM_CLASSES') else 10
        self.op_indices = None

        self.space_name = 'nasbench201_hpo'
        self._construct_graph()

    def _construct_graph(self):
        assert self.NUM_CELL_STAGES >= 1, f"The search space must have at least one cell stage but " \
                                          f"{self.NUM_CELL_STAGES} stages were specified."

        cell_repeat = self.config.get("N")
        # channels = self.KNOWN_CHANNEL_WIDTHS[self.config.get("W")]
        channels = tuple(self.config.get("W") * (2 ** i) for i in range(3))
        activation = self.config.get("Activation")

        #
        # Cell definition
        #
        cell = Graph()
        cell.name = "cell"  # Use the same name for all cells with shared attributes

        # Input node
        cell.add_node(1)

        # Intermediate nodes
        cell.add_node(2)
        cell.add_node(3)

        # Output node
        cell.add_node(4)

        # Edges
        cell.add_edges_densly()

        #
        # Makrograph definition
        #
        self.name = "makrograph"

        # Cell is on the edges
        # 1-2:                                      Preprocessing
        # 2-3, ..., (2+N-1)-(2+N):                  cells stage 1
        #
        # (2+N)-(3+N):                              residual block stride 2
        # (3+N)-(3+N+1), ..., (3+2*N-1)-(3+2*N):    cells stage 2
        #
        # (3+2*N)-(4+2*N):                          residual block stride 2
        # (4+2*N)-(4+2*N+1), ..., (4+3*N-1)-(4+3*N): cells stage 3
        # .
        # .
        # .
        # (1+S*(N+1))-(1+S*(N+1)+1):                          post-processing

        total_num_nodes = 1 + self.NUM_CELL_STAGES * (1 + cell_repeat) + 1
        self.add_nodes_from(range(1, total_num_nodes + 1))
        self.add_edges_from([(i, i + 1) for i in range(1, total_num_nodes)])

        edge_names = {}

        edge_names["preproc"] = (1, 2)
        edge_names["stage1"] = [(i, i + 1) for i in range(2, 2 + cell_repeat)]

        # TODO: Make the number of stages programmatically dynamic
        edge_names["res1"] = (edge_names["stage1"][-1][-1], edge_names["stage1"][-1][-1] + 1)
        edge_names["stage2"] = [(i, i + 1) for i in range(edge_names["res1"][1], edge_names["res1"][1] + cell_repeat)]
        edge_names["res2"] = (edge_names["stage2"][-1][1], edge_names["stage2"][-1][1] + 1)
        edge_names["stage3"] = [(i, i + 1) for i in range(edge_names["res2"][1], edge_names["res2"][1] + cell_repeat)]
        edge_names["postproc"] = (edge_names["stage3"][-1][1], edge_names["stage3"][-1][1] + 1)

        #
        # operations at the edges
        #

        # preprocessing
        self.edges[edge_names["preproc"]].set(
            'op', StemGrayscale(channels[0]) if self.GRAYSCALE else core_ops.Stem(channels[0]))

        # stage 1
        for e in edge_names["stage1"]:
            self.edges[e].set('op', cell.copy().set_scope('stage_1'))

        # stage 2
        self.edges[edge_names["res1"]].set('op', ResNetBasicblock(C_in=channels[0], C_out=channels[1], stride=2,
                                                                  activation=activation))
        for e in edge_names["stage2"]:
            self.edges[e].set('op', cell.copy().set_scope('stage_2'))

        # stage 3
        self.edges[edge_names["res2"]].set('op', ResNetBasicblock(C_in=channels[1], C_out=channels[2], stride=2,
                                                                  activation=activation))
        for e in edge_names["stage3"]:
            self.edges[e].set('op', cell.copy().set_scope('stage_3'))

        # post-processing
        self.edges[edge_names["postproc"]].set('op', core_ops.Sequential(
            nn.BatchNorm2d(channels[-1]),
            # nn.SiLU(inplace=False) if use_swish else nn.ReLU(inplace=False),
            Activations.__members__[activation].value[1](inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], self.num_classes)
        ))

        self.edge_names = edge_names

        # set the ops at the cells (channel dependent)
        for c, scope in zip(channels, self.OPTIMIZER_SCOPE):
            self.update_edges(
                update_func=lambda edge: _set_cell_ops(edge, C=c, activation=activation),
                scope=scope,
                private_edge_data=True
            )

        op_indices = [self.config.get(f"Op{i}") for i in range(1, 7)]
        self.set_op_indices(op_indices)
        if self.COMPILE_ON_CREATE:
            self.compile()

    @property
    def config(self) -> ConfigSpace.Configuration:
        return self._config

    @config.setter
    def config(self, val: ConfigSpace.Configuration):
        try:
            self.config_space.check_configuration(val)
        except Exception as e:
            raise RuntimeError(f"The given config value could not be verified against the configuration space."
                               f"\nConfiguration space: {str(self.config_space)}\nGiven config: {str(val)}") \
                from e
        else:
            self._config = val

    def get_op_indices(self):
        if self.op_indices is None:
            self.op_indices = convert_naslib_to_op_indices(self)
        return self.op_indices

    def get_hash(self):
        return tuple(self.config.get_dictionary().items())

    def set_op_indices(self, op_indices):
        # This will update the edges in the naslib object to op_indices
        self.op_indices = op_indices
        convert_op_indices_to_naslib(op_indices, self)

    def sample_random_architecture(self, dataset_api=None, rng=None):
        """
        This will sample a random architecture from the search space. Sampling here means that first a fidelity will be
        sampled from the fidelity space, a new graph will be constructed, random indices for the graph will be sampled
        and finally edges in the naslib object will be updated accordingly. Fidelity sampling can be turned off by
        setting sample_fidelity to False (default: True). If fidelity sampling is off, the naslib object's fidelity
        attribute must be set to a valid fidelity value before this function is called. 'rng' can be either None, in
        which case a fresh random number generator is used, or an integer value used to seed a new instance of
        numpy.random.default_rng() or any compatible random state for the same, including an existing RNG to ensure
        consistent, reproducible results.
        """
        self.clear()
        if not isinstance(rng, np.random.RandomState):
            # Assume that rng is either None or a compatible source of entropy
            rng = np.random.RandomState(rng)
        self.config_space.random = rng
        self.config = self.config_space.sample_configuration()
        self._construct_graph()

    def get_type(self):
        return 'nasbench201_hpo'


def _set_cell_ops(edge, C, activation=Activations.ReLU.name):
    edge.data.set('op', [
        core_ops.Identity(),
        core_ops.Zero(stride=1),
        ConvBN(C, C, kernel_size=3, activation=activation),
        ConvBN(C, C, kernel_size=1, activation=activation),
        core_ops.AvgPool1x1(kernel_size=3, stride=1),
    ])
