""" This file contains a number of constants that are or can be re-used throughout the search space definition. """

import enum
import torch.nn as nn

# Names of cell ops
OP_NAMES = ['Identity', 'Zero', 'ConvBN3x3', 'ConvBN1x1', 'AvgPool1x1']

# 2-tuples, (in-node, out-node), that uniquely identify cell edges
EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))

# Mapping from the original NASBench-201 dataset's naming convention to NASLib's naming convention
nb201_to_ops = {
    'avg_pool_3x3': 'AvgPool1x1',
    'nor_conv_1x1': 'ConvBN1x1',
    'nor_conv_3x3': 'ConvBN3x3',
    'skip_connect': 'Identity',
    'none': 'Zero',
}

# Mapping from NASLib's naming conversion to the original NASBench-201 dataset's naming convention
ops_to_nb201 = {
    'AvgPool1x1': 'avg_pool_3x3',
    'ConvBN1x1': 'nor_conv_1x1',
    'ConvBN3x3': 'nor_conv_3x3',
    'Identity': 'skip_connect',
    'Zero': 'none',
}

@enum.unique
class Activations(enum.Enum):
    ReLU = enum.auto(), nn.ReLU
    # SiLU = enum.auto(), nn.SiLU
    Hardswish = enum.auto(), nn.Hardswish
    Mish = enum.auto(), nn.Mish,
    # LeakyReLU = enum.auto(), nn.LeakyReLU
