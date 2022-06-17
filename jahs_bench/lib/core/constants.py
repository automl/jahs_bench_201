""" A collection of constant values that are or may be re-used throughout the rest of the
code base. """

import enum

datasets = ["cifar10", "colorectal_histology", "fashion_mnist"]
fidelity_types = {"N": int, "W": int, "Resolution": float}
fidelity_params = tuple(fidelity_types.keys())

## The following are a number of constants that are related to the search space definition

# Names of cell ops
OP_NAMES = ['Identity', 'Zero', 'ConvBN3x3', 'ConvBN1x1', 'AvgPool1x1']

# 2-tuples, (in-node, out-node), that uniquely identify cell edges
EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))


# Available choices for activation function.

@enum.unique
class Activations(enum.Enum):
    ReLU = enum.auto()
    Hardswish = enum.auto()
    Mish = enum.auto()


# Mapping from the original NASBench-201 dataset's naming convention to NASLib's naming
# convention
nb201_to_ops = {
    'avg_pool_3x3': 'AvgPool1x1',
    'nor_conv_1x1': 'ConvBN1x1',
    'nor_conv_3x3': 'ConvBN3x3',
    'skip_connect': 'Identity',
    'none': 'Zero',
}

# Mapping from NASLib's naming conversion to the original NASBench-201 dataset's naming
# convention
ops_to_nb201 = {
    'AvgPool1x1': 'avg_pool_3x3',
    'ConvBN1x1': 'nor_conv_1x1',
    'ConvBN3x3': 'nor_conv_3x3',
    'Identity': 'skip_connect',
    'Zero': 'none',
}
