import torch.nn as nn
import ConfigSpace as CS
from typing import Union, Optional

from naslib.search_spaces.core.primitives import AbstractPrimitive, Identity


class ConvBN(AbstractPrimitive):

    def __init__(self, C_in, C_out, kernel_size, stride=1, affine=True, use_swish=False, **kwargs):
        super().__init__(locals())
        self.kernel_size = kernel_size
        pad = 0 if stride == 1 and kernel_size == 1 else 1
        self.op = nn.Sequential(
            nn.SiLU(inplace=False) if use_swish else nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )


    def forward(self, x, edge_data):
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += '{}x{}'.format(self.kernel_size, self.kernel_size)
        return op_name


class StemGrayscale(AbstractPrimitive):
    """
    This is used as an initial layer directly after the
    image input in the case of grayscale input images.
    """

    def __init__(self, C_out, **kwargs):
        super().__init__(locals())
        self.seq = nn.Sequential(
            nn.Conv2d(1, C_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_out))

    def forward(self, x, edge_data):
        return self.seq(x)

    def get_embedded_ops(self):
        return None


"""
Code below from NASBench-201 and slighly adapted
@inproceedings{dong2020nasbench201,
  title     = {NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {International Conference on Learning Representations (ICLR)},
  url       = {https://openreview.net/forum?id=HJxyZkBKDr},
  year      = {2020}
}
"""


class ResNetBasicblock(AbstractPrimitive):

    def __init__(self, C_in, C_out, stride, affine=True, use_swish=False):
        super().__init__(locals())
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ConvBN(C_in, C_out, 3, stride, use_swish=use_swish)
        self.conv_b = ConvBN(C_out, C_out, 3, use_swish=use_swish)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False))
        else:
            self.downsample = None


    def forward(self, x, edge_data):
        basicblock = self.conv_a(x, None)
        basicblock = self.conv_b(basicblock, None)
        residual = self.downsample(x) if self.downsample is not None else x
        return residual + basicblock
    

    def get_embedded_ops(self):
        return None


class HPOBlock(Identity):
    """
    A NoOp block that enables sampling a hyper-parameter configuration from within a naslib object.
    """

    # TODO: Finalize how this block is to be initialized.
    # Options for initialization:
    # 1. Pass in entire ConfigSpace object -- entirely dynamic configuration space
    # 2. Pass in a list of ConfigSpace Hyperparameter objects -- mostly dynamic configuration space
    # 3. Pass in specific Python primitives that will only enable/disable sampling of certain hyperparameters -- rigid
    #     configuration space.
    # For now, use option 3 for testing and initial implementation

    config_space: CS.ConfigurationSpace
    configuration: CS.Configuration

    def __init__(self, learning_rate: Optional[float] = None, weight_decay: Optional[float] = None,
                 momentum: Optional[float] = None, **kwargs):
        # Parameters can be either None, in which case they are sampled from pre-specified ranges, or be given a
        # specific value.
        super().__init__(**dict({"learning_rate": learning_rate, "weight_decay": weight_decay, "momentum": momentum},
                                **kwargs))
        self.config_space = CS.ConfigurationSpace("NASB201_HPOBlock")
        known_parameters = {
            "learning_rate": CS.UniformFloatHyperparameter("learning_rate", 0.001, 1.0, default_value=0.01, log=True,
                                                           q=0.001),
            "weight_decay": CS.UniformFloatHyperparameter("weight_decay", 0.001, 1.0, default_value=0.001, log=True,
                                                          q=0.001),
            "momentum": CS.UniformFloatHyperparameter("momentum", 0.01, 1.0, default_value=0.9, log=False, q=0.01),
        }

        loc = locals()

        for param, default in known_parameters.items():
            if loc[param] is None:
                self.config_space.add_hyperparameter(default)
            else:
                self.config_space.add_hyperparameter(CS.Constant(param, loc[param]))

        self.configuration = None