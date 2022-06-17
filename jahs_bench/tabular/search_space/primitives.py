import torch.nn as nn
import ConfigSpace as CS
from typing import Union, Optional

from jahs_bench.tabular.lib.naslib.search_spaces.core.primitives import \
    AbstractPrimitive, Identity
from .constants import Activations


class ConvBN(AbstractPrimitive):

    def __init__(self, C_in, C_out, kernel_size, stride=1, affine=True, activation=Activations.ReLU.name, **kwargs):
        super().__init__(locals())
        self.kernel_size = kernel_size
        pad = 0 if stride == 1 and kernel_size == 1 else 1
        activation_fn = Activations.__members__[activation].value[1]
        self.op = nn.Sequential(
            # nn.SiLU(inplace=False) if use_swish else nn.ReLU(inplace=False),
            activation_fn(inplace=False),
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
Code below from NASBench-201 and adapted
@inproceedings{dong2020nasbench201,
  title     = {NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {International Conference on Learning Representations (ICLR)},
  url       = {https://openreview.net/forum?id=HJxyZkBKDr},
  year      = {2020}
}
"""


class ResNetBasicblock(AbstractPrimitive):

    def __init__(self, C_in, C_out, stride, affine=True, activation=Activations.ReLU.name):
        super().__init__(locals())
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ConvBN(C_in, C_out, 3, stride, activation=activation)
        self.conv_b = ConvBN(C_out, C_out, 3, activation=activation)
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
