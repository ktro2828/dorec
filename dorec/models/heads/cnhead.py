#!/usr/bin/env python

import torch.nn.functional as F

from dorec.core import HEADS

from .head_base import HeadBase
from ..layers import conv3x3_nlayers


@HEADS.register()
class CNHead(HeadBase):
    """N layers convolutional module

    Args:
        num_classes (int, optional)
        fc_dim (int, optional)
        use_softmax (bool, optional)
        deep_sup (bool, optional)
    """

    def __init__(self, num_layers, num_classes, fc_dim, deep_sup=False):
        super(CNHead, self).__init__()
        self.deep_sup = deep_sup

        self.cbr = conv3x3_nlayers(fc_dim, fc_dim // 4, num_classes,
                                   num_layers=num_layers, stride=1)

        if self.deep_sup:
            self.cbr_ds = conv3x3_nlayers(
                fc_dim // 2, fc_dim // 4, num_classes,
                num_layers=num_layers, stride=1)

        # Weight init
        self.apply(self.init_weight)

    def forward(self, conv_out, segSize=None):
        feature = conv_out[-1]
        x = self.cbr(feature)

        if segSize is not None:
            x = F.interpolate(
                x, size=segSize, mode="bilinear", align_corners=False)

        if self.deep_sup:
            outs = [x]
            for fe in conv_out[:-1]:
                x_ds = self.cbr_ds(fe)
                if segSize is not None:
                    x_ds = F.interpolate(
                        x_ds, size=segSize,
                        mode="bilinear", align_corners=False)
                outs.append(x_ds)
            return outs
        return x
