#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from dorec.core import HEADS
from dorec.models.layers import BatchNorm2d, conv3x3_bn_relu

from .head_base import HeadBase


@HEADS.register()
class PPM(HeadBase):
    """Pylamid Pooling Module

    Args:
        num_classes (int)
        fc_dim (int)
        pool_scales (tuple[int], optional)
        deep_sup (bool, optional)
    """

    def __init__(self, num_classes, fc_dim, pool_scales=(1, 2, 3, 6), deep_sup=False):
        super(PPM, self).__init__(deep_sup)
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                    BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
            )
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales) * 512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )

        if self.deep_sup:
            self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
            self.conv_last_deepsup = nn.Conv2d(
                fc_dim // 4, num_classes, 1, 1, 0)
            self.dropout_deepsup = nn.Dropout2d(0.1)

        # Weight init
        self.apply(self.init_weight)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(
                nn.functional.interpolate(
                    pool_scale(conv5), (input_size[2], input_size[3]), mode="bilinear", align_corners=False
                )
            )
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)
        if segSize is not None:
            x = F.interpolate(
                x, size=segSize, mode="bilinear", align_corners=False)

        if self.deep_sup:
            outs = [x]
            for fe in conv_out[:-1]:
                x_ds = self.cbr_deepsup(fe)
                x_ds = self.dropout_deepsup(x_ds)
                x_ds = self.conv_last_deepsup(x_ds)
                if segSize is not None:
                    x_ds = F.interpolate(
                        x_ds, size=segSize,
                        mode="bilinear", align_corners=False)
                outs.append(x_ds)
            return outs

        return x
