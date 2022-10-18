#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from dorec.core import HEADS
from dorec.models.layers import BatchNorm2d, conv3x3_bn_relu

from .head_base import HeadBase


@HEADS.register()
class UPerNet(HeadBase):
    """UPerNet

    Args:
        num_classes (int, optional)
        fc_dim (int, optional)
        pool_scales (tuple[int], optional)
        fpn_inplaces (tuple[int], optional)
        fpn_dim (bool, optional)
    """

    def __init__(
        self,
        num_classes,
        fc_dim,
        pool_scales=(1, 2, 3, 6),
        fpn_inplanes=(256, 512, 1024, 2048),
        fpn_dim=256,
        deep_sup=False
    ):
        super(UPerNet, self).__init__(deep_sup)

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(
                nn.Sequential(
                    nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                    BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                )
            )
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(
            fc_dim + len(pool_scales) * 512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:  # skip the top layer
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                    BatchNorm2d(fpn_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(
                nn.Sequential(
                    conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
                )
            )
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim,
                            1), nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        )

        # Weight init
        self.apply(self.init_weight)

    def forward(self, features, segSize=None):
        """
        Args:
            features (list[torch.Tensor])
            segSize (tuple, optional)
        Returns:
            if self.deep_sup == True:
                list[torch.Tensor]
            else:
                torch.Tensor
        """
        conv5 = features[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(
                pool_conv(
                    nn.functional.interpolate(
                        pool_scale(conv5),
                        (input_size[2], input_size[3]),
                        mode="bilinear", align_corners=False
                    )
                )
            )
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(features) - 1)):
            conv_x = features[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:],
                mode="bilinear", align_corners=False
            )  # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(
                F.interpolate(
                    fpn_feature_list[i],
                    output_size,
                    mode="bilinear",
                    align_corners=False)
            )
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        if segSize is not None:
            x = F.interpolate(
                x, size=segSize,
                mode="bilinear", align_corners=False)

        if self.deep_sup:
            outs = [x]
            for x_ds in self.fusion_list:
                if segSize is not None:
                    x_ds = F.interpolate(
                        x_ds,
                        size=segSize,
                        mode="bilinear",
                        align_corners=False)
                outs.append(x_ds)
            return outs
        return x
