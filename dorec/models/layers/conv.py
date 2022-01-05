#!/usr/bin/env python

import torch.nn as nn

from .batchnorm import BatchNorm2d


def conv3x3_nlayers(in_ch, planes, out_ch, num_layers=3, stride=1):
    """Returns num_layers * (3x3 convolution + BN + relu) + 1x1 convolution
    Args:
        in_ch (int): number of input channel
        planes (int): number of hidden channel
        out_ch (int): number of output channel
        num_layers (int): number of hidden layers (default: 3)
        stride (int, optional): stride size (default: 1)
    """
    modules = []
    last_ch = in_ch
    for _ in range(num_layers - 1):
        modules.extend(
            [nn.Conv2d(last_ch, planes, kernel_size=3,
                       stride=stride, padding=1, bias=False),
             BatchNorm2d(planes),
             nn.ReLU(inplace=True)]
        )
        last_ch = planes

    modules.append(nn.Conv2d(last_ch, out_ch, 1,
                             stride=1, padding=0, bias=False))

    return nn.Sequential(*modules)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  stride=stride, padding=1, bias=False),
        BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), BatchNorm2d(oup), nn.ReLU6(inplace=True))
