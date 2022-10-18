#!/usr/bin/env python

from functools import partial

import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101

from dorec.core import BACKBONES

from ..base import ModuleBase


class ResNetasEncoder(ModuleBase):
    """ResNet dilated
    Args:
        orig_resnet (torch.nn.Module)
        dilate_scale (int, optional)
        max_layer (int, optional)
    """

    def __init__(self, orig_resnet, dilate_scale=None, max_layer=4):
        super(ResNetasEncoder, self).__init__()
        if max_layer < 1 or max_layer > 4:
            raise ValueError("``max_layer`` must be in [1, 4]")
        self.max_layer = max_layer

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        conv_out = []

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        for layer in (self.layer1, self.layer2, self.layer3, self.layer4)[:self.max_layer]:
            x = layer(x)
            conv_out.append(x)

        return conv_out


@BACKBONES.register()
class ResNet18(ResNetasEncoder):
    """ResNet18
    Args:
        in_channels (int)
        dilate_scale (int, optional)
        max_layer (int, optional)
        pretrain (bool, optional)
    """

    def __init__(self, in_channels, dilate_scale=None, max_layer=4, pretrain=True):
        orig_resnet = resnet18(pretrain)
        if in_channels != 3:
            orig_resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)
        super(ResNet18, self).__init__(orig_resnet, dilate_scale, max_layer)
        if not pretrain:
            self.apply(self.init_weight)


@BACKBONES.register()
class ResNet34(ResNetasEncoder):
    """ResNet34
    Args:
        in_channels (int)
        dilate_scale (int, optional)
        max_layer (int, optional)
        pretrain (bool, optional)
    """

    def __init__(self, in_channels, dilate_scale=None, max_layer=4, pretrain=True):
        orig_resnet = resnet34(pretrain)
        if in_channels != 3:
            orig_resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)
        super(ResNet34, self).__init__(orig_resnet, dilate_scale, max_layer)
        if not pretrain:
            self.apply(self.init_weight)


@BACKBONES.register()
class ResNet50(ResNetasEncoder):
    """ResNet50
    Args:
        in_channels (int)
        dilate_scale (int, optional)
        max_layer (int, optional)
        pretrain (bool, optional)
    """

    def __init__(self, in_channels, dilate_scale=None, max_layer=4, pretrain=True):
        orig_resnet = resnet50(pretrain)
        if in_channels != 3:
            orig_resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)
        super(ResNet50, self).__init__(orig_resnet, dilate_scale, max_layer)
        if not pretrain:
            self.apply(self.init_weight)


@BACKBONES.register()
class ResNet101(ResNetasEncoder):
    """ResNet101
    Args:
        in_channels (int)
        dilate_scale (int, optional)
        max_layer (int, optional)
        pretrain (bool, optional)
    """

    def __init__(self, in_channels=3, dilate_scale=None, max_layer=4, pretrain=True):
        orig_resnet = resnet101(pretrain)
        if in_channels != 3:
            orig_resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)
        super(ResNet101, self).__init__(orig_resnet, dilate_scale, max_layer)
        if not pretrain:
            self.apply(self.init_weight)
