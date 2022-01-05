#!/usr/bin/env python

import math

import torch.nn as nn
from dorec.core.utils.io import load_from_url

from dorec.core.utils.manager import BACKBONES
from dorec.models.blocks.bottleneck import GroupBottleneck

from .resnet import ResNetasEncoder
from ..layers import BatchNorm2d, conv3x3


URLS = {
    "resnext50": "http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnext50-imagenet.pth",
    "resnext101": "http: // sceneparsing.csail.mit.edu/model/pretrained_resnet/resnext101-imagenet.pth"
}


class ResNeXt(nn.Module):
    def __init__(self, in_channels, block, layers, groups=32, num_classes=1000):
        self.inplanes = 128
        super(ResNeXt, self).__init__()
        self.conv1 = conv3x3(in_channels, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], groups=groups)
        self.layer2 = self._make_layer(
            block, 256, layers[1], stride=2, groups=groups)
        self.layer3 = self._make_layer(
            block, 512, layers[2], stride=2, groups=groups)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], stride=2, groups=groups)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(1024 * block.EXPANSION, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * \
                    m.out_channels // m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.EXPANSION:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.EXPANSION,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.EXPANSION),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, groups, downsample))
        self.inplanes = planes * block.EXPANSION
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


@BACKBONES.register()
class ResNext50(ResNetasEncoder):
    """ResNext50
    Args:
        in_channels (int)
        dialte_scale (int, optional)
        max_layer (int, optional)
        pretrain (bool, optional)
    """

    def __init__(self, in_channels, dilate_scale=None, max_layer=4, pretrain=True):
        if pretrain:
            orig_resnext = ResNeXt(3, GroupBottleneck, [3, 4, 6, 3])
            orig_resnext.load_state_dict(
                load_from_url(URLS["resnext50"]), strict=False)
            if in_channels != 3:
                orig_resnext.conv1 = conv3x3(in_channels, 64, stride=2)
        else:
            orig_resnext = ResNeXt(
                in_channels, GroupBottleneck, [3, 4, 6, 3])
        super(ResNext50, self).__init__(orig_resnext, dilate_scale)
        if not pretrain:
            self.apply(self.init_weight)


@BACKBONES.register()
class ResNext101(ResNetasEncoder):
    """ResNext101
    Args:
        in_channels (int)
        dilate_scale (int, optional)
        max_layer (int, optional)
        pretrain (bool, optional)
    """

    def __init__(self, in_channels, dilate_scale=None, max_layer=4, pretrain=True):
        if pretrain:
            orig_resnext = ResNeXt(3, GroupBottleneck, [3, 4, 23, 3])
            orig_resnext.load_state_dict(
                load_from_url(URLS["resnext101"]), strict=False)
            if in_channels != 3:
                orig_resnext.conv1 = conv3x3(in_channels, 64, stride=2)
        else:
            orig_resnext = ResNeXt(in_channels, GroupBottleneck, [3, 4, 23, 3])
        super(ResNext101, self).__init__(orig_resnext, dilate_scale)
        if not pretrain:
            self._model.apply(self.init_weight)
