#!/usr/bin/env python

import torch.nn as nn

from ..base import ModuleBase


class CAModule(ModuleBase):
    """Channel align module for different input channel(!=3) with input Conv2d
    Args:
        in_channel (int)
        model (nn.Module)
    """

    def __init__(self, model, in_channels=4):
        super(CAModule, self).__init__()
        assert isinstance(in_channels, int)
        assert isinstance(model, nn.Module), "{}".format(type(model))

        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)
        self.model = model

        self._name = model.name

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.model(x)

        return x
