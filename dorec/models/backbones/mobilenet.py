from functools import partial

import torch.nn as nn
from torchvision.models import mobilenet_v2

from dorec.core import BACKBONES

from ..base import ModuleBase


class MobileNetV2asEncoder(ModuleBase):
    """MobileNetv2 dilated
    Args:
        orig_net (torch.nn.Module)
        dilate_scale (int, optional)
    """

    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2asEncoder, self).__init__()

        # take pretrained mobilenet features
        self.features = orig_net.features

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2))

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
        for i in range(self.total_idx):
            x = self.features[i](x)
            if i in self.down_idx:
                conv_out.append(x)
        conv_out.append(x)
        return conv_out


@BACKBONES.register()
class MobileNetV2(MobileNetV2asEncoder):
    def __init__(self, in_channels=3, dilate_scale=None, pretrain=True):
        orig_mobilenet = mobilenet_v2(pretrain)
        if in_channels != 3:
            orig_mobilenet.features[0][0] = nn.Conv2d(
                in_channels, 32, kernel_size=(3, 3),
                stride=(2, 2), padding=(1, 1), bias=False
            )
        super(MobileNetV2, self).__init__(orig_mobilenet, dilate_scale)
        if not pretrain:
            self.apply(self.init_weight)
