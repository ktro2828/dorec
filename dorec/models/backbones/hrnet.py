"""
This HRNet implementation is modified from the following repository:
https://github.com/HRNet/HRNet-Semantic-Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dorec.core import BACKBONES
from dorec.core.utils import load_from_url

from ..base import ModuleBase
from ..blocks import BasicBlock, Bottleneck, HighResolutionModule
from ..layers import BatchNorm2d


BLOCKS_DICT = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}
URL = "http://sceneparsing.csail.mit.edu/model/pretrained_resnet/hrnetv2_w48-imagenet.pth"


class _HRNetV2(ModuleBase):
    BN_MOMENTUM = 0.1

    def __init__(self, in_channels, deep_sup=False, **kwargs):
        super(_HRNetV2, self).__init__()
        extra = {
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": (4, 4),
                "NUM_CHANNELS": (48, 96),
                "FUSE_METHOD": "SUM",
            },
            "STAGE3": {
                "NUM_MODULES": 4,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": (4, 4, 4),
                "NUM_CHANNELS": (48, 96, 192),
                "FUSE_METHOD": "SUM",
            },
            "STAGE4": {
                "NUM_MODULES": 3,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": (4, 4, 4, 4),
                "NUM_CHANNELS": (48, 96, 192, 384),
                "FUSE_METHOD": "SUM",
            },
            "FINAL_CONV_KERNEL": 1,
        }

        # stem net
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=self.BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, momentum=self.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        self.stage2_cfg = extra["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = BLOCKS_DICT[self.stage2_cfg["BLOCK"]]
        num_channels = [num_channels[i] *
                        block.EXPANSION for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = BLOCKS_DICT[self.stage3_cfg["BLOCK"]]
        num_channels = [num_channels[i] *
                        block.EXPANSION for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = BLOCKS_DICT[self.stage4_cfg["BLOCK"]]
        num_channels = [num_channels[i] *
                        block.EXPANSION for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        self.deep_sup = deep_sup

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                            BatchNorm2d(
                                num_channels_cur_layer[i], momentum=self.BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - \
                        num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels,
                                      3, 2, 1, bias=False),
                            BatchNorm2d(
                                outchannels, momentum=self.BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.EXPANSION:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.EXPANSION,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.EXPANSION,
                            momentum=self.BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.EXPANSION
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = BLOCKS_DICT[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(
                    num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): in shape (B, C, H, W)
        Returns:
            outs (list[torch.Tensor])
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                           mode="bilinear", align_corners=False)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                           mode="bilinear", align_corners=False)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w),
                           mode="bilinear", align_corners=False)

        x_ = torch.cat([x[0], x1, x2, x3], dim=1)

        if self.deep_sup:
            return [x3, x2, x1, x[0]]
        return [x_]


@BACKBONES.register()
class HRNetV2(_HRNetV2):
    def __init__(self, in_channels, pretrain=True, deep_sup=False):
        if pretrain:
            in_channels_ = 3
        else:
            in_channels_ = in_channels
        super(HRNetV2, self).__init__(in_channels_, deep_sup)
        if pretrain:
            self.load_state_dict(load_from_url(URL), strict=False)
            if in_channels != 3:
                self.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=3,
                    stride=2, padding=1, bias=False)
        else:
            self.apply(self.init_weight)
