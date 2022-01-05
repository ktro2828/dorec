#!/usr/bin/env python

import torch.nn as nn

from dorec.core import LOSSES
from dorec.core.ops import probalize


@LOSSES.register()
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=False, **kwargs):
        super(IoULoss, self).__init__()
        self.weight = weight
        self.size_average = size_average

    def forward(self, preds, targets, smooth=1e-5):
        preds = probalize(preds)

        # flatten inputs
        preds = preds.view(-1).float()
        targets = targets.view(-1).float()

        inner = (preds * targets).sum()
        total = (preds + targets).sum()
        union = total - inner

        iou = (inner + smooth) / (union + smooth)

        return 1 - iou
