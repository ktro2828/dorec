#!/usr/bin/env python

import torch.nn as nn
import torch.nn.functional as F

from dorec.core import LOSSES
from dorec.core.ops import probalize


@LOSSES.register()
class DiceLoss(nn.Module):
    """
    Dice = 2 |A and B| / (|A| + |B|)
    |A and B| = sum(A * B)
    """

    def __init__(self, weight=None, size_average=False, use_softmax=True):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.use_softmax = use_softmax

    def forward(self, preds, targets):
        preds = probalize(preds, use_softmax=self.use_softmax)

        # flatten inputs
        preds = preds.view(-1)
        targets = targets.view(-1)

        inner = (preds * targets).sum()
        dice = (2.0 * inner + 1e-5) / (preds.sum() + targets.sum() + 1e-5)

        return 1 - dice


@LOSSES.register()
class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=False, use_softmax=True):
        super(SoftDiceLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.use_softmax = use_softmax

    def forward(self, preds, targets):
        preds = probalize(preds, use_softmax=self.use_softmax)

        # flatten inputs
        preds = preds.view(-1)
        targets = targets.view(-1)

        inner = (preds * targets).sum()
        soft_dice = (2 * inner + 1e-5) / \
            ((preds ** 2).sum() + (targets ** 2).sum() + 1e-5)

        return 1 - soft_dice


@LOSSES.register()
class DiceBCELoss(nn.Module):
    """Dice CrossEntropy Loss"""

    def __init__(self, weight=None, size_average=False, use_softmax=True):
        super(DiceBCELoss, self).__init__()
        self.wieght = weight
        self.size_average = size_average
        self.use_softmax = use_softmax

    def forward(self, preds, targets):
        preds = probalize(preds, use_softmax=self.use_softmax)

        # flatten inputs
        preds = preds.view(-1)
        targets = targets.view(-1)

        inner = (preds * targets).sum()
        dice_loss = 1 - (2.0 * inner + 1e-5) / \
            (preds.sum() + targets.sum() + 1e-5)
        bce = F.binary_cross_entropy(preds, targets)
        dice_bce = bce + dice_loss

        return dice_bce
