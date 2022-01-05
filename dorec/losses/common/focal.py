#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from dorec.core.ops.tensor import probalize
from dorec.core import LOSSES


@LOSSES.register()
class FocalLoss(nn.Module):
    """
    https://amaarora.github.io/2020/06/29/FocalLoss.html
    """

    def __init__(self, alpha=1.0, gamma=2, weight=None, size_average=False, use_softmax=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.use_softmax = use_softmax

    def forward(self, preds, targets, smooth=1e-5):
        preds = probalize(preds, use_softmax=self.use_softmax)

        # flatten inputs
        preds = preds.view(-1).float()
        targets = targets.view(-1).float()

        # compute cross-entropy
        bce = F.binary_cross_entropy(preds, targets)
        bce_exp = torch.exp(-bce)
        focal_loss = self.alpha * (1 - bce_exp) ** self.gamma * bce

        return focal_loss
