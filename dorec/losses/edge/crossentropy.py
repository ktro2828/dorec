#!/usr/bin/env/python

import torch
import torch.nn as nn
import torch.nn.functional as F

from dorec.core import LOSSES
from dorec.core.ops import probalize


@LOSSES.register()
class RCFCrossEntropy(nn.Module):
    """Cross-Entropy for edge detection"""

    def __init__(
            self,
            ignore_label=-1,
            weight=None,
            align_corners=True,
            balance_weights=[1],
            use_softmax=True,
            reduction="none"
    ):
        super(RCFCrossEntropy, self).__init__()
        self.align_corners = align_corners
        self.balance_weights = balance_weights
        self.weight = weight
        self.ignore_label = ignore_label
        self.use_softmax = use_softmax
        self.reduction = reduction

    def forward(self, preds, targets):
        preds = probalize(preds, use_softmax=self.use_softmax)

        targets = targets.long()
        mask = targets.float()
        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()

        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
        mask[mask == 2] = 0

        cost = F.binary_cross_entropy(
            preds.float(),
            targets.float(),
            weight=mask,
            reduction=self.reduction
        )

        return torch.sum(cost) / (num_negative + num_positive)
