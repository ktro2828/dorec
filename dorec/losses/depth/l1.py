#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from dorec.core import LOSSES


@LOSSES.register()
class L1Loss(nn.Module):
    """
    Reference:
    - https://github.com/pierlj/ken-burns-effect/blob/cae3db9decdf3a20de319e662261468796bc047b/utils/losses.py#L10
    """

    def __init__(self, reduction="sum", **kwargs):
        super(L1Loss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction=reduction)

    def forward(self, preds, targets):
        preds = F.threshold(preds, threshold=0.0, value=0.0)
        mask = torch.zeros_like(targets)
        mask[targets != 0] = 1.0

        num_pos = torch.sum(mask).item()

        if num_pos != 0:
            loss = self.l1_loss(preds * mask, targets * mask) / num_pos
        else:
            loss = torch.zeros(1).to(targets.device)
        return loss
