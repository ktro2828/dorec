#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from dorec.core import LOSSES


@LOSSES.register()
class RMSELoss(nn.Module):
    """
    Referece:
    - https://github.com/pierlj/ken-burns-effect/blob/cae3db9decdf3a20de319e662261468796bc047b/utils/losses.py#L20
    """

    def __init__(self, **kwargs):
        super(RMSELoss, self).__init__()

    def forward(self, preds, targets):
        preds = F.threshold(preds, threshold=0.0, value=0.0)
        mask = torch.zeros_like(targets)
        mask[targets != 0] = 1.0

        r_i = (preds - targets) * mask
        num_pos = torch.sum(mask).item()

        if num_pos != 0:
            rmse = 1 / num_pos * (torch.sum(r_i ** 2)) - \
                (1 / num_pos * torch.sum(r_i)) ** 2
        else:
            rmse = torch.zeros(1).to(targets.device)
        return rmse


class LogRMSELoss(nn.Module):
    """
    Reference:
    - https://github.com/pierlj/ken-burns-effect/blob/cae3db9decdf3a20de319e662261468796bc047b/utils/losses.py#L28
    """

    def __init__(self, smooth=1e-7, **kwargs):
        super(LogRMSELoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = F.threshold(preds, threshold=0.0, value=0.0)
        mask = torch.zeros_like(targets)
        mask[targets != 0] = 1.0

        r_i = torch.log10(preds * mask + self.smooth) - \
            torch.log10(targets * mask + self.smooth)
        num_pos = torch.sum(mask).item()

        if num_pos != 0:
            logrmse = 1 / num_pos * \
                (torch.sum(r_i ** 2)) - (0.5 / num_pos * torch.sum(r_i)) ** 2
        else:
            logrmse = torch.zeros(1).to(targets.device)
        return logrmse
