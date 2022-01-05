#!/usr/bin/env python

import torch.nn as nn

from dorec.core import LOSSES


@LOSSES.register()
class MultiTaskLoss(nn.Module):
    """Loss function for multi task
    Args:
        cfg (dict)
    """

    def __init__(self, **kwargs):
        super(MultiTaskLoss, self).__init__()
        self.criterions = {}
        self.task = []
        for tsk in kwargs.keys():
            self.criterions[tsk] = LOSSES.build(kwargs[tsk])
            self.task.append(tsk)

    def forward(self, outputs, targets):
        out = {}
        total = 0
        for tsk in self.task:
            cost = self.criterions[tsk](outputs[tsk], targets[tsk])
            total += cost
            out[tsk] = cost

        out["total"] = total

        return out
