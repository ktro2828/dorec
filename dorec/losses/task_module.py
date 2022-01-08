#!/usr/bin/env python


from box import Box
import torch.nn as nn

from dorec.core import LOSSES


class TaskAssignModule(nn.Module):
    """Loss function for multi task
    Args:
        cfg (Box)
    """

    def __init__(self, cfg):
        super(TaskAssignModule, self).__init__()
        assert isinstance(cfg, Box)
        self.criterions = {}
        self.task = []
        for tsk in cfg.keys():
            self.criterions[tsk] = LOSSES.build(cfg[tsk])
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
