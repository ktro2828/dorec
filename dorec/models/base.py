#!/usr/bin/env python

import torch
import torch.nn as nn


class ModuleBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ModuleBase, self).__init__()
        self._name = None

    @property
    def name(self):
        if self._name is None:
            self._name = self.__class__.__name__
        return self._name

    def init_weight(self, m):
        """Initialize weights
        Args:
            m (torch.nn.Module)
        """
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(1e-4)
        elif classname.find("Linear") != -1:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(1e-4)

    def __repr__(self):
        return self.name


class TaskModule(ModuleBase):
    def __init__(self, task, model):
        super(TaskModule, self).__init__()
        if not isinstance(task, tuple):
            raise TypeError(
                "type of ``task`` must be a tuple, but got {}".format(task))
        self.task = task
        self.model = model

        if hasattr(model, "deep_sup"):
            self.deep_sup = model.deep_sup
        else:
            self.deep_sup = False

        self._name = model.name

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): in shape (B, C, H, W)
        Returns:
            outputs (dict[str, torch.Tensor])
        """
        preds = self.model(inputs)

        if isinstance(preds, torch.Tensor) or self.deep_sup:
            preds = [preds]

        if len(preds) != len(self.task):
            raise ValueError(
                "the numbet of model output must be same with the number of task")

        outputs = {}
        for tsk, pd in zip(self.task, preds):
            outputs[tsk] = pd

        return outputs
