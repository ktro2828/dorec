#!/usr/bin/env python

from box import Box
from dorec.core import Config

from .task_module import TaskAssignModule


def build_loss(cfg):
    """Build loss function
    Args:
        cfg (Config, Box)
    Returns:
        loss (torch.nn.Module)
    """
    if isinstance(cfg, Config):
        loss_cfg = cfg.loss
    elif isinstance(cfg, Box):
        loss_cfg = cfg.copy()
    else:
        raise TypeError(
            "``cfg`` must be a type of Config or Box, but got {}".format(type(cfg)))

    loss = TaskAssignModule(loss_cfg)

    return loss
