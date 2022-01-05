#!/usr/bin/env python

from box import Box
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from .config import Config


def build_optimizer(cfg, params):
    """Returns torch.optim.Optimiezer

    Args:
        cfg (Config, Box)
        params (generator): return of model.parameters()

    Returns:
        optimizer (torch.optim.Optimizer)
    """
    if isinstance(cfg, Config):
        optimizer_cfg = cfg.optimizer
    elif isinstance(cfg, Box):
        optimizer_cfg = cfg.copy()
    else:
        raise TypeError(
            "``cfg`` must be a type of Config or Box, but got {}".format(type(cfg)))
    name = optimizer_cfg.pop("name")

    # Filtering by requires_grad = True
    params = filter(lambda p: p.requires_grad, params)

    if name == "Adam":
        optimizer = optim.Adam(params, **optimizer_cfg)
    elif name == "SGD":
        optimizer = optim.SGD(params, **optimizer_cfg)
    elif name == "RMSprop":
        optimizer = optim.RMSprop(params, **optimizer_cfg)
    elif name == "Adadelta":
        optimizer = optim.Adadelta(params, **optimizer_cfg)
    elif name == "AdamW":
        optimizer = optim.AdamW(params, **optimizer_cfg)
    else:
        raise ValueError("unsupported optimizer: {}".format(name))

    setattr(optimizer, "name", name)

    return optimizer


def build_scheduler(cfg, optimizer):
    """Returns scheduler

    Args:
        cfg (Config, Box)
        optimier (torch.optim.Optimizer)

    Returns:
        scheduler (torch.optim.lr_scheduler)
    """
    if isinstance(cfg, Config):
        scheduler_cfg = cfg.scheduler
    elif isinstance(cfg, Box):
        scheduler_cfg = cfg.copy()
    else:
        raise TypeError(
            "``cfg`` must be a type of Config or Box, but got {}".format(type(cfg)))
    name = scheduler_cfg.pop("name")

    if name == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(optimizer, **scheduler_cfg)
    elif name == "ExponentialLR":
        scheduler = lr_scheduler.ExponentialLR(optimizer, **scheduler_cfg)
    elif name == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_cfg)
    elif name == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, **scheduler_cfg)
    elif name == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_cfg)
    elif name == "CyclicLR":
        scheduler = lr_scheduler.CyclicLR(optimizer, **scheduler_cfg)
    elif name == "LambdaLR":
        scheduler = lr_scheduler.LambdaLR(optimizer, **scheduler_cfg)
    else:
        raise ValueError("unsupported scheduler: {}".format(name))

    setattr(scheduler, "name", name)

    return scheduler
