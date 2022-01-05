#!/usr/bin/env python

from box import Box

from dorec.core import MODELS, BACKBONES, HEADS, Config

from .base import TaskModule, ModuleBase
from .modules import EDModule
from .heads import HeadBase

USAGE = r"""
[USAGE]Assuming config following,

- Building from ``name``
model:
    name: XXX
    ``kwargs``

- Building from ``backbones`` and ``heads``
model:
    backbones:
        name: XXX
        in_channels: <int>
        pretrain: <bool>
        ``kwargs``
    heads:
        name: YYY
        deep_sup: <bool>
        ``kwargs``
"""


def build_model(cfg, task=None, assign_task=True):
    """If assign_task=True, model is wrapped by ``TaskModule()`` and \
        output a dict(TASK_NAME: torch.Tensor)
    Args:
        cfg (Config, Box)
        task (tuple[str], optional)
        assign_task (bool, optional)
    Returns:
        model (ModuleBase)
    """
    if isinstance(cfg, Config):
        model_cfg = cfg.model
        task = cfg.task
    elif isinstance(cfg, Box):
        model_cfg = cfg.copy()
    else:
        raise TypeError(
            "``cfg`` must be a type of Config or Box, but got {}".format(type(cfg)))

    if set(model_cfg.keys()) >= set(("backbones", "heads")):
        # if set(model_cfg.keys()) >= set(("backbones", "heads")) and ("name" not in model_cfg.keys()):
        backbone = build_backbone(model_cfg.backbones)
        head = build_head(model_cfg.heads)
        model = EDModule(backbone, head)
    elif "name" in model_cfg.keys():
        model = MODELS.build(model_cfg)
    else:
        raise KeyError("unexpected keys: {}\n{}".format(cfg.keys(), USAGE))

    if assign_task:
        if task is None:
            raise NotImplementedError("``task`` must be specified")
        # Assign tasks to model outputs
        model = TaskModule(task, model)

    return model


def build_backbone(cfg):
    """Build BACKBONES
    Args:
        cfg (Box[str, any])
    Returns:
        backbone (ModuleBase)
    """
    if not isinstance(cfg, Box):
        raise TypeError(
            "``cfg`` must be a type of Box, but got {}".format(type(cfg)))

    backbone = BACKBONES.build(cfg)

    if not isinstance(backbone, ModuleBase):
        raise TypeError(
            "backbone must be a type of ModuleBase, but got {}".format(type(backbone)))

    return backbone


def build_head(cfg):
    """Build HEADS
    Args:
        cfg (OrdedDict[str, any])
    Returns:
        head (ModuleBase)
    """
    if not isinstance(cfg, Box):
        raise TypeError(
            "``cfg`` must be a type of box.Box, but got {}".format(type(cfg)))

    head = HEADS.build(cfg)

    if not isinstance(head, HeadBase):
        raise TypeError(
            "head must be a type of HeadBase, but got {}".format(type(head)))

    return head
