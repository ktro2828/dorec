#!/usr/bin/env python

from .eval_keypoint import keypoint_eval
from .eval_seg import segmentation_eval
from .eval_edge import edge_eval
from .eval_depth import depth_eval

from dorec.core import Config

__all__ = ("load_metrics", "keypoint_eval",
           "segmentation_eval", "edge_eval", "depth_eval")


EVALUATION_TYPES = ("edge", "depth", "keypoint", "segmentation")


def load_metrics(cfg, preds, targets):
    """Load evaluation function for specified task
    Args:
        cfg (dict, dorec.core.Config)
        preds (dict[str, torch.Tensor])
        targets (dict[str, torch.Tensor])
    Returns:
        scores (dict[str, float])
    """
    if isinstance(cfg, Config):
        cfg_args = cfg.evaluation
    elif isinstance(cfg, dict):
        cfg_args = cfg.copy()

    scores = {}
    for task, eval_cfg in cfg_args.items():
        pd = preds[task]
        gt = targets[task]
        if task == "edge":
            score = edge_eval(pd, gt, **eval_cfg)
        elif task == "depth":
            score = depth_eval(pd, gt, **eval_cfg)
        elif task == "keypoint":
            score = keypoint_eval(pd, gt, **eval_cfg)
        elif task == "segmentation":
            score = segmentation_eval(pd, gt, **eval_cfg)
        else:
            raise ValueError("unexpected name: {}".format(task))

        scores[task] = score

    return scores
