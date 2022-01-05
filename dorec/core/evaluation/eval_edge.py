#!/usr/bin/env python

"""References:
papers:
- https://arxiv.org/abs/1801.00454
code:
- http://cs230.stanford.edu/projects_fall_2020/reports/55819628.pdf
- https://github.com/Walstruzz/edge_eval_python/blob/main/impl/edges_eval_dir.py
"""
import torch

from dorec.core.ops import probalize
from .utils import mean_fscore


def edge_eval(
    preds,
    targets,
    thresh=(99, 86, 63, 50),
    methods=("ods"),
    omitnans=False
):
    """
    Args:
        pred (torch.Tensor): in shape (B, 1, H, W)
        target (torch.Tensor): in shape (B, 1, H, W)
        thresh (sequence): list of threshold values(0~255)
        methods (tuple[str]): [ois, ods]
        omitnans (bool)
    Reutrns:
        scores (dict[str, float])
    """
    assert preds.shape == targets.shape
    if isinstance(methods, str):
        methods = (methods, )

    preds = probalize(preds)
    targets = torch.topk(targets, k=1, dim=1).indices.squeeze(1)
    scores = {}
    for mth in methods:
        if mth == "ois":
            score = _ois(preds, targets, thresh, omitnans)
        elif mth == "ods":
            score = _ods(preds, targets, thresh, omitnans)
        else:
            raise ValueError("unsupported method: {}".format(mth))

        scores.update({mth: score})

    return scores


def _apply_threshold(pd, thresh):
    """
    Args:
        pd (torch.Tensor): in shape (B, 1, H, W)
        thresh (float)
    Returns:
        out (torch.Tensor, dtype=long): in shape (B, H, W)
    """
    out = torch.zeros_like(pd).to(pd.device)
    out[pd < thresh] = 0.0
    out[pd >= thresh] = 1.0

    out = out.squeeze(1).long()

    return out


def _ois(pred, target, thresh, omitnans=False):
    """
    Args:
        pred (torch.Tensor): (B, 1, H, W)
        target (torch.Tensor): (B, H, W)
        thresh (tuple[float])
    Returns:
        ois (float)
    """
    fscores = []
    for pd, tgt in zip(pred, target):
        b_score = []
        for th in thresh:
            pd_ = _apply_threshold(pd, th / 255.0)
            b_score.append(
                mean_fscore(pd_, tgt, num_classes=2, omitnans=omitnans))
        fscores.append(b_score)

    fscores = torch.tensor(fscores)
    max_ois = fscores.max(dim=0).values
    ois = max_ois.mean().item()

    return ois


def _ods(pred, target, thresh, omitnans=False):
    """
    """
    fscores = []
    for th in thresh:
        pd_ = _apply_threshold(pred, th / 255.0)
        fscores.append(
            mean_fscore(pd_, target, num_classes=2, omitnans=omitnans))
    ods = max(fscores).item()

    return ods
