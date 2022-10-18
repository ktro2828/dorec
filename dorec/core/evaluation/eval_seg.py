#!/usr/bin/env python

import torch

from dorec.core.ops import probalize
from .utils import accuracy, mean_iou, mean_dice, mean_fscore, mean_sq


def segmentation_eval(
    preds,
    targets,
    methods=("pxacc", "miou"),
    omitnans=False
):
    """Compute score for 2D segmentation

    Args:
        preds (torch.Tensor): predicted segmentation maps, in shape (B, N, H, W)
        targets (torch.Tensor): ground truth maps, in shape (B, N, H, W)
        methods (list[str]): evaluation method(default: mIoU)
        omitnans (bool)
    Returns:
        score (dict[str, float])
    """
    assert preds.shape == targets.shape
    if isinstance(methods, str):
        methods = (methods, )

    num_classes = preds.shape[1]
    preds = probalize(preds)

    # Convert values to tensor of top indices
    preds = torch.topk(preds, k=1, dim=1).indices.squeeze(1)
    targets = torch.topk(targets, k=1, dim=1).indices.squeeze(1)

    scores = {}
    for mth in methods:
        if mth == "pxacc":
            score = accuracy(preds, targets)
        elif mth == "miou":
            score = mean_iou(preds, targets, num_classes)
        elif mth == "mdice":
            score = mean_dice(preds, targets, num_classes)
        elif mth == "mfscore":
            score = mean_fscore(preds, targets, num_classes)
        elif mth == "msq":
            score = mean_sq(preds, targets, num_classes)
        else:
            raise ValueError("unsupported method: {}".format(mth))

        scores.update({mth: score})

    return scores
