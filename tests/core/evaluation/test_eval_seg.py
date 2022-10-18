#!/usr/bin/env python

import torch

from dorec.core.evaluation import segmentation_eval


def test_segmentation_eval():
    B, C, H, W = 5, 3, 100, 100
    preds = torch.rand(B, C, H, W)
    targets = torch.randint(0, 2, (B, C, H, W))

    methods = ["pxacc", "miou", "mdice", "mfscore", "msq"]

    scores = segmentation_eval(preds, targets, methods=methods)
    assert isinstance(scores, dict)
    for key, score in scores.items():
        assert key in methods
        assert isinstance(score, float)
