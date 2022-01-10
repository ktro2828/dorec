#!/usr/bin/env python

import torch

from dorec.core.evaluation import depth_eval


def test_depth_eval(method):
    B, C, H, W = 5, 1, 100, 100
    preds = torch.rand(B, C, H, W)
    targets = torch.randint(0, 2, (B, C, H, W))

    methods = ["absrel", "sqrel", "rmse", "logrmse"]

    scores = depth_eval(preds, targets, methods=methods)
    assert isinstance(scores, dict)
    for key, score in scores.items():
        assert key in methods
        assert isinstance(score, float)
