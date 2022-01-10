#!/usr/bin/env python

import numpy as np
import torch

from dorec.core.evaluation import keypoint_eval


def test_eval_keypoint():
    B, C, H, W = 5, 8, 100, 100
    preds = torch.rand(B, C, H, W)
    targets = torch.randint(0, 2, (B, C, H, W))
    mask = torch.randint(0, 2, (B, C))

    methods = ["nme", "pck", "oks"]
    scores = keypoint_eval(preds, targets, mask, methods=methods)
    assert isinstance(scores, dict)
    for key, score in scores.keys():
        assert key in methods
        assert isinstance(score["all"], float)
        assert isinstance(score["each"], np.ndarray)
        assert score["each"].shape == (C, )
