#!/usr/bin/env python

import random

import numpy as np

from dorec.core.ops.keypoint import (
    normalize_keypoint, unnormalize_keypoint, get_max_pred, load_keypoint)


def test_load_keypoint():
    names = []
    for i in range(9):
        names.append(str(i))
    data = {}
    for nm in names:
        data[nm]["pos"] = [random.random(), random.random()]
        data[nm]["vis"] = random.randint(0, 2)
        data[nm]["in_pic"] = random.randint(0, 2)

    pos, vis, in_pic = load_keypoint(data, names)
    assert isinstance(pos, np.ndarray)
    assert pos.shape == (8, 2)
    assert isinstance(vis, np.ndarray)
    assert vis.shape == (8,)
    assert isinstance(in_pic, np.ndarray)
    assert in_pic.shape == (8,)


def test_normalize_keypoint():
    H, W, K = 100, 100, 8
    x = np.random.randint(0, W - 1, K)
    y = np.random.randint(0, H - 1, K)
    keypoint_pos = np.stack((x, y), -1)

    n = normalize_keypoint(W, H, keypoint_pos)
    assert ((n >= 0) & (n <= 1)).all()


def test_unnormalize_keypoints():
    H, W, K = 100, 100, 8
    x = np.random.randint(0, 1, K)
    y = np.random.randint(0, 1, K)
    keypoint_pos = np.stack((x, y), -1)

    n = unnormalize_keypoint(W, H, keypoint_pos)
    assert (n >= 0).all()
    assert ((n[:, 0] <= W) & (n[:, 1] <= H)).all()


def test_get_max_preds():
    B, K, H, W = 10, 8, 100, 100
    heatmaps = np.random.rand(B, K, H, W)
    maxvals_ = np.max(heatmaps.reshape(-1, K, H * W), axis=-1, keepdims=True)
    preds, maxvals = get_max_pred(heatmaps, return_type="numpy")
    assert np.allclose(maxvals, maxvals_)
