#!/usr/bin/env python

import numpy as np
import torch

from tasks.visualization import imshow_keypoint


def test_imshow_keypoint():
    K, H, W = 8, 100, 100
    img = np.random.randint(0, 256, (H, W, 3)).astype(np.uint8)
    heatmaps = torch.rand(K, H, W)
    kpts = np.random.randint(0, H, (K, 2))

    img1 = imshow_keypoint(img, heatmaps=heatmaps, img_ord="hwc")
    img2 = imshow_keypoint(img, kpts=kpts, img_ord="hwc")
    assert isinstance(img1, np.ndarray)
    assert isinstance(img2, np.ndarray)
