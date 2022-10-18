#!/usr/bin/env python

import numpy as np
import torch

from tasks.visualization import imshow_segmentation


def test_imshow_segmentation():
    N, H, W = 3, 100, 100
    img = np.random.randint(0, 255, (H, W, 3)).astype(np.uint8)
    segmaps = torch.rand(N, H, W)

    img_ = imshow_segmentation(img, segmaps, img_ord="hwc")
    assert isinstance(img_, np.ndarray)
