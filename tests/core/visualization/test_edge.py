#!/usr/bin/env python

import numpy as np
import torch

from tasks.visualization import imshow_edge


def test_imshow_edge():
    C, H, W = 1, 100, 100
    img = np.random.randint(0, 256, (H, W, 3)).astype(np.uint8)
    edge_map1 = torch.rand(C, H, W)
    edge_map2 = np.random.rand(C, H, W)

    out1 = imshow_edge(edge_map1)
    out2 = imshow_edge(edge_map2, img=img, img_ord="hwc")
    assert isinstance(out1, np.ndarray)
    assert isinstance(out2, np.ndarray)
