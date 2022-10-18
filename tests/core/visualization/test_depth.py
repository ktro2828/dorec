#!/usr/bin/env python

import numpy as np
import torch
import pytest

from tasks.visualization import imshow_depth


@pytest.mark.parametrize(("colorize"), [(True), (False)])
def test_imshow_depth(colorize):
    C, H, W = 1, 100, 100
    depth_map1 = torch.rand(C, H, W)
    depth_map2 = np.random.rand(C, H, W)

    out1 = imshow_depth(depth_map1, colorize=colorize)
    out2 = imshow_depth(depth_map2, colorize=colorize)
    assert isinstance(out1, np.ndarray)
    assert isinstance(out2, np.ndarray)
