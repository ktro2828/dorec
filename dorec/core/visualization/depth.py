#!/usr/bin/env python

import cv2
import numpy as np
import torch
import torch.nn.functional as F

"""
NOTE:
https://stackoverflow.com/questions/13840013/opencv-how-to-visualize-a-depth-image
"""

USAGE = """
[*] USAGE:(TYPE: SHAPE)
- TYPE: numpy.ndarray, torch.Tensor
- SHAPE:
    - 3 dim: 1xHxW, HxWx1
    - 2 dim: HxW
"""


def imshow_depth(depth_map, thresh=0.5, colorize=False):
    """
    Args:
        depth_map (np.ndarray, torch.Tesor): in shape 1xHxW
        thresh (float, optional): segmentation score threshold(default: 0.5)
        colorize (bool): indicates whether apply colormap to depth(default: False)
    Reutrns:
        np.ndarray: HxWx3 or HxW
    """
    assert depth_map.ndim in (2, 3), \
        "shape of depth map must be 2 or 3 dimentioal and have 1 channel in case of 3 dimenion"

    if depth_map.ndim == 3:
        ignore_dim = depth_map.shape.index(1)
        depth_map = depth_map[ignore_dim]

    if isinstance(depth_map, torch.Tensor):
        depth_map = F.threshold(depth_map, threshold=0.0, value=0.0)
        depth_map = depth_map.cpu().detach().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth_map = np.clip(depth_map, 0.0, None)
    else:
        raise ValueError(
            "type or shape of depth map is unsupported \n" + USAGE)

    min_val, max_val, _, _ = cv2.minMaxLoc(depth_map)
    adj_map = cv2.convertScaleAbs(
        depth_map,
        cv2.CV_8UC1,
        255 / (max_val - min_val + 1e-8),
        -min_val)
    if colorize:
        adj_map = cv2.applyColorMap(adj_map, cv2.COLORMAP_AUTUMN)

    return adj_map
