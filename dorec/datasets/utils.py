#!/usr/bin/env python

import cv2
import numpy as np


def load_gt_img(filepath, img_type, **kwargs):
    if img_type == "mask":
        num_classes = kwargs["num_classes"]
        return load_gt_mask(filepath, num_classes)
    elif img_type == "edge":
        return load_gt_edge(filepath)
    elif img_type == "depth":
        return load_gt_depth(filepath)
    else:
        raise ValueError("unexpected GT type: {}".format(img_type))


def load_gt_mask(filepath, num_classes, ksize=15):
    """Load segmentation mask GT
    Args:
        filenpath (str)
        num_classes (int)
        ksize (int, optional)
    Returns:
        gt_mask (np.ndarray)
    """
    mask = cv2.imread(filepath, cv2.IMREAD_COLOR)
    mask = cv2.medianBlur(mask, ksize)

    # mask0: top turn, mask1: lower turn, mask2: no turn, mask3: background
    H, W = mask.shape[:2]
    mask0 = np.zeros((H, W), dtype=np.float32)
    mask1 = np.zeros((H, W), dtype=np.float32)
    mask2 = np.zeros((H, W), dtype=np.float32)
    mask3 = np.zeros((H, W), dtype=np.float32)

    # mask0: red, mask1: green, mask2: blur, mask3: black
    mask0[mask[:, :, 2] > 180] = 1.0
    mask1[mask[:, :, 1] > 180] = 1.0
    mask2[mask[:, :, 0] > 180] = 1.0
    mask3[(mask0 == 0.0) * (mask1 == 0.0) * (mask2 == 0.0)] = 1.0

    if num_classes == 4:
        gt_mask = np.stack((mask0, mask1, mask2, mask3), axis=-1)
    elif num_classes == 3:
        mask12 = np.clip(mask1 + mask2, 0, 1)
        gt_mask = np.stack((mask0, mask12, mask3), axis=-1)
    else:
        raise ValueError(
            "expected number of classes is 3 or 4, but got {}".format(num_classes))

    return gt_mask


def load_gt_edge(filepath):
    """Load edge GT
    Args:
        filepath (str)
    Returns:
        gt_edge (np.ndarray)
    """
    gt_edge = cv2.imread(filepath, 0)
    gt_edge[gt_edge < 128] = 0.0
    gt_edge[gt_edge >= 128] = 1.0
    gt_edge = gt_edge.astype(np.float32)
    return gt_edge


def load_gt_depth(filepath):
    """Load depth GT
    Args:
        filepath (str)
    Returns:
        gt_depth (np.ndarray)
    """
    return cv2.imread(filepath, 0)
