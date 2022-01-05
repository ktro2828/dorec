#!/usr/bin/env python

import cv2
import numpy as np
import torch

from dorec.core.ops import imread, COLORMAP, probalize


def imshow_edge(
    edge_map,
    img=None,
    img_ord="chw",
    thresh=0.5,
    alpha=0.2
):
    """
    Args:
        edge_map (np.ndarray, torch.Tesor): in shape 1xHxW
        img (str, np.ndaray, torch.Tensor): image path or tensor, in shape CxHxW or HxWxC
        img_ord (str, optional): order of image shape(default: 'chw')
        thresh (float, optional): segmentation score threshold(default: 0.5)
        alpha (float, optional): alpha weight value(default: 0.2)
    Reutrns:
        np.ndarray: HxWx3 or HxW
    """
    assert len(edge_map.shape) == 3, "expected `edge_map` is in shape 1xHxW"
    if isinstance(edge_map, torch.Tensor):
        edge_map = probalize(edge_map)
        edge_map = edge_map.squeeze(0).cpu().detach().numpy()
    elif isinstance(edge_map, np.ndarray):
        edge_map = _normalize(edge_map)

    if img is not None:
        img = imread(img, img_ord)
        mask = np.zeros_like(img)
        mask = np.expand_dims(edge_map[0], axis=-1) * COLORMAP[0]
        mask = mask.astype(np.uint8)
        img = cv2.addWeighted(img, 1, mask, alpha, 0, dtype=cv2.CV_32F)
        return img
    else:
        return (255 * edge_map).astype(np.uint8)


def _normalize(edge_map):
    """
    Args:
        edge_map (np.ndarray): in shape 1xHxW
    Returns:
        norm_edge_map (np.ndarray)
    """
    max_val = edge_map.max()
    min_val = edge_map.min()

    norm_edge_map = edge_map / (max_val - min_val + 1e-8)

    return norm_edge_map
