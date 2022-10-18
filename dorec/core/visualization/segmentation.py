#!/usr/bin/env python

import cv2
import imgviz
import numpy as np
import torch

from dorec.core.ops import imread, COLORMAP


def imshow_segmentation(
    img,
    segmaps,
    img_ord="chw",
    target_idx=None,
    alpha=0.3,
    ignore_idx=-1
):
    """Draw segmentaion mask

    Args:
        img (str, np.ndarray, torch.Tensor): image path or tensor, in shape CxHxW or HxWxC
        segmaps (np.ndarray, torch.Tensor): predicted or GT segmentaion maps, in shape NxHxW
        img_ord (str, optional): order of image shape(default: 'chw')
        target_idx (int, optional): if specified, only visualize output specified index channels(default: None)
        alpha (float, optional): alpha weight value(default: 0.2)
        ignore_idx (int, optional): index to ignore

    Returns:
        img (np.ndarray)
    """
    img = imread(img, img_ord)

    if segmaps.ndim != 3:
        raise ValueError(
            "expected `segmaps` is in shape NxHxW, but got {} dimensions".format(segmaps.ndim))

    if isinstance(segmaps, torch.Tensor):
        segmaps = segmaps.cpu().detach().numpy()

    N, H, W = segmaps.shape
    mask = np.zeros_like(img)
    mask_idx = np.argmax(segmaps, axis=0)

    if ignore_idx == -1:
        ignore_idx = N - 1

    if target_idx is not None:
        assert target_idx <= N - \
            1, "{} is out of range {}".format(target_idx, N - 1)
        mask[mask_idx == target_idx] = COLORMAP[target_idx % 8]
    else:
        for n in range(N):
            if n == ignore_idx:
                continue
            mask[mask_idx == n] = COLORMAP[n % 8]

    img = cv2.addWeighted(img, 1, mask, alpha, 0)

    return img


# DEBUG
def _imshow_segmentation(
    img,
    segmaps,
    img_ord="chw",
):
    """Draw segmentaion mask

    Args:
        img (str, np.ndarray, torch.Tensor): image path or tensor, in shape CxHxW or HxWxC
        segmaps (np.ndarray, torch.Tensor): predicted or GT segmentaion maps, in shape NxHxW
        img_ord (str, optional): order of image shape(default: 'chw')

    Returns:
        img (np.ndarray)
    """
    img = imread(img, img_ord)

    label = torch.topk(segmaps, k=1, dim=0).indices.squeeze(0)
    label = label.cpu().detach().numpy()
    label = label + 1
    label[label == 4] = 0

    viz = imgviz.label2rgb(
        label,
        imgviz.rgb2gray(img.astype(np.uint8)),
        font_size=15,
        label_names=("background", "top", "low", "no"),
        loc="rb"
    )

    viz = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)

    return viz
