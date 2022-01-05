#!/usr/bin/env python

import os.path as osp
import cv2

from .keypoint import imshow_keypoint
from .segmentation import imshow_segmentation
from .edge import imshow_edge
from .depth import imshow_depth

__all__ = ("imshow_keypoint", "imshow_segmentation",
           "imshow_edge", "imshow_depth",
           "saveviz", "LIMBS_PAIR")

VISUALIZAION_TYPES = ("keypoint", "segmetation", "edge", "depth")

LIMBS_PAIR = {
    "halfshirt": ((0, 2), (0, 4), (4, 6), (1, 3), (1, 5), (5, 7)),
    "shirts": ((0, 2), (0, 4), (4, 6), (1, 3), (1, 5), (5, 7)),
    "towel": ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)),
    "shorts": ((0, 2), (2, 4), (1, 3), (3, 5))
}


def saveviz(imgs, maps, tasks, filepath):
    """Load visualization function for specified task
    Args:
        task (str): task name
    Returns:
        ``function``
    """
    for task in tasks:
        outmaps = maps[task]
        if task == "keypoint":
            ret_img = imshow_keypoint(imgs, outmaps)
        elif task == "segmentation":
            ret_img = imshow_segmentation(imgs, outmaps)
        elif task == "edge":
            ret_img = imshow_edge(outmaps)
        elif task == "depth":
            ret_img = imshow_depth(outmaps)
        else:
            raise ValueError("unsupported task: {}".format(task))

        # Add "task name" to save path
        save_dir = osp.dirname(filepath)
        filename, ext = osp.splitext(osp.basename(filepath))
        filename = task + filename + ext
        save_path = osp.join(save_dir, filename)
        cv2.imwrite(save_path, ret_img)
