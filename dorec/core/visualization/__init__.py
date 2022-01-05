#!/usr/bin/env python

import os.path as osp
import cv2

from .keypoint import imshow_keypoint
from .segmentation import imshow_segmentation
from .edge import imshow_edge
from .depth import imshow_depth

__all__ = ("imshow_keypoint", "imshow_segmentation",
           "imshow_edge", "imshow_depth",
           "do_visualize", "LIMBS_PAIR")

VISUALIZAION_TYPES = ("keypoint", "segmetation", "edge", "depth")

LIMBS_PAIR = {
    "halfshirt": ((0, 2), (0, 4), (4, 6), (1, 3), (1, 5), (5, 7)),
    "shirts": ((0, 2), (0, 4), (4, 6), (1, 3), (1, 5), (5, 7)),
    "towel": ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)),
    "shorts": ((0, 2), (2, 4), (1, 3), (3, 5))
}


def do_visualize(maps, imgs=None, filepath=None):
    """Visualize results for specified task
    Args:
        maps (dict[str, torch.Tensor]): output maps
        imgs (torch.Tensor, np.ndarray, optional): needs for keypoint and segmentaion
        filepath (str, optional): filepath to save result image
    Returns:
        resutls (dict[str, np.ndarray])
    """
    if not isinstance(maps, dict):
        raise TypeError("``maps`` must a type of dict and task specified")

    results = {}
    for tsk, outmaps in maps.items():
        if tsk == "keypoint":
            if imgs is None:
                raise ValueError("``imgs`` must be specified")
            ret_img = imshow_keypoint(imgs, outmaps)
        elif tsk == "segmentation":
            if imgs is None:
                raise ValueError("``imgs`` must be specified")
            ret_img = imshow_segmentation(imgs, outmaps)
        elif tsk == "edge":
            ret_img = imshow_edge(outmaps)
        elif tsk == "depth":
            ret_img = imshow_depth(outmaps)
        else:
            raise ValueError("unsupported task: {}".format(tsk))

        results[tsk] = ret_img

        if filepath is not None:
            # Add "task name" to save path
            save_dir = osp.dirname(filepath)
            filename, ext = osp.splitext(osp.basename(filepath))
            filename = tsk + filename + ext
            save_path = osp.join(save_dir, filename)
            cv2.imwrite(save_path, ret_img)

    return results
