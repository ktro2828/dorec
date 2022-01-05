#!/usr/bin/env python

import cv2
import numpy as np

from dorec.core.utils import TokenParser

from .custum_dataset2d import CustumDataset2D


class TokenDataset2D(CustumDataset2D):
    """Base dataset class for our token specified style
    Args:
        task (tuple(str))
        root (str)
        input_type (str)
        use_dims (int)
        pipelines (list[dict[str, any]])
    """

    def __init__(self, task, root, input_type, use_dims, pipelines):
        super(TokenDataset2D, self).__init__(
            task, root, input_type, use_dims, pipelines)
        self.tp = TokenParser(self.root)

    def _load_inputs(self, idx):
        """Load inputs
        Args:
            idx (int)
        Returns:
            image (np.ndarray)
        """
        if self.input_type == "rgbd":
            color_path = self.tp.get_filepath(idx, "rgb")
            color = cv2.imread(color_path)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            depth_path = self.tp.get_filepath(idx, "depth")
            depth = cv2.imread(depth_path, 0)
            depth = np.expand_dims(depth, axis=-1)
            depth = np.tile(depth, self.reps)
            image = np.concatenate((color, depth), axis=-1)
        else:
            image_path = self.tp.get_filepath(idx, self.image_type)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.use_edge:
                image = cv2.Canny(image, self.min_th, self.max_th)
                image = np.expand_dims(image, axis=-1)
            image = np.tile(image, self.reps)

        return image
