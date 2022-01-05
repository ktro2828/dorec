#!/usr/bin/env python

import random

import numpy as np
from skimage.util import random_noise

from dorec.core.utils import TRANSFORMS
from .base import TransformBase


@TRANSFORMS.register()
class RandomNoise(TransformBase):
    """Random noise
    Args:
        mode (list[str], str, optional)
        target (str, optional)
        p (float, optional): transform probability (default: 0.5)
        name (str, optional): transform object name (default: None)
    """
    VALID_MODE = (
        "gaussian",
        "localvar",
        "poisson",
        "salt",
        "pepper",
        "s&p",
        "speckle"
    )

    def __init__(self, mode=None, target=None, p=0.5, name=None):
        super(RandomNoise, self).__init__(name)
        assert 0.0 <= p <= 1.0

        self.mode = self.VALID_MODE if mode is None else mode
        self.target = target.lower() if target is not None else None
        self.p = p

        if set(self.mode) > set(self.VALID_MODE):
            raise ValueError("unexpected mode: {}".format(self.mode))
        if self.target not in (None, "rgb", "depth"):
            raise ValueError("unexpected target: {}".format(self.target))

    def _apply_noise(self, img):
        if isinstance(self.mode, (list, tuple)):
            mode_idx = random.randint(0, len(self.mode) - 1)
            mode = self.mode[mode_idx]
        else:
            mode = self.mode

        img = img / 255.0

        img = random_noise(img, mode=mode)

        return img * 255

    def __call__(self, data):
        """
        Args:
            data (dict[str, any])
        """
        if random.random() < self.p:
            if self.target is None:
                data["inputs"] = self._apply_noise(data["inputs"])
            elif self.target == "rgb":
                data["inputs"][:, :, :3] = self._apply_noise(
                    data["inputs"][:, :, :3])
            elif self.target == "depth":
                data["inputs"][:, :, -1] = self._apply_noise(
                    data["inputs"][:, :, -1])

        return data

    def __repr__(self):
        return self.name


@TRANSFORMS.register()
class RandomErase(TransformBase):
    """Random erasing
    Args:
        sl (float, optional)
        sh (float, optional)
        r1 (float, optional)
        r2 (float, optional)
        occlusion (bool, optional): whether consider occlusiton
        p (float, optional): transform probability (default: 0.5)
        name (str, optional): transform object name (default: None)
    """

    def __init__(self, sl=0.02, sh=0.4, r1=0.3, r2=3.3, p=0.5, occlusion=False, name=None):
        super(RandomErase, self).__init__(name)
        assert 0.0 <= p <= 1.0

        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2
        self.p = p
        self.occlusion = occlusion

    def __call__(self, data):
        if random.random() < self.p:
            img = data["inputs"]
            h, w, c = img.shape
            s = h * w

            while True:
                se = np.random.uniform(self.sl, self.sh) * s
                re = np.random.uniform(self.r1, self.r2)

                he = int(np.sqrt(se * re))
                we = int(np.sqrt(se / re))

                xe = np.random.randint(0, w)
                ye = np.random.randint(0, h)

                if xe + we <= w and ye + he <= h:
                    break

            rect = np.random.randint(0, 255, (he, we, 3))
            img[ye: ye + he, xe: xe + we, :3] = rect

            if c == 4:
                max_val = img[:, :, -1].max()
                min_val = img[:, :, -1].min()
                d_rect = np.random.uniform(min_val, max_val, (he, we))
                img[ye: ye + he, xe: xe + we, -1] = d_rect

            data["inputs"] = img

            if self.occlusion and data["targets"].get("keypoints") is not None:
                kpts = data["targets"]["keypoints"]
                indices = (xe <= kpts[:, 0]) * (kpts[:, 0] <= xe + we) * \
                    (ye <= kpts[:, 1]) * (kpts[:, 1] <= ye + he)
                data["targets"]["keypoints"][indices] = -1

        return data
