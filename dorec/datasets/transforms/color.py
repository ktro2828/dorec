#!/usr/bin/env python

import random

import cv2
import numpy as np

from dorec.core import TRANSFORMS
from .base import TransformBase


@TRANSFORMS.register()
class ColorJitter(TransformBase):
    """
    Copyright (c) OpenMMLab. All rights reserved.
    Apply photometric distrotion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is is
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    Args:
        brightness_delta (int): delta of brightness
        contrast_range (tuple): range of contrast
        saturation_range (tuple): range of saturation
        hue_delta (int): delta of hue
        name (str, optional): transform object name (default: None)
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
        name=None
    ):
        super(ColorJitter, self).__init__(name)
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Mltiple with alpha and add beta with clip"""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion"""
        if random.randint(0, 1):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
            )
        return img

    def contrast(self, img):
        """Contrast distortion"""
        if random.randint(0, 1):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower,
                                     self.contrast_upper)
            )
        return img

    def saturation(self, img):
        """Saturation distortion"""
        if random.randint(0, 1):
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper)
            )
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img

    def hue(self, img):
        """Hue distortion"""
        if random.randint(0, 1):
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
            img[:, :, 0] = (img[:, :, 0].astype(
                int) + random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img

    def __call__(self, data):
        """
        Args:
            data (dict[str, any])
        """
        mode = random.randint(0, 1)
        rgb = data["inputs"][:, :, :3]
        rgb = self.brightness(rgb)

        # mode == 0 -> do random contrast first
        # mode == 1 -> do random contrast last
        if mode == 1:
            rgb = self.contrast(rgb)

        rgb = self.saturation(rgb)
        rgb = self.hue(rgb)

        if mode == 0:
            rgb = self.contrast(rgb)
        data["inputs"][:, :, :3] = rgb

        return data


@TRANSFORMS.register()
class EdgeFilter(TransformBase):
    """Edge filter
    Args:
        method (str): [laplacian, sobel, canny, pyramid]
        p (float, optional): transform probability (default: 0.5)
        name (str, optional): transform object name (default: None)
    """

    def __init__(self, method, p=1.0, name=None, **kwargs):
        super(EdgeFilter, self).__init__(name)
        assert 0.0 <= p and p <= 1.0
        self.p = p
        self.method = method.lower()
        if self.method == "laplacian":
            self.ksize = kwargs.get("ksize", 11)
            self.func = self._laplacian
        elif self.method == "sobel":
            self.ksize = kwargs.get("ksize", 11)
            self.func = self._sobel
        elif self.method == "canny":
            self.min_val = kwargs.get("min_val", 50)
            self.max_val = kwargs.get("max_val", 255)
            self.func = self._canny
        elif self.method == "pyramid":
            self.max_level = kwargs.get("max_level", 3)
            self.use_level = kwargs.get("use_level", 0)
            self.min_val = kwargs.get("min_val", 50)
            self.max_val = kwargs.get("max_val", 255)
            self.func = self._pyramid
        else:
            raise ValueError("unsupported method: {}".format(method))

        self.name = self.__class__.__name__ if name is None else name

    def _laplacian(self, image):
        return cv2.Laplacian(image, ddepth=cv2.CV_64F, ksize=self.ksize)

    def _sobel(self, image):
        dx = cv2.Sobel(image.copy(), cv2.CV_64F, 1, 0, ksize=self.ksize)
        dy = cv2.Sobel(image.copy(), cv2.CV_64F, 0, 1, ksize=self.ksize)
        return np.sqrt(dx ** 2 + dy ** 2)

    def _canny(self, image):
        return cv2.Canny(image, self.min_val, self.max_val)

    def _pyramid(self, image):
        h, w = image.shape[:2]
        tmp = image.copy()
        ret = [self._canny(image)]
        for idx in range(self.max_level - 1):
            tmp = cv2.pyrDown(tmp)
            ret.append(self._canny(tmp))
        return cv2.resize(ret[self.use_level], (w, h))

    def __call__(self, data):
        """
        Args:
            data (dict[str, any])
        """
        if random.random() < self.p:
            rgb_orig = data["inputs"][:, :, 3].astype(np.uint8)
            rgb_gray = cv2.cvtColor(rgb_orig, cv2.COLOR_BGR2GRAY)
            rgb_gray = self.func(rgb_gray)
            rgb = np.expand_dims(rgb_gray, axis=-1)
            rgb = np.tile(rgb, (1, 1, 3))
            data["inputs"][:, :, :3] = rgb

        return data
