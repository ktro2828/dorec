#!/usr/bin/env python

from copy import deepcopy
import random

import cv2
import numpy as np

from dorec.core import TRANSFORMS
from dorec import GT_IMAGE_TYPES, GT_GEOMETRY_TYPES, TASK_GTMAP

from .base import TransformBase
from .utils import warp_perspective, imrotate


@TRANSFORMS.register()
class RandomPerspective(TransformBase):
    """Random perspective
    Args:
        scale (tuple, optional)
        offset (int, optional)
        p (float, optional): transform probability (default: 0.5)
        name (str, optional): transform object name (default: None)
    """

    def __init__(self, scale=(0.0, 0.06), offset=10, p=0.5, name=None):
        super(RandomPerspective, self).__init__(name)
        assert scale[0] < scale[1]
        assert 0.0 < p <= 1.0

        self.scale_lower = scale
        self.scale_upper = (1 - scale[1], 1 - scale[0])
        self.offset = offset
        self.p = p

    def get_perspective_matrix(self, width, height):
        src = np.float32(self.get_mask_coord(width, height))
        dst = np.float32(
            [
                [self.offset, width],
                [self.offset, 0],
                [height - self.offset, 0],
                [height - self.offset, width]
            ]
        )

        M = cv2.getPerspectiveTransform(src, dst)
        return M

    def get_mask_coord(self, width, height):
        # pts1 ~ 4 is clock wise
        pts1 = (
            random.uniform(*self.scale_lower) * width,
            random.uniform(*self.scale_lower) * height
        )
        pts2 = (
            random.uniform(*self.scale_upper) * width,
            random.uniform(*self.scale_lower) * height
        )
        pts3 = (
            random.uniform(*self.scale_upper) * width,
            random.uniform(*self.scale_upper) * height
        )
        pts4 = (
            random.uniform(*self.scale_lower) * width,
            random.uniform(*self.scale_upper) * height
        )
        pts = np.array((pts1, pts2, pts3, pts4), dtype=np.int32)
        return pts

    def __call__(self, data):
        """
        Args:
            data (dict[str, any])
        """
        if random.random() < self.p:
            img = data["inputs"]
            height, width = img.shape[:2]
            M = self.get_perspective_matrix(width, height)
            data["inputs"] = warp_perspective(img, M, (height, width), "image")
            for img_type in GT_IMAGE_TYPES:
                if data["targets"].get(img_type) is not None:
                    img = data["targets"][img_type]
                    data["targets"][img_type] = warp_perspective(
                        img, M, (height, width), img_type
                    )

            for geo_type in GT_GEOMETRY_TYPES:
                if data["targets"].get(geo_type) is not None:
                    raise NotImplementedError(
                        "RandomPerspective for geometry data is under construction"
                    )

        return data


@TRANSFORMS.register()
class RandomCrop(TransformBase):
    """Random crop
    Args:
        crop_size (tuple)
        p (float, optional): transform probability (default: 0.5)
        name (str, optional): transform object name (default: None)
    """

    def __init__(self, crop_size, p=0.5, name=None):
        super(RandomCrop, self).__init__(name)
        assert crop_size[0] > 0 and crop_size[1] > 0
        assert 0.0 < p <= 1.0

        self.crop_size = crop_size
        self.p = p

    def _get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def _crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def _crop_geometry(self, geometries, crop_bbox):
        """Crop from ``geometries``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        geometries[:, 0] = np.clip(geometries[:, 0], crop_y1, crop_y2)
        geometries[:, 1] = np.clip(geometries[:, 1], crop_x1, crop_x2)

        return geometries

    def __call__(self, data):
        """
        Args:
            data (dict[str, np.ndarray])
        """
        if random.random() < self.p:
            img = data["inputs"]
            crop_bbox = self.get_crop_bbox(img)
            for img_type in GT_IMAGE_TYPES:
                if data["targets"].get(img_type) is not None:
                    # Repeat 10 times
                    data["targets"][img_type] = self.crop(
                        data["targets"][img_type], crop_bbox
                    )

            img = self._crop(img, crop_bbox)
            data["inputs"] = img

            for geo_type in GT_GEOMETRY_TYPES:
                if data["targets"].get(img_type) is not None:
                    raise NotImplementedError(
                        "RadnomCrop for geometry data is under construction"
                    )
                data["targets"][geo_type] = self._crop_geometry(
                    data["targets"][geo_type], crop_bbox
                )

        return data


@TRANSFORMS.register()
class RandomRotate(TransformBase):
    """Rotate random
    Args:
        degree (float, tuple[float]): Range of degrees to select from
        pad_val (float, optional): Padding value of image. (default 0)
        seg_pad_val (float, optional): Padding value of segmentation map
            (default 255)
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. (default: None)
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. (default False)
        p (float, optional): transform probability (default: 0.5)
        name (str, optional): transform object name (default: None)
    """

    def __init__(self, degree, pad_val=0, center=None, auto_bound=False, p=0.5, name=None):
        super(RandomRotate, self).__init__(name)
        if isinstance(degree, (float, int)):
            assert degree > 0, "degree {} should be positive".format(degree)
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, "degree {} should be a " "tuple of (min, max)".format(
            self.degree)
        assert 0.0 < p <= 1.0

        self.pad_val = pad_val
        self.center = center
        self.auto_bound = auto_bound
        self.p = p

    def __call__(self, data):
        """
        Arg:
            data (dict[str, any])
        """
        if random.random() < self.p:
            degree = np.random.uniform(min(*self.degree), max(*self.degree))

            data["inputs"] = imrotate(
                data["inputs"],
                angle=degree,
                border_value=self.pad_val,
                center=self.center,
                auto_bound=self.auto_bound,
                interpolation="nearest"
            )

            for task, item in data["targets"].items():
                gt_type = TASK_GTMAP[task]
                if gt_type in GT_IMAGE_TYPES:
                    data["targets"][task] = imrotate(
                        item,
                        angle=degree,
                        border_value=self.pad_val,
                        center=self.center,
                        auto_bound=self.auto_bound,
                        interpolation="nearest"
                    )
                elif gt_type in GT_GEOMETRY_TYPES:
                    raise NotImplementedError(
                        "RandomRotate for geometry data is under construction"
                    )

        return data


@TRANSFORMS.register()
class VerticalFlip(TransformBase):
    """Vertical Flip
    Args:
        p (float, optional): transform probability (default: 0.5)
        name (str, optional): transform object name (default: None)
    """

    def __init__(self, p=0.5, name=None):
        super(VerticalFlip, self).__init__(name)
        assert 0.0 < p <= 1.0

        self.p = p

    def __call__(self, data):
        """
        Args:
            data (dict[str, any])
        """
        if random.random() < self.p:
            data["inputs"] = np.flipud(data["inputs"]).copy()

            for task, item in data["targets"].items():
                gt_type = TASK_GTMAP[task]
                if gt_type in GT_IMAGE_TYPES:
                    data["targets"][task] = np.flipud(item).copy()
                elif gt_type in GT_GEOMETRY_TYPES:
                    h = data["inputs"].shape[0]
                    item[:, 1] = h - item[:, 1]
                    tmp = deepcopy(item)
                    item[::2, :] = tmp[1::2, :]
                    item[1::2, :] = tmp[::2, :]
                    data["targets"][task] = item

        return data


@TRANSFORMS.register()
class HorizontalFlip(TransformBase):
    """Horizontal Flip
    Args:
        p (float, optional): transform probability (default: 0.5)
        name (str, optional): transform object name (default: None)
    """

    def __init__(self, p=0.5, name=None):
        super(VerticalFlip, self).__init__(name)
        assert 0.0 < p <= 1.0

        self.p = p

    def __call__(self, data):
        """
        Args:
            data (dict[str, any])
        """
        if random.random() < self.p:
            data["inputs"] = np.flipud(data["inputs"]).copy()

            for task, item in data["targets"].items():
                gt_type = TASK_GTMAP[task]
                if gt_type in GT_IMAGE_TYPES:
                    data["targets"][task] = np.fliplr(item).copy()
                elif gt_type in GT_GEOMETRY_TYPES:
                    w = data["inputs"].shape[1]
                    item[:, 0] = w - item[:, 0]
                    data["targets"][task] = item

        return data


@TRANSFORMS.register()
class RandomFlip(TransformBase):
    """Flip verticaly or horizontaly
    Args:
        p (float, optional): transform probability (default: 0.5)
        name (str, optional): transform object name (default: None)
    """

    def __init__(self, p=0.5, name=None):
        super(RandomFlip, self).__init__(name)
        assert 0.0 < p <= 1.0

        self.vflip = VerticalFlip(p)
        self.hflip = HorizontalFlip(p)
        self.p = p

    def __call__(self, data):
        """
        Args:
            data (dict[str, any])
        """
        data = self.vflip(data)
        data = self.hflip(data)

        return data
