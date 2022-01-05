#!/usr/bin/env python

import random

import torch
from torchvision.transforms import ToTensor as ToTensorBase
from torchvision.transforms import Normalize

from dorec import GT_IMAGE_TYPES
from dorec.core import TRANSFORMS
from dorec.core.utils import build_from_cfg
from dorec.core.ops import normalize_depth

from .utils import imresize
from .base import TransformBase


def build_transforms(pipelines, compose=True):
    """
    Args:
        pipelines (list[dict[str, any]])
        compose (bool, optional)
    Returns:
        if compose=True:
            ``obj`` (Compose)
        else:
            ``list[transforms]``
    """
    assert isinstance(pipelines, (list, tuple))
    assert isinstance(compose, bool)
    transforms = []
    for pipeline in pipelines:
        transforms.append(build_from_cfg(pipeline, TRANSFORMS))

    if compose:
        return Compose(transforms)
    else:
        return transforms


@TRANSFORMS.register()
class Compose(TransformBase):
    """Compose several augmentations together
    Args:
        transforms (list[transforms])
        with_normalize (bool): whether normalize image (default: True)
        name (str, optional): transform object name (default: None)
    """

    def __init__(self, transforms, name=None):
        super(Compose, self).__init__(name)
        self.transforms = transforms
        self.normalize = Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)

        # Normalize RGB
        img = data["inputs"].float()
        img[:3, :, :] = self.normalize(img[:3, :, :])
        if len(img) == 4:
            # In case of RGBD, Normalize Depth
            img[3, :, :] = normalize_depth(img[3, :, :])

        data["inputs"] = img

        return data


@TRANSFORMS.register()
class RandomApply(TransformBase):
    """Apply several transforms randomly
    Args:
        pipelines (list[dict[str, any]])
        name (str, optional): transform object name (default: None)
    """

    def __init__(self, pipelines, name=None):
        super(RandomApply, self).__init__(name)
        self.trans = build_transforms(pipelines, compose=False)
        self.num_trans = len(self.trans)

    def __call__(self, data):
        trans_idx = random.randint(0, self.num_trans - 1)
        t = self.trans[trans_idx]
        data = t(data)

        return data


@TRANSFORMS.register()
class Resize(TransformBase):
    """Resize image
    Args:
        size (union[tuple, list, int])
        name (str, optional): transform object name (default: None)
    """

    def __init__(self, size, name=None):
        super(Resize, self).__init__(name)
        if isinstance(size, int):
            self.size = tuple((size, size))
        elif isinstance(size, (list, tuple)):
            self.size = size

        assert len(self.size) == 2

    def __call__(self, data):
        """
        Args:
            data (dict[str, any])
        """
        data["inputs"] = imresize(data["inputs"], self.size, img_type="image")

        for img_type in GT_IMAGE_TYPES:
            if data["targets"].get(img_type) is not None:
                data["targets"][img_type] = imresize(
                    data["targets"][img_type], self.size, img_type=img_type)

        return data


@TRANSFORMS.register()
class ToTensor(TransformBase):
    """Convert to torch.Tesor
    Args:
        name (str, optional): transform object name (default: None)
    """

    def __init__(self, name=None):
        super(ToTensor, self).__init__(name=name)
        self.to_tensor = ToTensorBase()

    def __call__(self, data):
        data["inputs"] = self.to_tensor(data["inputs"])

        for img_type in GT_IMAGE_TYPES:
            if data["targets"].get(img_type) is not None:
                data["targets"][img_type] = self.to_tensor(
                    data["targets"][img_type])

        for key, item in data["targets"].items():
            data["targets"][key] = torch.from_numpy(item)

        return data
