#!/usr/bin/env python

from dorec import TASK_GTMAP
from dorec.core import DATASETS

from .bases import TokenDataset2D
from .utils import load_gt_img
from .transforms import build_transforms


@DATASETS.register()
class ImageTaskDataset(TokenDataset2D):
    """Dataset class for image task, (segmentation, edge and depth)

    Args:
        task (tuple(str))
        root (str)
        input_type (str)
        use_dims (int, optional)
        pipelines (list[dict[str, any]])
        num_classes (int, optional)
    """

    def __init__(self, task, root, input_type, use_dims, pipelines, num_classes=4):
        super(ImageTaskDataset, self).__init__(
            task, root, input_type, use_dims, pipelines)
        if "keypoint" in task:
            raise ValueError("use KeypointTaskDataset()")
        self.num_classes = num_classes
        self.transform = build_transforms(pipelines, compose=True)

    def _load_targets(self, idx):
        out = {}
        for task in self.task:
            img_type = TASK_GTMAP[task]
            filepath = self.tp.get_filepath(idx, img_type)
            img = load_gt_img(filepath, img_type, num_classes=self.num_classes)
            out[task] = img
        return out

    def __getitem__(self, idx):
        images = self._load_inputs(idx)
        targets = self._load_targets(idx)

        data = {"inputs": images, "targets": targets}

        data = self.transform(data)

        return data
