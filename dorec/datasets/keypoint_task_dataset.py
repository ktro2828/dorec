#!/usr/bin/env python

from dorec.core import DATASETS
from dorec.core.ops import load_keypoint, check_keypoint, gen_heatmap

from .image_task_dataset import ImageTaskDataset


@DATASETS.register()
class KeypointTaskDataset(ImageTaskDataset):
    """Dataset class including keypoint task
    Args:
        task (tuple(str))
        root (str)
        input_type (str)
        use_dims (int)
        pipelines (list[dict[str, any]])
        num_keypoints (int)
    """

    def __init__(self, task, root, input_type, use_dims, pipelines, num_keypoints, num_classes=None, gaussian_R=8):
        super(KeypointTaskDataset, self).__init__(
            task, root, input_type, use_dims, pipelines, num_classes)
        self.num_keypoints = num_keypoints
        self.gaussian_R = gaussian_R

    def _load_keypoints(self, idx):
        data = self.tp.get_keypoints(idx)
        pos, vis, in_pic = load_keypoint(data)
        return pos, vis, in_pic

    def __getitem__(self, idx):
        images = self._load_inputs(idx)
        img_targets = self._load_targets(idx)
        pos, vis, in_pic = self._load_keypoints(idx)

        targets = {}
        targets.update(img_targets)
        targets.update({"keypoint": pos})
        data = {"inputs": images, "targets": targets}
        data = self.transform(data)

        h, w = data["inputs"].shape[1:]
        # Check keypoints visivility
        trans_pos = data["targets"].pop("keypoint")
        pos, vis, in_pic = check_keypoint(w, h, trans_pos, vis, in_pic)

        # Generate heatmap
        heatmap = gen_heatmap(w, h, in_pic, trans_pos, self.gaussian_R)

        data["targets"]["keypoint"] = heatmap

        return data
