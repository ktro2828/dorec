#!/usr/bin/env python

import os.path as osp
import secrets

import cv2
from tqdm import tqdm

from dorec import TASK_GTMAP, GT_IMAGE_TYPES, GT_GEOMETRY_TYPES
from dorec.core.utils import load_yaml, save_json, save_yaml, save_txt, makedirs, TokenParser
from dorec.core.ops import load_keypoint, check_keypoint
from dorec.datasets import build_transforms

USAGE = """
task: <str, list[str]>
root <str>: ...root directory path
piplines <list[dict]>: ...transform pipelines
"""


def augment(config_filepath, out_dir=None):
    """Augment data
    Args:
        config_filepath(str): filepath of augmentation configuration
        out_dir (str, optional): directory path to save augmented data (default: None)
    """
    config = load_yaml(config_filepath)
    try:
        task = config.task
        root = config.root
        pipelines = config.pipelines
        num_repeat = config.num_repeat
    except Exception as e:
        print(e)
        print(USAGE)

    if isinstance(task, str):
        task = (task)
    with_keypoints = True if "keypoint" in task else False

    if out_dir is None:
        out_dir = root.rstrip("/") + "_augment"
    anno_dir = osp.join(out_dir, "annotations")
    makedirs(out_dir, exist_ok=False)
    makedirs(anno_dir, exist_ok=False)

    # Make directories to save augmented data
    img_dirs = {}
    for tsk in task:
        gt_type = TASK_GTMAP[tsk]
        dir_path = osp.join(out_dir, "images", gt_type)
        makedirs(dir_path, exist_ok=True)
        img_dirs[gt_type] = dir_path

    tp = TokenParser(root)

    # Data transformation
    transform = build_transforms(pipelines)

    num_data = len(tp)
    token_data = {}
    kpts_data = {}
    for idx in tqdm(range(num_data)):
        for n in tqdm(range(num_repeat), leave=False):
            # Generate token
            token = secrets.token_hex(16)
            # Images (RGB and depth)
            rgb = tp.get_filepath(idx, "rgb")
            data = {"inputs": rgb}

            targets = _load_targets(tp, idx, gt_type)
            # Keypoints
            if with_keypoints:
                pos = targets["keypoint"].pop("pos")
                vis = targets["keypoint"].pop("vis")
                in_pic = targets["keypoint"].pop("in_pic")
                targets["keypoint"] = pos
                data["targets"] = targets
                data = transform(data)
                pos = data["targets"]["keypoint"]
                h, w = data["inputs"].shape[:2]
                pos, vis, in_pic = check_keypoint(w, h, pos, vis, in_pic)
                ith_kpts_data = {}
                for i, (p, v, inp) in enumerate(zip(pos, vis, in_pic)):
                    ith_kpts_data.update(
                        {str(i): {"pos": p, "vis": v, "in_pic": inp}})
                kpts_data[token] = ith_kpts_data
            else:
                data["targets"] = targets
                data = transform(data)

            for gt_type, item in data["targets"].items():
                filename = str(idx) + str(n) + ".png"
                cv2.imwrite(osp.join(img_dirs[gt_type], filename), item)
                token_data[token].update({gt_type: filename})

    # Save annotation
    save_json(osp.join(anno_dir, "token.json"), token_data)
    save_json(osp.join(anno_dir, "keypoints.json"), kpts_data)
    save_txt(osp.join(anno_dir, "keypoint_names.txt"),
             tp.keypoint_names, mode="w")

    # Save augment info
    config.update({"original_root": root})
    save_yaml(osp.join(out_dir, "augment.yml"), config.to_dict())


def _load_targets(tp, idx, gt_type):
    out = {}
    if gt_type in GT_IMAGE_TYPES:
        fpath = tp.get_filepath(idx, gt_type)
        out[gt_type] = cv2.imread(fpath)
    elif gt_type in GT_GEOMETRY_TYPES:
        # TODO: support except keypoint
        kpts = tp.get_keypoints(idx)
        pos, vis, in_pic = load_keypoint(kpts, tp.keypoint_names)
        out["keypoint"]["pos"] = pos
        out["keypoint"]["vis"] = vis
        out["keypoint"]["in_pic"] = in_pic
    else:
        raise ValueError("unsupported gt_type: {}".format(gt_type))

    return out
