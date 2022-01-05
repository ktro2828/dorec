#!/usr/bin/env python

import os.path as osp
import secrets

import cv2
from tqdm import tqdm

from dorec.core.utils import load_yaml, save_json, save_txt, makedirs, TokenParser
from dorec.core.ops import load_keypoint
from dorec.datasets import build_transforms


def augment(root, config_filepath):
    """Augment data
    Args:
        root (str): root directory path
        config_config (dict): filepath of augmentation configuration
    """
    config = load_yaml(config_filepath)
    augment_info = config["augment_info"]
    num_repeat = config["num_repeat"]
    targets = config["targets"]
    with_keypoints = True if "keypoint" in targets else False

    out_dir = root.rstrip("/") + "_augment"
    anno_dir = osp.join(out_dir, "annotations")
    makedirs(out_dir, exist_ok=False)
    makedirs(anno_dir, exist_ok=False)

    img_dirs = {}
    for key in targets:
        dir_path = osp.join(out_dir, "images", key)
        makedirs(dir_path, exist_ok=True)
        img_dirs[key] = dir_path

    tp = TokenParser(root)
    keypoint_names = tp.keypoint_names

    # Data transformation
    transform = build_transforms(augment_info, compose=False)

    num_data = len(tp)
    token_data = {}
    kpts_data = {}
    for idx in tqdm(range(num_data)):
        for n in tqdm(range(num_repeat), leave=False):
            # Generate token
            token = secrets.token_hex(16)
            # Images (RGB and depth)
            targets_data = {}
            for key in targets:
                filepath = tp.get_filepath(idx, key)
                targets_data[key] = cv2.imread(filepath)

            # Keypoints
            if with_keypoints:
                keypoint_data = tp.get_keypoints(idx)
                keypoint_pos, keypoint_vis, keypoint_in_pic = load_keypoint(
                    keypoint_data, keypoint_names
                )
                targets_data["keypoint_pos"] = keypoint_pos
                targets_data["keypoint_vis"] = keypoint_vis
                targets_data["keypoint_in_pic"] = keypoint_in_pic
                targets_data_aug = transform(targets_data)
                ith_kpts_data = {}
                keypoint_pos_aug = targets_data_aug.pop("keypoint_pos")
                keypoint_vis_aug = targets_data_aug.pop("keypoint_vis")
                keypoint_in_pic_aug = targets_data_aug.pop("keypoint_in_pic")
                for i, (pos, vis, in_pic) in enumerate(zip(keypoint_pos_aug, keypoint_vis_aug, keypoint_in_pic_aug)):
                    ith_kpts_data.update(
                        {str(i): {"pos": pos, "vis": vis, "in_pic": in_pic}})
                kpts_data[token] = ith_kpts_data
            else:
                targets_data_aug = transform(targets_data)

            for key in targets:
                filename = str(idx) + str(n) + ".png"
                img = targets_data_aug[key]
                cv2.imwrite(osp.join(img_dirs[key], filename), img)
                token_data[token].update({key: filename})

    # Save annotation
    save_json(osp.join(anno_dir, "token.json"), token_data)
    save_json(osp.join(anno_dir, "keypoints.json"), kpts_data)
    save_txt(osp.join(anno_dir, "keypoint_names.txt"),
             keypoint_names, mode="w")

    # Save augment info
    config.update({"original_root": root})
    save_json(osp.join(out_dir, "augment_info.json"), config)
