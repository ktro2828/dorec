#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp

from dorec import IMAGE_TYPES
from .io import load_json, load_txt, save_json
from .logger import get_logger

logger = get_logger(modname=__name__)

FILE_PATH_USAGE = """
[!!]Assuming file system like below,

root/
├── annotations
│   ├── [Optional] camera_info.json
│   ├── [Optional] keypoint_names.json
│   ├── [Optional] keypoints.json
│   └── token.json
└── images
    ├── depth/
    ├── edge/
    ├── mask/
    └── rgb/
"""


class TokenParser:
    """Parse token and convert token to file names, pathes or annotation data

    Args:
        root (str): Root directory path of data (especially in cloth name)
    """

    def __init__(self, root):
        # Load annotations as dict
        assert osp.exists(root), "{} is not exists".format(root)
        self._root = root
        self.anno_root = osp.join(root, "annotations")
        self.image_root = osp.join(root, "images")

        # Load annotations
        self._tokens = load_json(osp.join(self.anno_root, "token.json"))

        self._camera_infos = load_json(
            osp.join(self.anno_root, "camera_info.json"))
        self._keypoints = load_json(osp.join(self.anno_root, "keypoints.json"))
        self._keypoint_names = load_txt(
            osp.join(self.anno_root, "keypoint_names.txt"))

        # Hold tokens as list
        self._tokens_list = list(self._tokens.keys())

        # Hold directory path for each image type as dict
        self.img_dirs = {}
        for img_type in IMAGE_TYPES:
            dir_path = osp.join(self.image_root, img_type)
            if not osp.exists(dir_path):
                logger.warn("{} is not exists".format(dir_path))
            else:
                self.img_dirs[img_type] = dir_path

    def __getitem__(self, idx):
        if self.flag:
            return self.tokens_list[idx]
        else:
            return None

    def __len__(self):
        return len(self.tokens_list)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= self.__len__():
            ret = self.__getitem__(self.n)
            self.n += 1
            return ret
        else:
            raise StopIteration

    def get(self, key, default=None):
        return self._tokens.get(key, default)

    def update(self, m):
        self._token.update(m)

    @property
    def tokens(self):
        return self._tokens

    @property
    def camera_infos(self):
        return self._camera_infos

    @property
    def keypoints(self):
        return self._keypoints

    @property
    def tokens_list(self):
        return self._tokens_list

    @property
    def class_names(self):
        return self._class_names

    @property
    def keypoint_names(self):
        return self._keypoint_names

    def get_filename(self, key, filetype):
        """Returns file name related to filetype[rgb, depth, edge]

        Args:
            token (str): Unique record identifier
            filetype (str): File format types, [rgb, depth, edge]

        Retuns:
            (str) : Name of target file
        """
        token = self._checkkey(key)
        assert filetype in IMAGE_TYPES, \
            "Unsupported file type {}".format(filetype)
        return self.tokens[token][filetype]

    def get_filepath(self, key, filetype):
        """Returns file path related to filetype [rgb, depth, mask, edge]

        Args:
            token (str): Unique record identifier
            filetype (str): File format types, [rgb, depth, make, edge]

        Returns:
            (str): Path to target file
        """
        filename = self.get_filename(key, filetype)
        return osp.join(self.img_dirs[filetype], filename)

    def save_token(self, filepath=None):
        """Save updated token file to original path, if filepath is None

        Args:
            filepath (str, optional): Path to save token.json, if None save to original path
        """
        if filepath is None:
            save_json(osp.join(self.anno_root, "token.json"), self.tokens)
        else:
            save_json(filepath, self.tokens)

    def get_token(self, idx):
        return self.tokens_list[idx]

    def get_camerainfo(self, key):
        """Returns data of camera info

        Args:
            key (int or str): `int` represents index of token list, `str` represents key name of token dict

        Returns:
            camera_info (dict[str, any])
        """
        assert self.camera_infos is not None, "camera_info.json is not loaded"
        token = self._checkkey(key)
        return self.camera_infos[token]

    def get_keypoints(self, key):
        """Returns data of keypoints

        Args:
            key (int or str): `int` represents index of token list, `str` represents key name of token dict

        Returns:
            keypoints (dict[str, any])
        """
        assert self.keypoints is not None, "keypoints.json is not loaded"
        token = self._checkkey(key)
        return self.keypoints[token]

    def _checkkey(self, key):
        """Check key and returns token

        Args:
            key (int, str)

        Returns:
            token (str)
        """
        if isinstance(key, int):
            return self.get_token(key)
        elif isinstance(key, str):
            return key
        else:
            raise TypeError(
                "type of `key` must be int or str, but got {}".format(type(key)))
