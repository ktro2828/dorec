#!/usr/bin/env python

from tempfile import TemporaryDirectory

import pytest
import torch

from dorec.core import Config
from dorec.apis import inference

from .conftest import CONFIG_PATHS


def test_inference():
    if not torch.cuda.is_available():
        pytest.skip("test needs CUDA device")

    rgb_dir = "./tests/sample/data/HalfShirt/images/rgb"
    depth_dir = "./tests/sample/data/HalfShirt/images/depth"

    for path in CONFIG_PATHS:
        cfg = Config(path)

        with TemporaryDirectory() as td:
            cfg.work_dir = td

            inference(cfg, rgb_dir=rgb_dir, depth_dir=depth_dir)
