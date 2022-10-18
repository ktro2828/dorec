#!/usr/bin/env python
from tempfile import TemporaryDirectory

import pytest
import torch

from dorec.core import Config
from dorec.apis import show_data
from .conftest import CONFIG_PATHS


def test_show_data():
    if not torch.cuda.is_available():
        pytest.skip("test needs CUDA device")

    for path in CONFIG_PATHS:
        cfg = Config(path)

        with TemporaryDirectory() as td:
            cfg.work_dir = td
            show_data(cfg, is_test=True, max_try=2)
