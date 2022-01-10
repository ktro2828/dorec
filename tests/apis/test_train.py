#!/usr/bin/env python

from tempfile import TemporaryDirectory

import pytest
import torch

from dorec.apis import train
from dorec.core import Config

from .conftest import CONFIG_PATHS, VALID_TASKS


def test_train():
    if not torch.cuda.is_available():
        pytest.skip("test needs CUDA device")

    for path in CONFIG_PATHS:
        cfg = Config(path)

        with TemporaryDirectory() as td:
            cfg.work_dir = td
            assert set(cfg.task) <= set(VALID_TASKS)

            train(cfg)
