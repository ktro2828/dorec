#!/usr/bin/env python

from tempfile import TemporaryDirectory

import pytest
import torch

from dorec.apis import test
from dorec.core import Config

from .conftest import CONFIG_PATHS


def test_test():
    if not torch.cuda.is_available():
        pytest.skip("test needs CUDA device")

    for path in CONFIG_PATHS:
        cfg = Config(path)

        with TemporaryDirectory() as td:
            cfg.work_dir = td

            test(cfg)
