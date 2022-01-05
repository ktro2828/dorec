#!/usr/bin/env python

from dorec.runners import Runner2d


def show_data(cfg, is_test, max_try):
    """
    Args:
        cfg (dorec.core.Config)
        is_test (bool)
        max_try (int)
    """
    runner = Runner2d(cfg, is_test)
    runner.show_data(max_try)
