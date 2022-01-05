#!/usr/bin/env python

from dorec.runners import Runner2d


def test(cfg):
    """
    Args:
        cfg (dorec.core.Config)
    """
    runner = Runner2d(cfg, is_test=True)
    runner.test()
