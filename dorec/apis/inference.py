#!/usr/bin/env python

from dorec.runners import Runner2d


def inference(cfg, rgb_dir=None, depth_dir=None):
    """
    Args:
        cfg (dorec.core.Config)
        rgb_dir (str, optional)
        depth_dir (str, optional)
    """
    runner = Runner2d(cfg, is_test=True)

    if (rgb_dir is None) and (depth_dir is None):
        raise NotImplementedError(
            "at least, rgb_dir or depth_dir must be specified")

    runner.inference(rgb_dir=rgb_dir, depth_dir=depth_dir)
