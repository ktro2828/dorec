#!/usr/bin/env python

from box import Box

from dorec.core.utils import DATASETS

from .bases import ConcatDataset


def build_dataset(cfg, input_type=None, use_dims=None):
    """Build dataset
    Args:
        cfg (Box[str, any])
        input_type (str, optional)
        use_dims (int, optional)
    Returns:
        dataset (torch.utils.data.Dataset)
    """
    if not isinstance(cfg, Box):
        raise TypeError(
            "``cfg`` must be a type of Box, but got {}".format(type(cfg)))

    if input_type is not None:
        if not isinstance(input_type, str):
            raise TypeError(
                "``input_type`` must be a type of str, but got {}".format(type(input_type)))
        cfg.input_type = input_type
    if use_dims is not None:
        if not isinstance(use_dims, int):
            raise TypeError(
                "``use_dims`` must be a type of int, but got {}".format(type(use_dims)))
        cfg.use_dims = use_dims

    # Check if input_type and use_dims are specified
    if cfg.get("input_type") is None:
        raise NotImplementedError("``input_type`` must be specified")
    if cfg.get("use_dims") is None:
        raise NotImplementedError("``use_dims`` must be specified")

    root = cfg.root
    if isinstance(root, str):
        dataset = DATASETS.build(cfg)
    elif isinstance(root, (list, tuple)):
        datasets = []
        for r in root:
            cfg.root = r
            datasets.append(build_dataset(cfg))
        dataset = ConcatDataset(datasets)
    else:
        raise TypeError(
            "``root`` must be a type of str or iterable, not {}".format(type(root)))

    return dataset
