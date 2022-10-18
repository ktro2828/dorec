#!/usr/bin/env python

from collections import OrderedDict
import errno
import json
import os
import os.path as osp
import sys

from box import Box
import numpy as np
import torch

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
import yaml

from .logger import get_logger

logger = get_logger(modname=__name__)


def load_json(filename, mode="r", strict=False):
    """Load json
    Args:
        filename (str): json file path
        mode (str, optional)
        strict (bool, optional): if true, in case the path doesn't exist, raises error
    Returns:
        data (Box[str, any]): dict data allowed dot access
    """
    if not osp.exists(filename):
        if strict:
            raise FileNotFoundError("no such file `{}`".format(filename))
        else:
            logger.warning("no such file `{}`".format(filename))
            return None

    with open(filename, mode) as f:
        data = json.load(f)

    # Allow dot access
    data = Box(data)

    return data


def save_json(filename, data, mode="w", sort_keys=True):
    """Save json file

    Args:
        filename (str): path to save .json file
        data (dict[str, any])
        mode (str, optional)
        sort_keys (bool, optional): indicates whether save sorting keys
    """
    if mode == "a":
        data_orig = load_json(filename, mode="r")
        if data_orig is not None:
            data.update(data_orig)

    with open(filename, mode) as f:
        json.dump(data, f, ensure_ascii=False, indent=4,
                  sort_keys=sort_keys, separators=(",", ": "), cls=NpEncoder)


def load_txt(filename, mode="r", strict=False):
    """Load txt file
    Args:
        fiilename (str): txt file path
        mode (str, optional)
        strict (bool, optional): if true, in case the path doesn't exist, raises error
    Returns:
        data [list[str]]: loaded data
    """
    if not osp.exists(filename):
        if strict:
            raise FileNotFoundError("no such file `{}`".format(filename))
        else:
            logger.warning("no such file `{}`".format(filename))
            return None

    data = []
    with open(filename, mode) as f:
        for line in f:
            data.append(line.rstrip("\n"))

    return data


def save_txt(filename, data, mode="w"):
    """Save data as txt file
    Args:
        filename (str)
        data (list[any])
        mode (str, optional)
    """
    with open(filename, mode) as f:
        if isinstance(data, (list, tuple)):
            for s in data:
                f.write("%s\n" % s)
        else:
            f.write(data)


def load_yaml(filename, mode="r", strict=True, as_orderdict=True):
    """Load yaml file
    Args:
        filename (str)
        mode (str, optional):
        strict (bool, optional)
        as_orderdict (bool, optional)
    Returns:
        data (Box[str, any]): dict data allowed dot access
    """
    if not osp.exists(filename):
        if strict:
            raise FileNotFoundError("no such file `{}`".format(filename))
        else:
            logger.warning("no such file `{}`".format(filename))
            return None

    if as_orderdict:
        # load as OrderedDict
        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            lambda loader, node: OrderedDict(loader.construct_pairs(node)),
        )
        Loader = OrderedLoader
    else:
        Loader = yaml.SafeLoader

    with open(filename, mode) as f:
        data = yaml.load(f, Loader=Loader)

    # Allow dot access
    data = Box(data)

    return data


def save_yaml(filename, data, mode="w"):
    """Save data as .yml file

    Args:
        filename (str)
        data (dict[str, any])
        mode (str, optional)
    """
    if mode == "a":
        data_orig = load_yaml(filename, mode="r")
        if data_orig is not None:
            data.update(data_orig)

    with open(filename, mode) as f:
        yaml.dump(data, f)


def load_from_url(url, cache_dir="./data/pretrained", map_location=torch.device("cpu")):
    """Load pretrained model weights from url
    Args:
        url (str)
        cache_dir (str)
        map_location (optional)
    Returns:
        weights (OrderedDict)
    """
    makedirs(cache_dir, exist_ok=True)
    filename = url.split("/")[-1]
    cached_file = osp.join(cache_dir, filename)
    if not osp.exists(cached_file):
        sys.stderr.write("Downloading: '{}' to {}\n".format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def load_model(model, filename, strict=True, eval=False, freeze=False):
    """Load pre-trained torch.nn.Module
    Args:
        model (torch.nn.Module)
        filename (str)
        strict (bool, optional)
        eval (bool, optional)
        freeze (bool, optional)
    Returns:
        model (torch.nn.Module)
    """
    device = next(model.parameters()).device
    state_dict = torch.load(filename, map_location=device)
    if "model" in state_dict.keys():
        model.load_state_dict(state_dict["model"], strict=strict)
    else:
        logger.warning(
            "type of state_dict is deprecated, make it contain keys `model` and `optimizer`")
        model.load_state_dict(state_dict, strict=strict)

    if eval:
        model.eval()
    else:
        model.train()

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        logger.info("{} is freezed".format(model.name))

    return model


def save_model(model, filename):
    """Save torch.nn.Module
    Args:
        model (torch.nn.Module)
        filename (str): path to save weights
    """
    if not filename.endswith(".pth"):
        raise ValueError("'filename' must end with .pth")

    if isinstance(model, torch.nn.DataParallel):
        state = model.module.state_dict()
    else:
        state = model.state_dict()

    logger.info("Saving model to: {}".format(filename))
    torch.save(state, filename)


def makedirs(name, exist_ok=True):
    """Make directories, supporting args ``exist_ok`` for python2.7
    Args:
        name (str)
        exist_ok (bool, optional)
    """
    if exist_ok:
        if not osp.exists(name):
            os.makedirs(name=name)
    else:
        try:
            os.makedirs(name=name)
        except FileExistsError:
            raise

        if osp.exists(name):
            FileExistsError("File exists: {}".format(name))
        os.makedirs(name=name)


class NpEncoder(json.JSONEncoder):
    """Encoder to avoid error with numpy"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class FileExistsError(OSError):
    """for python2.7"""

    def __init__(self, msg):
        super(FileExistsError, self).__init__(errno.EEXIST, msg)


class FileNotFoundError(IOError):
    """for  python2.7"""

    def __init__(self, msg):
        super(FileNotFoundError, self).__init__()


class OrderedLoader(yaml.SafeLoader):
    pass
