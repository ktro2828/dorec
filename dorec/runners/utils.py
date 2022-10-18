#!/usr/bin/env python

import os
from time import time

import numpy as np
import torch
import torch.nn as nn

from dorec.core.utils import get_logger
logger = get_logger(modname=__name__)


def weight_init_fn(m):
    if hasattr(m, "classname"):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(1e-4)
        elif m.classname.find("Linear") != -1:
            m.weight.data.normal_(0.0, 1e-4)


def worker_init_fn(x):
    return np.random.seed(x + int(time()))


def parse_device(device, gpu_ids):
    """Parse available device
    Args:
        device (str)
        gpu_ids (str)
    Returns:
        device (torch.device)
    """
    if device == "gpu":
        if not torch.cuda.is_available():
            raise Exception("Cannot find cuda device")
        logger.info("cuda visible devices: {}".format(gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    logger.info("Use device: {}".format(device))

    return device


class DataParallel(nn.DataParallel):
    def __init__(self, model, device_ids=None, output_device=None, dim=0):
        super().__init__(model, device_ids, output_device, dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
