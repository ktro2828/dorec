#!/usr/bin/env python

import os.path as osp

from dorec import TASK_TYPES

from .io import load_yaml, save_yaml
from .text import pretty_text as _pretty_text

# Keys the config must include
CONFIG_KEYS = (
    "task",
    "parameters",
    "model",
    "dataset",
    "loss",
    "evaluation",
    "optimizer",
    "scheduler"
)

PHASES = ("train", "test")


class _ConfigBase(object):
    """Base class of Config

    Args:
        filename (str): path of config file
    """

    def __init__(self, filename):
        self._config = load_yaml(filename, strict=True, as_orderdict=True)
        self._name = osp.splitext(osp.basename(filename))[0]

        # Register main keys
        _task = self._config.task
        # Hold task name(s) as sorted tuple
        if isinstance(_task, str):
            _task = (_task, )
        sorted_list = sorted(_task)
        self._task = tuple(sorted_list)

        self._parameters = self._config.parameters
        self._model = self._config.model
        self._dataset = self._config.dataset
        self._loss = self._config.loss
        self._evaluation = self._config.evaluation
        self._optimizer = self._config.optimizer
        self._scheduler = self._config.scheduler

        # Check keys
        self._check_keys()

    def _check_keys(self):
        """Check keys of config file"""
        # Check all keys
        assert isinstance(self._task, tuple)
        if set(self.keys()) != set(CONFIG_KEYS):
            raise KeyError(
                "config file must have keys: {}, but got {}".format(CONFIG_KEYS, self.keys()))

        # Check ``task``
        if not set(self._task) <= set(TASK_TYPES):
            raise KeyError(
                "expected tasks: {}, but got {}".format(TASK_TYPES, self._task))

        # Check ``parameters``
        if not set(self._parameters.keys()) >= set(PHASES):
            raise KeyError(
                "parameters must have train/test at least, but got {}".format(self._parameters.keys()))

        # Check ``model``
        if (set(self._model.keys()) != set(("backbones", "heads"))) and ("name" not in self.model.keys()):
            raise KeyError(
                "model must specifiy 'backbones' and 'heads', or 'name', but got {}".format(self._model.keys()))

        # Check ``dataset``
        if not set(self._dataset.keys()) >= set(PHASES):
            raise KeyError(
                "dataset must have train/test at least, but got {}".format(self._dataset.keys()))

        # Check ``evaluation``
        if set(self._task) != set(self._evaluation.keys()):
            raise KeyError("evaluation must have keys for each task, but got {}".format(
                self._evaluation.keys()))

        # Check ``loss``
        if ("name" not in self._loss.keys()):
            raise KeyError(
                "loss must specify 'name', but got {}".format(self._loss.keys()))
        if len(self._task) > 1 and (set(self._task) >= set(self._loss.keys())):
            raise KeyError(
                "in case of tackle multi task, loss must have keys for each task, but got {}".format(self._loss.keys()))

        # Check ``optimizer``
        if "name" not in self._optimizer.keys():
            raise KeyError("optimizer must specify 'name', but got {}".format(
                self._optimizer.keys()))

        # Check ``scheduler``
        if "name" not in self._scheduler.keys():
            raise KeyError("scheduler must specify 'name', but got {}".format(
                self._scheduler.keys()))

    def get(self, key, default=None):
        return self._config.get(key, default)

    def update(self, m):
        self._config.update(m)

    def keys(self):
        return self._config.keys()

    def items(self):
        return self._config.items()

    def __repr__(self):
        return self._name

    @ property
    def name(self):
        return self._name

    @ property
    def task(self):
        return self._task

    @ property
    def parameters(self):
        return self._parameters

    @ property
    def model(self):
        return self._model

    @ property
    def dataset(self):
        return self._dataset

    @ property
    def loss(self):
        return self._loss

    @ property
    def evaluation(self):
        return self._evaluation

    @ property
    def optimizer(self):
        return self._optimizer

    @ property
    def scheduler(self):
        return self._scheduler


class Config(_ConfigBase):
    """Configuration class

    Args:
        filename (str): path of config file
    """

    def __init__(self, filename):
        super(Config, self).__init__(filename=filename)

        # Parse ``parameters``
        self._work_dir = self.parameters.get("work_dir", None)
        if self.work_dir is None:
            task_name = ""
            for task in self.task:
                task_name += (task + "_")
            task_name = task_name.rstrip("_")
            self._work_dir = osp.join("./experiments", task_name)
        self._checkpoint_dir = self.parameters.get(
            "checkpoints_dir", osp.join(self.work_dir, "checkpoint"))

        # Checkpoint path
        self._checkpoint = self.parameters.get(
            "checkpoint", None)

    @ property
    def checkpoint(self):
        return self._checkpoint

    @ checkpoint.setter
    def checkpoint(self, filepath):
        self.parameters.update({"checkpoint": filepath})
        self._checkpoint = filepath

    @ property
    def checkpoints_dir(self):
        return self._checkpoint_dir

    @ property
    def work_dir(self):
        return self._work_dir

    @ work_dir.setter
    def work_dir(self, work_dir):
        self._work_dir = work_dir
        self.parameters.update({"work_dir": work_dir})

    @ property
    def pretty_text(self):
        dict_info = self.to_dict()
        return _pretty_text(dict_info)

    def to_dict(self):
        return self._config.to_dict()

    def save(self, filename):
        """Save config file as yaml file
        Args:
            filename (str): path to save config as yaml
        """
        save_yaml(filename, self.to_dict(), mode="w")
