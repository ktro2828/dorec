#!/usr/bin/env python

import datetime
import os.path as osp

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dorec.datasets import build_dataset
from dorec.core import Config
from dorec.core.utils import save_yaml, makedirs, get_logger, AverageMeter

from .utils import worker_init_fn, parse_device, DataParallel
logger = get_logger(modname=__name__)


class RunnerBase(object):
    """Base class for running train, test or inference
    Args:
        config (dorec.utils.Config): loaded config from /configs/config.yml
        is_test (bool): indicates whether do test
    """

    def __init__(self, config, is_test):
        assert isinstance(config, Config)
        assert isinstance(is_test, bool)
        self._config = config
        self._is_test = is_test

        # Reset parameters
        self._init_parameters()
        self._reset_phase_parameters()

        # Set working directory
        self._set_work_dir()

        # Summary writer
        self.writer = SummaryWriter(log_dir=self.work_dir)

        # Display and save parameters
        logger.info(self.config.pretty_text)
        cfg_save_path = osp.join(self.work_dir, self.config_name + ".yml")
        save_yaml(cfg_save_path, self.config.to_dict(), mode="w")
        logger.info("Config file is saved to: {}".format(cfg_save_path))

        # Global epoch
        self.epoch = 0
        self.viz_cnt = 0

    def _init_parameters(self):
        """Initialize basic parameters from config"""
        self._config_name = self.config.name
        self._checkpoint = self.config.checkpoint
        self._work_dir = self.config.work_dir

        self._task = self.config.task
        self._parameters_cfg = self.config.parameters
        self._model_cfg = self.config.model
        self._dataset_cfg = self.config.dataset
        self._optimizer_cfg = self.config.optimizer
        self._scheduler_cfg = self.config.scheduler
        self._loss_cfg = self.config.loss
        self._evaluation_cfg = self.config.evaluation

    def _set_work_dir(self):
        date_info = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._work_dir = osp.join(
            self.config.work_dir, self.config_name + "_" + date_info)
        self._checkpoint_dir = osp.join(self.work_dir, "checkpoints")
        self._vizdir = osp.join(self.work_dir, "viz")

        makedirs(self.work_dir, exist_ok=True)
        makedirs(self.checkpoint_dir, exist_ok=True)
        makedirs(self.vizdir, exist_ok=True)
        logger.info("Working directory is: {}".format(self.work_dir))

    def _reset_phase_parameters(self):
        """if change phase, reset parameters depends on phase"""
        if self.is_test:
            phase = "test"
            logger.info("****Test mode****")
            self._batch_size = int(self._dataset_cfg.test.get("batch_size", 1))
            self._device = parse_device(
                self._parameters_cfg.test.device,
                self._parameters_cfg.test.gpu_ids)
            self._max_epoch = int(
                self._parameters_cfg.test.get("max_epoch", 1))
        else:
            phase = "train"
            logger.info("****Train mode****")
            self._batch_size = int(
                self._dataset_cfg.train.get("batch_size", 1))
            self._device = parse_device(
                self._parameters_cfg.train.device,
                self._parameters_cfg.train.gpu_ids)
            self._max_epoch = int(
                self._parameters_cfg.train.get("max_epoch", 1))

        self.epoch = 0
        self.viz_cnt = 0

        logger.info("Parameters is reseted for: {}".format(phase))

    @property
    def config(self):
        return self._config

    @property
    def config_name(self):
        return self._config_name

    @property
    def is_test(self):
        return self._is_test

    @is_test.setter
    def is_test(self, b):
        assert isinstance(b, bool), "expected bool, but got {}".format(type(b))
        self._is_test = b
        phase = "test" if b else "train"
        logger.info("Switched phase to: {}".format(phase))
        self._reset_phase_parameters()

    @property
    def task(self):
        return self._task

    @property
    def model_cfg(self):
        return self._model_cfg

    @property
    def dataset_cfg(self):
        return self._dataset_cfg

    @property
    def parameters_cfg(self):
        return self._parameters

    @property
    def optimizer_cfg(self):
        return self._optimizer_cfg

    @property
    def scheduler_cfg(self):
        return self._scheduler_cfg

    @property
    def loss_cfg(self):
        return self._loss_cfg

    @property
    def evaluation_cfg(self):
        return self._evaluation_cfg

    @property
    def input_type(self):
        return self._dataset_cfg.input_type

    @property
    def use_dims(self):
        return self._dataset_cfg.use_dims

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_epoch(self):
        return self._max_epoch

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, d):
        self._device = d

    @property
    def work_dir(self):
        return self._work_dir

    @property
    def checkpoint(self):
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, p):
        assert isinstance(p, str), "expected str, but got {}".format(type(p))
        self._checkpoint = p

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def vizdir(self):
        return self._vizdir

    def train(self):
        """Execute train"""
        raise NotImplementedError

    def test(self):
        """Exectute test"""
        raise NotImplementedError

    def _evaluate(self, model, dataloader, avg_meters, optimizer=None):
        """Execute evaluation for validation or test"""
        raise NotImplementedError

    def inference(self, root):
        """Execute inference"""
        raise NotImplementedError

    def visualize(self, inputs, outputs, max_try=1, vis_random=True):
        """visualize model outputs result for inputs image"""
        raise NotImplementedError

    def show_data(self, max_try=10):
        """show reuslt of data augmentation"""
        raise NotImplementedError

    def load_data(self):
        """Load dataset
        Returns:
            - train_dataloader, val_dataloader (tuple[DataLoader]): if is_test=False
            - test_dataloader (torch.utils.data.DataLoader): if is_test=True
        """
        dataset_cfg = self.dataset_cfg.copy()
        train_dataset_cfg = dataset_cfg.pop("train")
        test_dataset_cfg = dataset_cfg.pop("test")

        if self.dataset_cfg.get("val") is not None:
            val_dataset_cfg = dataset_cfg.pop("val")
            val_dataset_cfg.update(dataset_cfg)
            val_dataset_cfg.task = self.task
        else:
            val_dataset_cfg = None

        train_dataset_cfg.update(dataset_cfg)
        train_dataset_cfg.task = self.task
        test_dataset_cfg.update(dataset_cfg)
        test_dataset_cfg.task = self.task

        if self.is_test:
            test_dataset = build_dataset(test_dataset_cfg)
            logger.info("Loaded dataset Test: {}".format(len(test_dataset)))

            assert len(test_dataset) >= self.batch_size, \
                "batch size must be smaller than total number of data"

            test_dataloader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=worker_init_fn
            )
            return test_dataloader

        if val_dataset_cfg is not None:
            train_dataset = build_dataset(train_dataset_cfg)
            val_dataset = build_dataset(val_dataset_cfg)
        else:
            # Split train : val = 8 : 2
            dataset = build_dataset(train_dataset_cfg)
            num_data = len(dataset)
            train_size = int(num_data * 0.8)
            val_size = num_data - train_size
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size])

        if len(train_dataset) <= self.batch_size:
            raise ValueError(
                "batch size must be smaller than total number of data")
        if len(val_dataset) <= (self.batch_size + 3) // 4:
            raise ValueError(
                "batch size must be smaller than total number of data")

        logger.info("Loaded dataset Train: {}".format(len(train_dataset)))
        logger.info("Loaded dataset Validation: {}".format(len(val_dataset)))

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=(self.batch_size + 3) // 4,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn
        )
        return train_dataloader, val_dataloader

    def load_checkpoint(self, model, optimizer=None, checkpoint=None):
        """Load checkpoint from chekpoint path
        if checkpoint_path is None, only load model on GPU
        Args:
            model (torch.nn.Module)
            optimizer (torch.optim.Optimizer, optional)
            chekpoint (str, optional)
            init (bool, optional)
        Returns:
            model (torch.nn.Module)
            optimizer (Optional[torch.optim.Optimizer])
        """
        # Load checkpoint
        if checkpoint is not None:
            logger.info("model is loaded with: {}".format((checkpoint)))
            state_dict = torch.load(checkpoint, map_location=self.device)
            if "model" in state_dict.keys():
                model.load_state_dict(state_dict["model"])
            else:
                model.load_state_dict()
            if optimizer is not None:
                optimizer.load_state_dict(state_dict["optimizer"])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        state[k] = v.to(self.device)

        # Apply model parralel
        if torch.cuda.device_count() > 1 and not self.device == torch.device("cpu"):
            model_name = model.name
            model = DataParallel(model)
            model.name = model_name

        model = model.to(self.device)

        if self.is_test:
            model.eval()
        else:
            model.train()

        return model, optimizer

    def save_checkpoint(self, model, checkpoint_path, optimizer=None):
        """save model to checkpoint path as .pth
        Args:
            model (torch.nn.Module)
            optimizer (torch.optim.Optimizer)
            checkpoint_path (str): Path to save checkpoint
        """
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        optimizer_state = optimizer.state_dict() if optimizer is not None else None

        state = {
            "model": model_state,
            "optimizer": optimizer_state
        }

        logger.info("Saving model to: {}".format(checkpoint_path))
        torch.save(state, checkpoint_path)

    def set_avg_meters(self):
        """Set AverageMeter
        Returns:
            avg_meters (dict[str, any])
        """
        avg_meters = {"loss": dict(), "score": dict()}
        avg_meters["loss"]["total"] = AverageMeter()
        for tsk in self.task:
            avg_meters["loss"][tsk] = AverageMeter()
            avg_meters["score"][tsk] = dict()
            for mth in list(self.evaluation_cfg[tsk].methods):
                avg_meters["score"][tsk][mth] = AverageMeter()

        return avg_meters

    def _reset_avg_meters(self, avg_meters):
        """
        Args:
            avg_meters (dict[str, any])
        """
        for tsk in self.task:
            avg_meters["loss"][tsk].reset()
            for mth in list(self.evaluation_cfg[tsk].methods):
                avg_meters["score"][tsk][mth].reset()

    def _record_results(self, avg_meters, losses, scores):
        """Record loss and score to AverageMeter()
        Args:
            avg_meters (dict[str, AverageMeters])
            losses (dict[str, torch.Tensor])
            scores (dict[str, float])
        """
        avg_meters["loss"]["total"].update(
            losses["total"].item(), self.batch_size)

        for tsk in self.task:
            avg_meters["loss"][tsk].update(
                losses[tsk].item(), self.batch_size)
            score_avg_meters = avg_meters["score"][tsk]
            score_dict = scores[tsk]
            for mth in list(self.evaluation_cfg[tsk].methods):
                score_avg_meters[mth].update(
                    score_dict[mth], self.batch_size)

    def _logging_results(self, avg_meters):
        """Logging results
        Args:
            avg_meters (dict[str, AverageMeter])
        """
        loss_msg = "total: {}\n".format(avg_meters["loss"]["total"].val)
        score_msg = ""
        for tsk in self.task:
            loss_msg += "{}: {}\n".format(
                tsk, avg_meters["loss"][tsk].val)
            score_msg += "{}:\n".format(tsk)
            for mth in list(self.evaluation_cfg[tsk].methods):
                score_msg += "  [{}]: {}".format(
                    mth, avg_meters["score"][tsk][mth].val)
            score_msg += "\n"

        msg = r"""
|Epoch|: {}
|Loss|:
{}
|Score|:
{}
        """.format(self.epoch, loss_msg, score_msg)
        logger.info(msg)

    def _update_summary_writer(self, train_avg_meters, val_avg_meters, epoch):
        """Update summary writer with AverageMeters()
        Args:
            train_avg_meters (dict[dict[str, AverageMeters()]])
            val_avg_meters (dict[dict[str, AverageMeters()]])
            epoch (int)
        """
        self.writer.add_scalars(
            "toal loss",
            {"train": train_avg_meters["loss"]["total"].val,
             "val": val_avg_meters["loss"]["total"].val}, epoch
        )

        for tsk in self.task:
            self.writer.add_scalars(
                tsk + " loss",
                {"train": train_avg_meters["loss"][tsk].val,
                 "val": val_avg_meters["loss"][tsk].val}, epoch
            )
            self.writer.add_scalars(
                tsk + " score",
                {"train": train_avg_meters["score"][tsk].val,
                 "val": val_avg_meters["score"][tsk].val}, epoch
            )
