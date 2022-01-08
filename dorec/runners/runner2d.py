#!/usr/bin/env python

import os.path as osp
import random

import torch
from tqdm import tqdm

from dorec.core.utils import build_optimizer, build_scheduler
from dorec.core.evaluation import do_evaluate
from dorec.core.visualization import do_visualize
from dorec.models import build_model
from dorec.losses import build_loss

from .base import RunnerBase


class Runner2d(RunnerBase):
    """Runner for 2D inputs and outputs task
    Args:
        config (dorec.core.utils.Config)
        is_test (bool)
    """

    def __init__(self, config, is_test):
        super(Runner2d, self).__init__(config, is_test)
        # Loss function
        self.criterion = build_loss(self.loss_cfg)

    def train(self):
        if self.is_test:
            raise ValueError("must be set ``is_test=False``")
        # Model, optimizer and scheduler
        model = build_model(self.model_cfg, task=self.task, assign_task=True)
        optimizer = build_optimizer(self.optimizer_cfg, model.parameters())
        model, optimizer = self.load_checkpoint(
            model,
            optimizer=optimizer,
            checkpoint=self.checkpoint
        )
        scheduler = build_scheduler(self.scheduler_cfg, optimizer)

        # Dataset
        train_dataloader, val_dataloader = self.load_data()

        # Result recorder
        train_avg_meters = self.set_avg_meters()
        val_avg_meters = self.set_avg_meters()

        for epoch in range(1, self.max_epoch + 1):
            self.epoch = epoch
            model.train()
            self._reset_avg_meters(train_avg_meters)

            for sample in tqdm(train_dataloader):
                optimizer.zero_grad()

                losses, _ = self._forward_one_epoch(
                    sample, model, train_avg_meters)

                losses["total"].backward()
                optimizer.step()

            # --- VALIDATION ---
            self._evaluate(model, val_dataloader,
                           val_avg_meters, optimizer=optimizer)

            if scheduler.name == "ReduceLROnPlateau":
                scheduler.step(val_avg_meters["losses"]["loss"].val)
            else:
                scheduler.step()

    def _forward_one_epoch(self, sample, model, avg_meters):
        """Train method for one epoch
        Args:
            sample (dict[str, any])
            model (nn.Module)
            train_avg_meters (dict[str, any])
        """
        images = sample["inputs"].to(self.device)
        targets = sample["targets"]
        for key, item in targets.items():
            targets[key] = item.to(self.device)

        outputs = model(images)
        losses = self.criterion(outputs, targets)
        scores = do_evaluate(self.evaluation_cfg, outputs, targets)

        self._record_results(avg_meters, losses, scores)

        return losses, scores

    def test(self):
        """Test method"""
        if not self.is_test:
            raise ValueError("must be set ``is_test=True``")

        model = build_model(self.model_cfg, task=self.task, assign_task=True)
        model, _ = self.load_checkpoint(model, checkpoint=self.checkpoint)

        # Dataset
        test_dataloader = self.load_data()

        # Result recorder
        test_avg_meters = self.set_avg_meters()

        self._evaluate(model, test_dataloader, test_avg_meters)

    def _evaluate(self, model, dataloader, avg_meters, optimizer=None):
        """Evaluation method, which used in test/val phase
        Args:
            model (torch.nn.Module)
            dataloader (torch.data.DataLoader)
            avg_meters (dict)
            optimizer (torch.nn.Optimizer)
        """
        model.eval()
        with torch.no_grad():
            # Reset AverageMeter
            self._reset_avg_meters(avg_meters)

            for sample in tqdm(dataloader):
                images = sample["inputs"].to(self.device)
                targets = sample["targets"]
                for key, item in targets.items():
                    targets[key] = item.to(self.device)

                outputs = model(images)
                losses = self.criterion(outputs, targets)
                scores = do_evaluate(self.evaluation_cfg, outputs, targets)

                self._record_results(avg_meters, losses, scores)

            # Visualize result
            self.visualize(images, outputs)

            # Logging evaluate results
            self._logging_results(avg_meters)

            # Save checkpoint
            checkpoint_path = osp.join(
                self.checkpoint_dir, model.name + "{}.pth".format(self.epoch))
            self.save_checkpoint(model, checkpoint_path, optimizer=optimizer)

    def visualize(self, imgs, maps, max_try=1, vis_random=True):
        """
        Args:
            imgs (torch.Tensor): (B, C, H, W): Input images
            maps (dict[str, torch.Tensor]): (B, N, H, W): Ouputs or GTs
            num_try (int, optional): number of try
            vis_random (bool, optional): whether visualize randomly
        """
        max_try = min(len(imgs), max_try)
        if self.use_dims >= 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        for n in range(max_try):
            idx = random.randint(0, max_try - 1) if vis_random else n
            if self.use_dims >= 3:
                one_img = imgs[idx][:3, :, :]
                one_img = 255.0 * \
                    (one_img * std[:, None, None] + mean[:, None, None])
            else:
                one_img = imgs[idx][0, :, :]

            for key, item in maps.items():
                maps[key] = item[idx]

            self.viz_cnt += 1
            filepath = osp.join(self.vizdir, str(self.viz_cnt) + ".jpg")
            do_visualize(maps, imgs=one_img, filepath=filepath)

    def show_data(self, max_try=10):
        """Show result of data augmentation
        Args:
            max_try (int): number of maximum trial (default: 10)
        """
        if self.is_test:
            dataloader = self.load_data()
        else:
            dataloader, _ = self.load_data()

        for itr, sample in enumerate(tqdm(dataloader)):
            images = sample["inputs"].to(self.device)
            targets = sample["targets"]
            for key, item in targets.items():
                targets[key] = item.to(self.device)

            self.visualize(images, targets)

            if itr >= max_try:
                break
