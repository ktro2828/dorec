#!/usr/bin/env python

import torch
import torch.nn as nn

from dorec.core import LOSSES


@LOSSES.register()
class KeypointLoss(nn.Module):
    """Basic Loss function for keypoint detection"""

    def __init__(self, loss_weight=1.0, **kwargs):
        super(KeypointLoss, self).__init__()
        self.loss_weight = loss_weight

        self.name = kwargs.get("name")

    def forward(self, pred_heatmaps, gt_heatmaps, keypoint_vis=None):
        B, N, H, W = pred_heatmaps.shape

        if keypoint_vis is None:
            keypoint_vis = torch.ones(B, N)

        device = pred_heatmaps.device
        pred_heatmaps = pred_heatmaps.reshape(B * N, -1)
        gt_heatmaps = gt_heatmaps.reshape(B * N, -1)
        vis_masks = torch.cat([keypoint_vis.reshape(
            B * N, -1)] * H * W, dim=1).float().to(device)

        loss = torch.pow(vis_masks * (pred_heatmaps - gt_heatmaps), 2).mean()

        return loss * self.loss_weight


@LOSSES.register()
class KeypointMSELoss(nn.Module):
    """Loss function with MSE"""

    def __init__(self, loss_weight=1.0, **kwargs):
        super(KeypointMSELoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.loss_weight = loss_weight

        self.name = kwargs.get("name")

    def forward(self, pred_heatmaps, gt_heatmaps, keypoint_vis=None):
        B, N, H, W = pred_heatmaps.shape

        pred_heatmaps = pred_heatmaps.reshape(B, N, -1).split(1, 1)
        gt_heatmaps = gt_heatmaps.reshape(B, N, -1).split(1, 1)

        loss = 0.0

        for idx in range(N):
            pred_heatmap = pred_heatmaps[idx].squeeze(1)
            gt_heatmap = gt_heatmaps[idx].squeeze(1)
            vis_mask = keypoint_vis.reshape(B, N, 1)
            if keypoint_vis is not None:
                loss += self.criterion(pred_heatmap *
                                       vis_mask[:, idx], gt_heatmap * vis_mask[:, idx])
            else:
                loss += self.criterion(pred_heatmap, gt_heatmap)

        return loss / N * self.loss_weight
