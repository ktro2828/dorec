#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from dorec.core.ops.tensor import probalize
from dorec.core import LOSSES


@LOSSES.register()
class WeightedCrossEntropy(nn.Module):
    """Weighted Cross Entropy loss"""

    def __init__(
            self,
            ignore_label=-1,
            weight=None,
            align_corners=True,
            balance_weights=[1],
            use_softmax=True
    ):
        super().__init__()
        self.align_corners = align_corners
        self.balance_weights = balance_weights
        self.ignore_label = ignore_label
        self.weight = weight
        self.use_softmax = use_softmax

    def _forward(self, preds, targets):
        loss = F.binary_cross_entropy(
            preds, targets, weight=self.weight)
        return loss

    def forward(self, preds, targets):
        preds = probalize(preds)

        if isinstance(preds, torch.Tensor):
            preds = [preds]

        assert len(self.balance_weights) == len(
            preds), "length of balence_weights and preds must be same"

        return sum([w * self._forward(x, targets) for (w, x) in zip(self.balance_weights, preds)])


@LOSSES.register()
class CrossEntropy(nn.Module):
    """Cross entropy loss
    Args:
        weight
        use_softmax (bool)
    """

    def __init__(self, weight=None, use_softmax=True):
        super(CrossEntropy, self).__init__()
        self.weight = weight
        self.use_softmax = use_softmax
        self.xe = nn.CrossEntropyLoss(weight=self.weight)

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): (B, C, H, W)
            targets (torch.Tensor): (B, C, H, W)
        Returns:
            loss (torch.Tensor)
        """
        preds = probalize(preds, use_softmax=self.use_softmax)
        preds = torch.topk(preds, k=1, dim=1).indices.squeeze(1)
        targets = torch.topk(targets, k=1, dim=1).indicessqueeze(1)

        loss = self.xe(preds, targets)

        return loss


@LOSSES.register()
class OhemCrossEntropy(nn.Module):
    def __init__(
        self,
        ignore_label=-1,
        thresh=0.7,
        min_kept=100000,
        weight=None,
        align_corners=True,
        balance_weights=[1],
        use_softmax=True
    ):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thresh
        self.min_kept = max(1, min_kept)
        self.weight = weight
        self.align_corners = align_corners
        self.balance_weights = balance_weights
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_label, reduction="none")
        self.use_softmax = use_softmax

    def _ce_forward(self, preds, targets):
        loss = self.criterion(preds, targets)
        return loss

    def _ohem_forward(self, preds, targets):
        pixel_losses = self.criterion(preds, targets).contiguous().view(-1)
        mask = targets.contiguous().view(-1) != self.ignore_label

        tmp_targets = targets.clone()
        tmp_targets[tmp_targets == self.ignore_label] = 0
        pred = preds.gather(1, tmp_targets.unsqueeze(1))
        pred, ind = (
            pred.contiguous().view(-1, )[mask].contiguous().sort()
        )
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]

        return pixel_losses.mean()

    def forward(self, preds, targets):
        preds = probalize(preds, use_softmax=self.use_softmax)
        preds = torch.topk(preds, k=1, dim=1).indices.squeeze(1)
        targets = torch.topk(targets, k=1, dim=1).indices.squeeze(1)

        if isinstance(preds, torch.Tensor):
            preds = [preds]

        assert len(self.balance_weights) == len(
            preds), "length of balance_weights and preds must be same"

        functions = [self._ce_forward] * \
            (len(self.balance_weights) - 1) + [self._ohem_forward]

        return sum([w * func(x, targets) for (w, x, func) in zip(self.balance_weights, preds, functions)])
