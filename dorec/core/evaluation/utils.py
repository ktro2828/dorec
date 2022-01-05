#!/usr/bin/env python

import torch
import torch.nn.functional as F


def accuracy(pred, target):
    """
    Args:
        pred (torch.Tensor)
        target (torch.Tensor)
    Returns:
        score (float)
    """
    return float((pred == target).sum()) / target.numel()


def true_positive(pred, target, num_classes):
    """Compute the number of TP;true positive predictions
    Args:
        pred (torch.Tensor)
        target (torch.Tensor)
        num_classes (int)
    Returns:
        TP (torch.Tensor, dtype=LongTensor): (N,)
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target == i)).sum())

    return torch.tensor(out, device=pred.device)


def true_negative(pred, target, num_classes):
    """Compute the number of TN;true negative predictions
    Args:
        pred (torch.Tensor)
        target (torch.Tensor)
        num_classes (int)
    Returns:
        TN (torch.Tensor, dtype=LongTensor): (N,)
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target != i)).sum())

    return torch.tensor(out, device=pred.device)


def false_positive(pred, target, num_classes):
    """Compute the number of FP;false positive predictions
    Args:
        pred (torch.Tensor)
        target (torch.Tensor)
        num_classes (int)
    Returns:
        FP (torch.Tensor, dtype=LongTensor): (N,)
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target != i)).sum())

    return torch.tensor(out, device=pred.device)


def false_negative(pred, target, num_classes):
    """Compute the number of FN;false negative predictions
    Args:
        pred (torch.Tensor)
        target (torch.Tensor)
        num_classes (int)
    Returns:
        FN (torch.Tensor, dtype=LongTensor): (N,)
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target == i)).sum())

    return torch.tensor(out, device=pred.device)


def precision(pred, target, num_classes):
    """Compute the precision
    Args:
        pred (torch.Tensor)
        target (torch.Tensor)
        num_classes (int)
    Returns:
        precision (torch.Tensor): (N,)
    """
    tp = true_positive(pred, target, num_classes).float()
    fp = false_positive(pred, target, num_classes).float()

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out


def recall(pred, target, num_classes):
    """Compute the recall
    Args:
        pred (torch.Tensor)
        target (torch.Tensor)
        num_classes (int)
    Returns:
        recall (torch.Tensor): (N,)
    """
    tp = true_positive(pred, target, num_classes).float()
    fn = false_negative(pred, target, num_classes).float()

    out = tp / (tp + fn)
    out[torch.isnan(out)] = 0

    return out


def f_score(pred, target, num_classes, beta=1.0):
    """Computes the F score
        F = (1 + beta) x (precision x recall) / ((beta)^2 x precision + recall)
    Args:
        pred (torch.Tensor)
        target (torch.Tensor)
        num_classes (int)
        beta (float)
    Returns:
        recall (torch.Tesor): (N,)
    """
    prec = precision(pred, target, num_classes)
    rec = recall(pred, target, num_classes)

    score = (1 + beta) * (prec * rec) / ((beta**2) * prec + rec)
    score[torch.isnan(score)] = 0

    return score


def intersection_and_union(pred, target, num_classes):
    """Computes the mean intersection over union score of predictions
    Args:
        pred (torch.Tensor)
        target (torch.Tensor)
        num_classes (int)
    Returns:
        i, u (tuple(torch.Tensor)): (N,)
    """
    pred, target = F.one_hot(pred, num_classes), F.one_hot(target, num_classes)

    i = (pred & target).view(-1, num_classes).sum(dim=0)
    u = (pred | target).view(-1, num_classes).sum(dim=0)

    return i, u


def mean_iou(pred, target, num_classes, omitnans=False):
    """Computes the mean intersection and union score of predictions
    Args:
        pred (torch.Tensor)
        target (torch.Tensor)
        num_classes (int)
        omitnans (bool)
    Returns:
        miou (float)
    """
    i, u = intersection_and_union(pred, target, num_classes)
    iou = i.float() / u.float()

    if omitnans:
        miou = iou[~iou.isnan()].mean().item()
    else:
        iou[torch.isnan(iou)] = 1.
        miou = iou.mean(dim=-1).item()

    return miou


def mean_dice(pred, target, num_classes, omitnans=False):
    """Computes the mean dice score of predictions
    Args:
        pred (torch.Tensor)
        target (torch.Tensor)
        num_classes (int)
        omitnans (bool)
    Returns:
        mdice (float)
    """
    prec = precision(pred, target, num_classes)
    rec = recall(pred, target, num_classes)

    dice = 2 * prec * rec / (prec + rec)

    if omitnans:
        mdice = dice[~dice.isnan()].mean().item()
    else:
        dice[torch.isnan(dice)] = 1.
        mdice = dice.mean(dim=-1).item()

    return mdice


def mean_fscore(pred, target, num_classes, beta=1.0, omitnans=False):
    """Computes mean f1 score of predicisions
    Args:
        pred (torch.Tensor)
        target (torch.Tensor)
        num_classes (int)
        omitnans (bool)
    Returns:
        mf (float)
    """
    f = f_score(pred, target, num_classes, beta=beta)

    if omitnans:
        mf = f[~f.isnan()].mean().item()
    else:
        f[torch.isnan(f)] = 1.
        mf = f.mean(dim=-1).item()

    return mf


def mean_sq(pred, target, num_classes, omitnans=False):
    """Compute mean segmentation quarity of predictions
    Args:
        pred (torch.Tensor)
        target (torch.Tensor)
        num_classes (int)
        omitnans (bool)
    Returns:
        msq (float)
    """
    tp = true_positive(pred, target, num_classes)
    i, u = intersection_and_union(pred, target, num_classes)
    iou = i.float() / u.float()

    if omitnans:
        sq = iou / tp
        msq = sq[~sq.isnan()].mean().item()
    else:
        iou[torch.isnan(iou)] = 1.
        msq = (iou / tp).mean(dim=-1).item()

    return msq
