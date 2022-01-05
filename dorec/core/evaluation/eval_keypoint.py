#!/usr/bin/env python

import numpy as np
import torch

from dorec.core.ops import normalize_keypoint, get_max_pred


def keypoint_eval(
    preds,
    targets,
    mask,
    thresh=0.05,
    methods=("nme")
):
    """
    Compute the accuracy for each individual keypoint and the
    averaged accuracy across all keypoints from heatmaps with PCK, OKS or NME

    Args:
        preds (torch.Tensor): output heatmaps, in shape (B, K, H, W)
        targets (torch.Tensor): target heatmaps, in shape (B, K, H, W)
        mask (torch.Tensor): visibility mask, in shape (B, K)
        thresh (float, optional): threshold of PCK calculation(default: 0.05)
        methods (str, optional): evaluation method name, (pck, oks, nme)

    Returns:
        scores (dict[str, dict[str, any]]):
            `method`: [`all`: float, `each`: np.ndarray([K,])]
    """
    assert preds.shape == targets.shape
    if isinstance(methods, str):
        methods = (methods, )

    scores = {}
    for mth in methods:
        if mth == "pck":
            acc, avg_acc, num_valid = _calc_pck(preds, targets, mask, thresh)
        elif mth == "oks":
            acc, avg_acc, num_valid = _calc_oks(preds, targets, mask, thresh)
        elif mth == "nme":
            B, K, H, W = preds.shape

            pred_pos, _ = get_max_pred(preds)
            gt_pos, _ = get_max_pred(targets)
            pred_pos = normalize_keypoint(W, H, pred_pos)
            gt_pos = normalize_keypoint(W, H, gt_pos)

            acc = _calc_distances(pred_pos, gt_pos, mask)
            avg_acc = acc.mean().item()
        else:
            raise ValueError("unsupported method: {}".format(mth))

        scores.update({mth: {"all": avg_acc, "each": acc}})

    return scores


def _calc_distances(pred_pos, gt_pos, mask):
    """Calculate the normalized distances between predictions and targets

    Note:
        B: batch size
        K: number of keypoints

    Args:
        pred_pos (np.ndarray) normalized predicted position, in shape (B, K, 2)
        gt_pos (np.ndarray): normalized keypoints GT position, in shape (B, K, 2)
        mask (np.ndarray, torch.Tensor): in shape (B, K)
    Returns:
        distances (np.ndarray): total batch distances for each keypoints, in shape (K,)
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()

    mask = np.stack((mask, mask), axis=-1)
    pred_pos = mask * pred_pos
    gt_pos = mask * gt_pos

    distances = np.linalg.norm(pred_pos - gt_pos, ord=2, axis=2).mean(axis=0)

    return distances


def _calc_pck(output, target, mask, thresh=0.05):
    """Calculate PCK

    Args:
        output (torch.Tensor): output heatmaps, in shape (B, K, H, W)
        target (torch.Tensor): target heatmaps, in shape (B, K, H, W)
        mask (torch.Tensor): visibility mask, in shape (B, K)
        thresh (float, optional): threshold of PCK calculation(default: 0.05)
    Returns:
        tuple[np.ndarray[K], float, int]:
            - acc (np.ndarray[k])] accuracy of each keypoint
            - avg_acc (float): averaged accuracy across all keypoints
            - num_valid (int): number of valid keypoints
    """
    B, K, H, W = output.shape
    pred_pos, _ = get_max_pred(output)
    gt_pos, _ = get_max_pred(target)
    pred_pos = normalize_keypoint(W, H, pred_pos)
    gt_pos = normalize_keypoint(W, H, gt_pos)

    distances = _calc_distances(pred_pos, gt_pos, mask)

    acc = np.array([_distance_acc(d, thresh) for d in distances])
    valid_acc = acc[acc >= 0]
    num_valid = len(valid_acc)
    avg_acc = valid_acc.mean(axis=1) if num_valid > 0 else 0

    return acc, avg_acc, num_valid


def _calc_oks(output, target, mask, s=1.0, k=1.0, thresh=0.05):
    """Calculate OKS
    Args:
        output (torch.Tensor): output heatmaps, in shape (B, K, H, W)
        target (torch.Tensor): target heatmaps, in shape (B, K, H, W)
        mask (torch.Tensor): visibility mask, in shape (B, K)
        s (float): scale, the square root of the object segment area
        k (np.ndarray, float): per-kerypoint constant that controlls fall off
        thresh (float): threshold value
    """
    B, K, H, W = output.shape
    pred_pos, _ = get_max_pred(output)
    gt_pos, _ = get_max_pred(target)
    pred_pos = normalize_keypoint(W, H, pred_pos)
    gt_pos = normalize_keypoint(W, H, gt_pos)

    if isinstance(k, np.ndarray) and k.shape[0] > 1:
        assert k.shape[0] == K

    distances = _calc_distances(pred_pos, gt_pos, mask)
    d = np.array([_distance_acc(d, thresh) for d in distances])

    oks = np.exp(-d**2 / (2 * s**2 * k**2))
    oks = oks[oks >= 0]
    num_valid = len(oks)
    avg_oks = oks.mean() if num_valid > 0 else 0

    return oks, avg_oks, num_valid


def _distance_acc(distances, thresh=0.05):
    """Return the percentage below the distance threshold, while ignoring distances values with -1
    Args:
        distances (np.ndarray): in shape (K,)
        thresh (float): threhold value
    Returns:
        acc (np.ndarray): in shape (K,)
    """
    distance_valid = (distances != -1)
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thresh).sum() / num_distance_valid
    return -1
