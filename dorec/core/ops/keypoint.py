#!/usr/bin/env python

import numpy as np
import torch


def load_keypoint(data, keypoint_names):
    """Load keypoint data as array

    Args:
        data (dict)
        keypoint_names (list[str])

    Returns:
        pos (np.ndarray): in shape (K, 2)
        vis (np.ndarray): in shape (K, 1)
        in_pic (np.ndarray): in shape (K, 1)
    """
    pos = []
    vis = []
    in_pic = []
    for name in keypoint_names:
        data_i = data[name]
        pos.append(np.array(data_i["pos"]))
        vis.append(data_i["vis"])
        in_pic.append(data_i["in_pic"])
    pos = np.array(pos)
    vis = np.array(vis)
    in_pic = np.array(in_pic)

    return pos, vis, in_pic


def gaussian_map(image_w, image_h, center_x, center_y, R):
    """Generate gaussian map
    Args:
        image_w (int): image width
        image_h (int): image height
        center_x (float): center x position
        center_y (float): center y position
        R (float): value equal to variance
    Returns:
        gauss (np.ndarray): in shape (H, W)
    """
    X, Y = np.meshgrid(np.linspace(0, image_w - 1, image_w) - center_x,
                       np.linspace(0, image_h - 1, image_h) - center_y)

    gauss = np.exp(-0.5 * (X ** 2 + Y ** 2) / R)

    return gauss


def gen_heatmap(image_w, image_h, keypoint_in_pic, keypoint_pos, R):
    """Returns heatmap of keypoints

    Args:
        image_w(int): width of image
        image_h(int): height of image
        keypoint_in_pic(0 or 1): if keypoint in picture 1, else 0, in shape (K)
        keypoint_pos(sequence): keypoint position in picture, in shape (K, 2)
        R(float): value corresponding to variance

    Returns:
        np.ndarray: generated heatmap, in shape (K, image_h, image_w)
    """
    ret = []
    assert len(keypoint_in_pic) == len(
        keypoint_pos), "number of keypoints must be same"
    for in_pic, (pos_x, pos_y) in zip(keypoint_in_pic, keypoint_pos):
        if in_pic == 0:
            ret.append(np.zeros((image_h, image_w)))
        else:
            channel_map = gaussian_map(
                image_w, image_h, float(pos_x), float(pos_y), R)
            ret.append(channel_map)
    return np.stack(ret, axis=0).astype(np.float32)


def normalize_keypoint(image_w, image_h, keypoint_pos):
    """Normalize keypoints position
    Args:
        image_w (int)
        image_h (int)
        keypoint_pos (np.ndarray, torch.Tensor): in shape (N, 2)
    Returns:
        np.ndarray, torch.Tensor: normalized keypoint position
    """
    normalized = np.zeros_like(keypoint_pos)
    normalized[:, 0] = keypoint_pos[:, 0] / float(image_w)
    normalized[:, 1] = keypoint_pos[:, 1] / float(image_h)
    return normalized


def unnormalize_keypoint(image_w, image_h, keypoint_pos):
    """Unnormalize keypoints position
    Args:
        image_w (int)
        image_h (int)
        keypoint_pos (np.ndarray, torch.Tensor): in shape (N, 2)
    Returns:
        np.ndarray, torch.Tensor: unnormalized keypoint position
    """
    unnormalized = np.zeros_like(keypoint_pos)
    unnormalized[:, 0] = keypoint_pos[:, 0] * float(image_w)
    unnormalized[:, 1] = keypoint_pos[:, 1] * float(image_h)
    return unnormalized


def check_keypoint(image_w, image_h, keypoint_pos, keypoint_vis, keypoint_in_pic):
    """Check keypoints visibility and whether keypoints in picture

    Args:
        image_w (int): width of image
        image_h (int): height of image
        keypoint_pos (np.ndarray, torch.Tensor): keypoint xy-position, in shape (K, 2)
        keypoint_vis (np.ndarray, torch.Tensor): keypoint visibility, in shape (K, 1)
        keypoint_in_pic (np.ndarray, torch.Tensor): whether keypoint in picture, in shape (K, 1)

    Returns:
        keypoint_pos
        keypoint_vis
        keypoint_in_pic
    """
    for i, vis in enumerate(keypoint_vis):
        if (
            (keypoint_pos[i, 0] < 0)
            or (keypoint_pos[i, 0] >= image_w)
            or (keypoint_pos[i, 1] < 0)
            or (keypoint_pos[i, 1] >= image_h)
        ):
            keypoint_vis[i] = 0
            keypoint_in_pic[i] = 0
    for i, in_pic in enumerate(keypoint_in_pic):
        if in_pic == 0:
            keypoint_pos[i, :] = 0
    return keypoint_pos, keypoint_vis, keypoint_in_pic


def get_max_pred(heatmaps, return_type="numpy"):
    """Returns keypoint predictions from score maps.

    Args:
        heatmaps (torch.Tensor, np.ndarray): model predicted heatmaps,
            in shape BxKxHxW or KxHxW
        return_vals (bool, optional): indicates whether return max values with preds

    Returns:
        preds (np.ndarry, torch.Tensor): in shape BxKx2 or Kx2
        maxvals (np.ndarray, torch.Tensor): in shape BxK or K
    """
    assert return_type in ("numpy", "torch")

    shape_len = len(heatmaps.shape)
    K, H, W = heatmaps.shape[-3::]

    if isinstance(heatmaps, torch.Tensor):
        device = heatmaps.device
        heatmaps_arr = heatmaps.cpu().detach().numpy().reshape((-1, K, H * W))
    elif isinstance(heatmaps, np.ndarray):
        device = torch.device("cpu")
        heatmaps_arr = heatmaps.reshape((-1, K, H * W))

    idx = np.argmax(heatmaps_arr, 2).reshape((-1, K, 1))
    maxvals = np.amax(heatmaps_arr, 2).reshape((-1, K, 1))
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)

    if shape_len == 3:
        preds = preds.reshape(K, 2)
        maxvals = maxvals.reshape(K)

    if return_type == "torch":
        preds = torch.from_numpy(preds).to(device)
        maxvals = torch.from_numpy(maxvals).to(device)

    return preds, maxvals
