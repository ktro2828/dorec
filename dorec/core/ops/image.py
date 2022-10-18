#!/usr/bin/env python

import cv2
import numpy as np
import torch

COLORMAP = (
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 255),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 255),
    (0, 0, 0),
)


def imread(img, dim_ord="chw"):
    """Read image

    Args:
         img (str, np.ndarray, torch.Tensor): image path or tensor
         dim_ord (str): order of dimension ["chw", "hwc"]
         is_normalized (bool, optional): indicates whether input image is normalized

    Returns:
         img (np.ndarray)
    """
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(img, torch.Tensor):
        assert len(img.shape) == 3, "expected CxHxW or HxWxC shape"
        img = img.cpu().detach().numpy()
    elif isinstance(img, np.ndarray):
        assert len(img.shape) == 3, "expected CxHxW or HxWxC shape"
        pass
    else:
        raise TypeError("unexpected type of img: {}".format(type(img)))

    img = _transpose(img, dim_ord=dim_ord)
    return img


def _transpose(img, dim_ord, copy=True):
    """Transpose image
    Args:
        img (np.ndarray): input image
        dim_ord (str): dimension order of input image
        copy (bool, optional): indicates whether return copied object
    """
    if dim_ord == "chw":
        img = img.transpose(1, 2, 0)
    elif dim_ord == "hwc":
        pass
    else:
        raise ValueError(
            "expected dim_ord=[`chw`, `hwc`], but got {}".format(dim_ord))

    if copy:
        return img.copy()
    return img


def edge_filter(img, method, **kwargs):
    if len(img.shape) == 3:
        dims = img.shape[-1]
        if dims > 1:
            ret = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    if method == "canny":
        ret = canny_edge(ret, **kwargs)
    elif method == "sobel":
        ret = sobel_edge(ret, **kwargs)
    elif method == "laplacian":
        ret = laplacian_edge(ret, **kwargs)
    elif method == "pyramid":
        ret = pyramid_edge(ret, **kwargs)
    else:
        raise ValueError("unexpected method: {}".format(method))
    ret = np.expand_dims(ret, axis=-1)
    ret = np.tile(ret, (1, 1, dims))

    return ret


def canny_edge(img, min_val=100, max_val=200, **kwargs):
    """Cannny edge
    Args:
        img (np.ndarray)
        min_val (float, optional)
        max_val (float, optional)
    Returns:
        edges (np.ndarray)
    """
    edges = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img, min_val, max_val)

    return edges


def laplacian_edge(img, dtype=cv2.CV_64F, **kwargs):
    """Laplacian edge"""
    edges = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
    edges = cv2.Laplacian(edges, dtype)
    return edges


def sobel_edge(img, ksize=5, dtype=cv2.CV_64F, mode="gradient", **kwargs):
    """Sobel edge"""
    assert mode in ("x", "y", "gradient")
    edges = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
    if mode == "x":
        return cv2.Sobel(edges, dtype, 1, 0, ksize=ksize)
    elif mode == "y":
        return cv2.Sobel(edges, dtype, 0, 1, ksize=ksize)
    else:
        dx = cv2.Sobel(edges, dtype, 1, 0, ksize=ksize)
        dy = cv2.Sobel(edges, dtype, 0, 1, ksize=ksize)
        return np.sqrt(dx ** 2 + dy ** 2)


def pyramid_edge(img, max_level=3, use_level=0, min_val=50, max_val=300, **kwargs):
    """Pyramid edge"""
    h, w = img.shape[:2]
    tmp = img.copy()
    ret = [canny_edge(img, min_val, max_val)]
    for idx in range(max_level - 1):
        tmp = cv2.pyrDown(tmp)
        ret.append(canny_edge(tmp, min_val, max_val))
    return cv2.resize(ret[use_level], (w, h))


def normalize_depth(depth, min_val=None, max_val=None):
    """Normalize depth image
    Args:
        depth (np.ndarray, torch.Tensor)
        min_val (float, optional)
        max_val (float, optional)
    Returns:
        depth (np.ndarray, torch.Tensor)
    """
    if isinstance(depth, np.ndarray):
        min_val = np.nanmin(depth) if min_val is None else min_val
        max_val = np.nanmax(depth) if max_val is None else max_val
        depth = remove_nan(depth)
        depth = (depth - min_val) / (max_val - min_val)
        depth = np.clip(depth, 0, 1)
    else:
        min_val = depth.min().item() if min_val is None else min_val
        max_val = depth.max().item() if max_val is None else max_val
        depth = (depth - min_val) / (max_val - min_val)

    return depth


def unnormalize_depth(depth, min_val, max_val):
    """Unnormalize depth image
    Args:
        depth (np.ndarray, torch.Tensor)
        min_val (float)
        max_val (float)
    Returns:
        depth (np.ndarray, torch.Tensor)
    """
    if isinstance(depth, np.ndarray):
        depth = remove_nan(depth)
    depth = depth * (max_val - max_val) + min_val

    return depth


def remove_nan(arr):
    """Remove nan value
    Args:
        arr (np.ndarray)
    Returns:
        arr (np.ndarray)
    """
    nan_mask = np.isnan(arr)
    arr[nan_mask] = 0

    return arr


def depth2mask(depth):
    """Create mask image from depth
    Args:
        depth (np.ndarray)
    Returns:
        mask (np.ndarray)
    """
    mask = np.zeros_like(depth, np.uint8)
    mask[depth > 0] = 255
    return mask


def erase_depth_edge(depth, edge, max_sample=100, erase_val=0):
    """Erase depth edge
    Args:
        depth (np.ndarray): depth image
        edge (np.ndarray): edge image
        max_sample (int): maximum number of indices to be erased
        erase_val (int): erasing value
    Returns:
        depth (np.ndarray)
    """
    assert edge.max() == 255
    edge, _ = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)
    edges_idx = np.array(zip(*np.where(edge == 255)))
    length = len(edges_idx)

    num_samples = np.random.randint(0, min(length, max_sample))
    sample_idx = np.unique(np.random.randint(0, length, num_samples))
    samples = edges_idx[sample_idx]
    depth[samples[:, 0], samples[:, 1]] = erase_val

    return depth
