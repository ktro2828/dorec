#!/usr/bin/env python

import torch


def probalize(tensor, dim=1, use_softmax=True):
    """Convert input tensor to probability

    Args:
        tensor (torch.Tensor): in shape BxCxHxW or CxHxW
        dim (int, optional): target dimension numnber
        use_softmax (bool, optional): if false use torch.log_softmax
    """
    device = tensor.device
    _sum = tensor.sum(dim=dim)

    if len(tensor.shape) == 4:
        # check whether inputs was probalized
        B, C, H, W = tensor.shape
        _ones = torch.ones(B, H, W).to(device)
    elif len(tensor.shape) == 3:
        C, H, W = tensor.shape
        _ones = torch.ones(H, W).to(device)
    else:
        raise ValueError("expected shape is BxCxHxW ot CxHxW")

    if torch.allclose(_sum, _ones):
        return tensor

    if C <= 2:
        # sigmoid
        x = torch.sigmoid(tensor)
    else:
        if use_softmax:
            # softmax
            x = torch.softmax(tensor, dim=dim)
        else:
            # log_softmax
            x = torch.log_softmax(tensor, dim=dim)

    return x


def to_index(src):
    """Convert BxCxHxW mask image to BxHxW index image
    Args:
        src (torch.Tensor): in shape (B, C, H, W)
    Returns:
        dst (torch.Tensor): in shape (B, H, W)
    """
    B, C, H, W = src.shape
    device = src.device
    dst = torch.zeros(B, H, W).to(device)
    for c in range(C):
        dst[src[:, c, :, :] == 1] = c
    dst = dst.long()

    return dst


def to_mask(src, idx=0, th=0.8):
    """Convert Bx2xHxW tensor to Bx1xHxW mask
    Args:
        src (torch.Tensor): in shape (B, 2, H, W)
        idx (int, optional): target index to be 1
        th (float, optional): threshold
    Returns:
        mask (torch.Tensor): in shape (B, 1, H, W)
    """
    B, C, H, W = src.shape
    assert C == 2, "expected 2 channels tensor, but got {}".format(C)
    device = src.device
    # assert max(1.0, th) == 1 and min(0.0, th) == 0
    mask = torch.zeros(B, 1, H, W).to(device)
    mask_idx = torch.argmax(src, dim=1, keepdim=True)
    mask[mask_idx == idx] = 1
    return mask


def to_onehot(tensor, num_classes):
    """One-hot encondig

    Args:
        tensor (torch.Tensor)
        num_classes (int)

    Returns:
        onehot (torch.Tensor)
    """
    device = tensor.device

    onehot = torch.eye(num_classes)[tensor].to(device)

    return onehot


def from_onehot(onehot, dim=1):
    """One-hot decoding

    Args:
        onehot (torch.Tensor)
        dim (int, optional)
    """
    return torch.argmax(onehot, dim=dim)


def normalize_img(src, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Normalize image tensor
    Args:
        src (torch.Tensor)
        mean (tuple)
        std (tuple)
    Returns:
        dst (torch.Tensor)
    """
    device = src.device
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    dst = ((src / 255.0) - mean[:, None, None]) / std[:, None, None]
    return dst


def unnormalize_img(src, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Revese normalize image tensor
    Args:
        src (torch.Tensor)
        mean (tuple)
        std (tuple)
    Returns:
        dst (torch.Tensor)
    """
    device = src.device
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    dst = 255.0 * (src * std[:, None, None] + mean[:, None, None])

    return dst
