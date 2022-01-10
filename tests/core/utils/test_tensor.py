#!/usr/bin/env python

import torch

from dorec.core.ops.tensor import probalize, normalize_img, unnormalize_img


def test_probalize():
    B, C, H, W = 10, 4, 100, 100
    tensor = torch.rand(B, C, H, W)
    out = probalize(tensor)

    assert torch.allclose(out.sum(1), torch.ones(B, H, W))


def test_normalize_img():
    src = torch.randint(0, 256, (3, 100, 100))
    dst = normalize_img(src)
    assert ((0.0 <= dst).all() * (dst <= 1.0).all()).item()


def test_unnormalize_img():
    src = torch.rand(3, 100, 100)
    dst = unnormalize_img(src)
    assert ((0.0 <= dst).all() * (dst <= 255).all()).item()
