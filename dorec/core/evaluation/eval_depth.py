#!/usr/bin/env python

import torch
import torch.nn.functional as F

"""
Refernces:
    papers:
        http://ylatif.github.io/papers/IROS2016_ccadena.pdf
    codes:
        https://github.com/pierlj/ken-burns-effect/blob/cae3db9decdf3a20de319e662261468796bc047b/utils/utils.py#L117
"""


def depth_eval(
    preds,
    targets,
    methods=("absrel", "rmse"),
    smooth=1e-7
):
    """Evaluate predicted depth
    NOTE:
        absrel; absolute relative error
        sqrel; squared relateve error
        rmse; linear root mean square error
        logrmse; log scale invariant rmse

    Args:
        preds (torch.Tensor): in shape (B, 1, H, W)
        targets (torch.Tensor): in shape (B, 1, H, W)
        method (lists[str]): (absrel, sqrel, rmse, logrmse)
        smooth (float)
    Returns:
        scores (dict[str, float])
    """
    assert preds.shape == targets.shape
    if isinstance(methods, str):
        methods = (methods, )

    preds = F.threshold(preds, threshold=0.0, value=0.0)
    mask = torch.zeros_like(targets)
    mask[targets != 0] = 1.0
    preds = preds * mask + smooth
    targets = targets * mask + smooth

    scores = {}
    for mth in methods:
        if mth == "absrel":
            score = torch.mean(torch.abs(targets - preds) / targets).item()
        elif mth == "sqrel":
            score = torch.mean(((targets - preds) ** 2) / targets).item()
        elif mth == "rmse":
            rmse = (targets - preds) ** 2
            score = torch.sqrt(rmse.mean()).item()
        elif mth == "logrmse":
            log_rmse = (torch.log10(targets) - torch.log10(preds)) ** 2
            score = torch.sqrt(log_rmse.mean()).item()
        else:
            raise ValueError("unsupported method: {}".format(mth))
        scores.update({mth: score})

    return scores
