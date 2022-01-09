#!/usr/bin/env python

from box import Box

import torch
import torch.nn.functional as F

from dorec.core import HEADS
from dorec.core.ops import probalize

from .head_base import HeadBase


@HEADS.register()
class MultiHead(HeadBase):
    """Head module which has multi head
    Args:
        cfg (Box)
    """

    def __init__(self, cfg):
        deep_sup = cfg.pop("deep_sup") \
            if cfg.get("deep_sup") is not None else False
        fc_dim = cfg.pop("fc_dim")
        super(MultiHead, self).__init__(deep_sup)
        self.heads = {}
        self.task = []
        for tsk in cfg.keys():
            head_cfg = cfg[tsk].copy()
            head_cfg.deep_sup = deep_sup
            head_cfg.fc_dim = fc_dim
            self.heads[tsk] = HEADS.build(head_cfg)
            self.task.append(tsk)

    def forward(self, features, segSize=None):
        """
        Args:
            features (list[torch.Tensor])
            segSize (tuple[int], optional)
        Returns:
            out (tuple[torch.Tensor, list[torch.Tensor]])
        """
        out = []
        for tsk in self.task:
            out.append(self.heads[tsk](features, segSize))
        return tuple(out)


@HEADS.register()
class DualHead(HeadBase):
    """Dual head module which has serialize option
    Args:
        head1 (Box[str, any], dict)
        head2 (Box[str, any], dict)
        fc_dim (int)
        concatenate (bool, optional)
    """

    def __init__(self, head1, head2, fc_dim, concatenate=True, deep_sup=False):
        super(DualHead, self).__init__(deep_sup)
        if not isinstance(head1, Box):
            head1 = Box(head1)
        if not isinstance(head2, Box):
            head2 = Box(head2)

        head1.fc_dim = fc_dim
        head1.deep_sup = deep_sup
        head2.deep_sup = deep_sup
        if concatenate:
            head2.fc_dim = head1.num_classes + fc_dim
        else:
            head2.fc_dim = fc_dim

        self.head1 = HEADS.build(head1)
        self.head2 = HEADS.build(head2)
        self.concatenate = concatenate

        # Weight init
        self.apply(self.init_weight)

    def _do_concatenate(self, features, head1_out):
        """
        Args:
            features (list[torch.Tensor])
            head1_out (list[torch.Tensor], torch.Tensor)
        Returns:
            x (list[torch.Tensor])
        """
        if not self.deep_sup:
            out = probalize(head1_out)
            x = [torch.cat((features[-1], out), dim=1)]
        else:
            x = []
            for feat, h1_out in zip(features, head1_out):
                out = probalize(h1_out)
                x.append(torch.cat((feat, out), dim=1))
        return x

    def forward(self, features, segSize=None):
        """If self.deep_sup=True, returns list of torch.Tensor
        Args:
            features (list[torch.Tensor])
            segSize (tuple, optional)
        Returns:
            head1_out (list[torch.Tensor], torch.Tensor)
            head2_out (list[torch.Tensor], torch.Tensor)
        """
        head1_out = self.head1(features)

        x = self._do_concatenate(features, head1_out) \
            if self.concatenate else features

        head2_out = self.head2(x)

        if segSize is not None:
            if self.deep_sup:
                h1_outputs = []
                for h1_out in head1_out:
                    h1_out = F.interpolate(
                        h1_out,
                        size=segSize,
                        mode="bilinear",
                        align_corners=False
                    )
                    h1_outputs.append(h1_out)
                h2_outputs = []
                for h2_out in head2_out:
                    h2_out = F.interpolate(
                        h2_out,
                        size=segSize,
                        mode="bilinear",
                        align_corners=False
                    )
                    h2_outputs.append(h2_out)
                return h1_outputs, h2_outputs
            else:
                head1_out = F.interpolate(
                    head1_out,
                    size=segSize,
                    mode="bilinear",
                    align_corners=False
                )
                head2_out = F.interpolate(
                    head2_out,
                    size=segSize,
                    mode="bilinear",
                    align_corners=False
                )

        return head1_out, head2_out
