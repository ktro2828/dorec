#!/usr/bin/env python

from ..base import ModuleBase


class HeadBase(ModuleBase):
    """Base class for head module
    Args:
        deep_sup (bool)
    """

    def __init__(self, deep_sup=False, *args, **kwargs):
        super(HeadBase, self).__init__(*args, **kwargs)
        self.deep_sup = deep_sup
