#!/usr/bin/env python

from torch.utils.data import ConcatDataset as _ConcatDataset


class ConcatDataset(_ConcatDataset):
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(self)
        self._root = self._cumsum_root(datasets)
        self._task = datasets[0].task
        self._image_type = datasets[0].image_type
        self._use_dims = datasets[0].use_dims

    @staticmethod
    def _cumsum_root(datasets):
        """Cumsum root of datasets as list"""
        root = []
        for d in datasets:
            root.append(datasets.root)
        return root

    @property
    def root(self):
        return self._root

    @property
    def task(self):
        return self._task

    @property
    def image_type(self):
        return self._image_type

    @property
    def use_dims(self):
        return self._use_dims
