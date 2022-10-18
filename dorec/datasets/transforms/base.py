#!/usr/bin/env python

class TransformBase(object):
    def __init__(self, name=None):
        super(TransformBase, self).__init__()
        self._name = self.__class__.__name__ if name is None else name

    @property
    def name(self):
        return self._name

    def __call__(self, data):
        raise NotImplementedError

    def __repr__(self):
        return self.name
