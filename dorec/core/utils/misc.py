#!/usr/bin/env python

import time

import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MultiAverageMeter(object):
    """Computes and stores the multi length averages and current values
    Args:
        length (int): length of values
    """

    def __init__(self, length):
        self.length = length
        self.reset()

    def reset(self):
        self.vals = np.zeros(self.length)
        self.avgs = np.zeros(self.length)
        self.sums = np.zeros(self.length)
        self.count = 0

    def update(self, vals, n=1):
        self.vals = vals
        self.sums = self.sums + (vals * n)
        self.count += n
        self.avgs = self.sums / self.count

    def vals2dict(self):
        out = {}
        for n, val in enumerate(self.vals):
            out.update({str(n): val})
        return out


class Timer(object):
    """Compute time"""

    def __init__(self):
        self.clock = {}

    def start(self, *keys):
        """Start timer naming clock"""
        if len(keys) == 0:
            self.clock["default"] = time.time()
        else:
            for key in keys:
                assert isinstance(
                    key, str), "`keys` must be str iterator, but got {}".format(key)
                self.clock[key] = time.time()

    def end(self, *keys):
        if len(keys) == 0:
            return round(time.time() - self.clock["default"], 5)
        elif len(keys) == 1:
            assert keys[0] in self.clock.keys(
            ), "{} is not in the clock".format(keys[0])
            return round(time.time() - self.clock[keys[0]], 5)
        else:
            intervals = []
            for key in keys:
                assert key in self.clock.keys(), "{} is not in the clock".format(key)
                intervals.append(round(time.time() - self.clock[key], 5))
            return intervals
