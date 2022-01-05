#!/usr/bin/env python

from torch.utils.data import Dataset

from dorec import TASK_TYPES


class CustumDatasetBase(Dataset):
    """Abstract base class for Dataset custumization
    Args:
        task (tuple(str))
        root (str)
        input_type (str)
        use_dims (int, optional)
        pipelines (list[dict[str, any]])
    """

    def __init__(self, task, root, input_type, use_dims, pipelines):
        assert set(task) <= set(TASK_TYPES)
        self._task = task
        self._root = root
        self._input_type = input_type
        self._use_dims = use_dims
        self._pipelines = pipelines

    @property
    def task(self):
        return self._task

    @property
    def root(self):
        return self._root

    @property
    def input_type(self):
        return self._input_type

    @property
    def use_dims(self):
        return self._use_dims

    @property
    def pipelines(self):
        return self._pipelines

    def _load_inputs(self, idx):
        """Load inputs
        Args:
            idx (int)
        Returns:
            inputs (np.ndarray)
        """
        raise NotImplementedError

    def _load_targets(self, idx):
        """Load targets
        Args:
            idx (int)
        Returns:
            targets (dict[str, np.ndarray])
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
