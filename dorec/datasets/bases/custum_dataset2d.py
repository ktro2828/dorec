#!/usr/bin/env python

from dorec import INPUT_IMAGE_TYPES

from .base import CustumDatasetBase


class CustumDataset2D(CustumDatasetBase):
    """Abstract base class for Image based Dataset
    Args:
        task (tuple(str))
        root (str)
        input_type (str)
        use_dims (int, optional)
        pipelines (list[dict[str, any]])
    """

    def __init__(self, task, root, input_type, use_dims, pipelines):
        super(CustumDataset2D, self).__init__(
            task, root, input_type, use_dims, pipelines)

        assert input_type in INPUT_IMAGE_TYPES, \
            "`input_type` must be in {},"\
            " but got {}".format(INPUT_IMAGE_TYPES, input_type)
        self._check_valid_dims(input_type, use_dims)
        self._input_type = input_type
        self._use_dims = use_dims

        if self._input_type == "depth":
            self.reps = (1, 1, self.use_dims)
        elif self.input_type == "rgbd":
            self.reps = (1, 1, self.use_dims - 3)
        else:
            self.reps = (1, 1, 1)

    @staticmethod
    def _check_valid_dims(input_type, use_dims):
        """Check if the dims is valid for image type"""

        if input_type == "rgbd":
            assert use_dims in (4, 6), \
                "in case of RGBD, `use_dims` must be 4 or 6, "\
                "but got {}".format(use_dims)
        elif input_type == "rgb":
            assert use_dims == 3, \
                "in case of RGB, `use_dims` must be 3, "\
                "but got {}".format(use_dims)
        else:
            assert use_dims in (1, 3), \
                "in case of Depth or Edge, "\
                "`use_dims` must be in 1 or 3, but got {}".format(use_dims)
