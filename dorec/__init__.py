from .version import __version__

__all__ = ("TASK_TYPES", "IMAGE_TYPES", "IMAGE_TYPES",
           "INPUT_IMAGE_TYPES", "GT_IMAGE_TYPES",
           "GT_GEOMETRY_TYPES", "GT_KEYPOINT_TYPES",
           "TASK_MAP", "__version__")

# Support task types
TASK_TYPES = ("segmentation", "edge", "depth", "keypoint")
# Support image types
IMAGE_TYPES = ("rgb", "mask", "depth", "edge")

INPUT_IMAGE_TYPES = ("rgb", "depth", "rgbd")
GT_IMAGE_TYPES = ("depth", "mask", "edge")
GT_GEOMETRY_TYPES = ("keypoint_pos")
GT_KEYPOINT_TYPES = ("keypoint_pos", "keypoint_in_pic", "keypoint_vis")

# Mapping of GT keys for each task
TASK_MAP = {
    "segmentation": "mask",
    "edge": "edge",
    "depth": "depth"
}
