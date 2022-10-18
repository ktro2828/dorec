from .version import __version__

__all__ = ("TASK_TYPES", "IMAGE_TYPES", "IMAGE_TYPES",
           "INPUT_IMAGE_TYPES", "GT_IMAGE_TYPES",
           "GT_GEOMETRY_TYPES", "GT_KEYPOINT_TYPES",
           "TASK_GTMAP", "__version__")

# Support task types
TASK_TYPES = ("segmentation", "edge", "depth", "keypoint")
# Support image types
IMAGE_TYPES = ("rgb", "mask", "depth", "edge")

# Support input image types
INPUT_IMAGE_TYPES = ("rgb", "depth", "rgbd")

# Support GT types for image tasks
GT_IMAGE_TYPES = ("depth", "mask", "edge")
# Support GT types for geometry tasks
GT_GEOMETRY_TYPES = ("keypoint")

# Mapping of GT keys for each task
TASK_GTMAP = {
    "segmentation": "mask",
    "edge": "edge",
    "depth": "depth",
    "keypoint": "keypoint"
}
