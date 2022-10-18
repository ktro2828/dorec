
VALID_TASKS = ("keypoint_and_segmentation",
               "keypoint_detection",
               "semantic_segmentation_2d",
               "depth_and_segmentation",
               "depth_edge_segmentation",
               "edge_and_segmentation")

RGB_CONFIG_PATHS = ("tests/sample/config/des_rgb_3cls.yml",
                    "tests/sample/config/des_rgb_4cls.yml",
                    "tests/sample/config/ds_rgb_3cls.yml",
                    "tests/sample/config/ds_rgb_4cls.yml",
                    "tests/sample/config/es_rgb_3cls.yml",
                    "tests/sample/config/es_rgb_4cls.yml",
                    "tests/sample/config/kd_rgb.yml",
                    "tests/sample/config/ks_rgb_3cls.yml",
                    "tests/sample/config/ks_rgb_4cls.yml",
                    "tests/sample/config/s2d_rgb_3cls.yml",
                    "tests/sample/config/s2d_rgb_4cls.yml")

RGBD_CONFIG_PATHS = ("tests/sample/config/des_rgbd.yml",
                     "tests/sample/config/ds_rgbd.yml",
                     "tests/sample/config/es_rgbd.yml",
                     "tests/sample/config/kd_rgbd.yml",
                     "tests/sample/config/ks_rgbd.yml",
                     "tests/sample/config/s2d_rgbd.yml")

CONFIG_PATHS = RGB_CONFIG_PATHS + RGBD_CONFIG_PATHS
