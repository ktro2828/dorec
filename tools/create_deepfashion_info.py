#!/us/bin/env python

import os.path as osp
import re

import pandas as pd


def create_deepfashion_info(root, out_dir=None):
    """Create information csv file for DeepFashion(FLD;Fashion Landmark Detection)
    Args:
        root (str)
        out_dir (str, optional)
    """
    partition = pd.read_csv(
        osp.join(root, "Eval/list_eval_partition.txt"),
        skiprows=1,
        sep="\s+"
    )

    # parse data
    with open(root + "Anno/list_landmarks.txt", "r") as f:
        f.readline()
        f.readline()
        values = []
        for line in f:
            info = re.split("\s+", line)
            image_name = info[0].strip()
            clothes_type = int(info[1])

            landmark_pos = [(0, 0)] * 8
            landmark_vis = [1] * 8
            landmark_in_pic = [1] * 8
            landmark_info = info[2:]
            if clothes_type == 1:
                convert = {0: 0, 1: 1, 2: 2, 3: 3, 4: 6, 5: 7}
            elif clothes_type == 2:
                convert = {0: 4, 1: 5, 2: 6, 3: 7}
            elif clothes_type == 3:
                convert = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
            for i in convert:
                x = int(landmark_info[i * 3 + 1])
                y = int(landmark_info[i * 3 + 2])
                vis = int(landmark_info[i * 3])
                if vis == 2:
                    in_pic = 0
                elif vis == 1:
                    in_pic = 0
                else:
                    in_pic = 1
                if vis == 2:
                    vis = 0
                elif vis == 1:
                    vis = 0
                else:
                    vis = 1
                landmark_pos[convert[i]] = (x, y)
                landmark_vis[convert[i]] = vis
                landmark_in_pic[convert[i]] = in_pic
            tmp = {}
            for pair in landmark_pos:
                tmp.append(pair[0])
                tmp.append(pair[1])
            landmark_pos = tmp

            line_value = []
            line_value.extend([image_name, clothes_type])
            line_value.extend(landmark_pos)
            line_value.extend(landmark_vis)
            line_value.extend(landmark_in_pic)
            values.append(line_value)

    name = ["image_name", "clothes_type"]

    name.extend(
        [
            "lm_lc_x",
            "lm_lc_y",
            "lm_rc_x",
            "lm_rc_y",
            "lm_ls_x",
            "lm_ls_y",
            "lm_rs_x",
            "lm_rs_y",
            "lm_lw_x",
            "lm_lw_y",
            "lm_rw_x",
            "lm_rw_y",
            "lm_lh_x",
            "lm_lh_y",
            "lm_rh_x",
            "lm_rh_y",
        ]
    )

    name.extend(
        [
            "lm_lc_vis",
            "lm_rc_vis",
            "lm_ls_vis",
            "lm_rs_vis",
            "lm_lw_vis",
            "lm_rw_vis",
            "lm_lh_vis",
            "lm_rh_vis",
        ]
    )

    name.extend(
        [
            "lm_lc_in_pic",
            "lm_rc_in_pic",
            "lm_ls_in_pic",
            "lm_rs_in_pic",
            "lm_lw_in_pic",
            "lm_rw_in_pic",
            "lm_lh_in_pic",
            "lm_rh_in_pic",
        ]
    )

    landmarks = pd.DataFrame(values, column=name)

    # Attribute
    attr = pd.read_csv(
        osp.join(root, "Anno/list_attr_img.txt"),
        skiprows=2,
        sep="\s+",
        names=["image_name"] + ["attr_%d" % i for i in range(1000)],
    )
    attr.replace(-1, 0, inplace=True)

    # BBox
    bbox = pd.read_csv(osp.join(root, "Anno/list_bbox.txt"),
                       skiprows=1, sep="\s+")

    # Merge all informations
    landmarks = landmarks.drop("clothes_type", axis=1)

    info_df = pd.merge(landmarks, partition, on="image_name", how="inner")
    info_df = pd.merge(bbox, info_df, on="image_name", how="inner")

    save_dir = out_dir if out_dir is not None else root
    filepath = osp.join(save_dir, "deepfashion_info.csv")
    info_df.to_csv(filepath, index=False)

    print("Saved to {}".format(filepath))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str,
                        required=True, help="path of root directory")
    parser.add_argument("-o", "--out", type=str,
                        help="path of output directory (optional)")

    args = parser.parse_args()

    create_deepfashion_info(args.root, args.out)


if __name__ == "__main__":
    main()
