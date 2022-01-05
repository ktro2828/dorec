#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from dorec.apis import labelme2token

FILE_PATH_USAGE = """
[!!]Assuming file system like below,
- IF INCLUDE *DEPTH*
    indir/
    ├── rgb
    │   ├── xxx.png
    │   └── xxx.json
    ├── depth
    │   └── xxx.png
    └── camera_info
        └── xxx.json
- ELSE
    indir/
    ├── xxx.png
    └── xxx.json
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str,
                        help="input directory path")
    parser.add_argument("-o", "--output_dir", type=str,
                        help="output directory path")
    parser.add_argument("--labels", type=str,
                        help="label file path (default: ./tools/lables.txt)")
    parser.add_argument("--with_depth", action="store_true",
                        help="whether inclue depth data (default: false")
    parser.add_argument("--noviz", action="store_true",
                        help="whether viz (default: false)")
    parser.add_argument("--num_keypoints", type=int,
                        default=8,
                        help="maximum number of keypoints (default: 8)")
    args = parser.parse_args()

    try:
        if args.labels is not None:
            labelme2token(
                args.input_dir,
                args.output_dir,
                labels=args.labels,
                with_depth=args.with_depth,
                noviz=args.noviz,
                num_keypoints=args.num_keypoints)
        else:
            labelme2token(
                args.input_dir,
                args.output_dir,
                with_depth=args.with_depth,
                noviz=args.noviz,
                num_keypoints=args.num_keypoints)
    except Exception as e:
        print(e)
        print(FILE_PATH_USAGE)
