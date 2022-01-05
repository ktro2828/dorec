#!/usr/bin/env python


import argparse
import os.path as osp
import random

from dorec.core.utils import TokenParser, save_json


def main():
    parser = argparse.ArgumentParser(description="split data randomly")
    parser.add_argument("-r", "--root", type=str, help="root directory path")
    parser.add_argument("-n", "--num_data", type=int,
                        help="number of data to be split. if not specified, split to be half")
    args = parser.parse_args()

    tp = TokenParser(args.root)
    orig_savepath = osp.join(tp.anno_dir, "token_orig.json")
    tp.saveTokenFile(filepath=orig_savepath)
    print("Original is saved to: {}".format(orig_savepath))

    num_data = len(tp)
    print("Got {} data".format(num_data))
    num_split = num_data // 2 if args.num_data is None else args.num_data
    assert num_split < num_data
    print("num_split: ", num_split)

    new_data = {}
    for _ in range(num_split):
        idx = random.randint(0, num_data - 1)
        token = tp.tokens_list[idx]
        data = tp.tokens[token]
        new_data.update({token: data})

    save_json(osp.join(tp.anno_dir, "token.json"), new_data)
    print("New data: {}".format(len(new_data)))


if __name__ == "__main__":
    main()
