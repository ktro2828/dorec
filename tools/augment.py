#!/usr/bin/env python

import argparse

from dorec.apis import augment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str,
                        required=True, help="root directory path")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="path of config file containing augment info")
    args = parser.parse_args()
    augment(args.input_dir, args.config)
