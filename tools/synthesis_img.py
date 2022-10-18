#!/usr/bin/env python

import argparse
import os
import os.path as osp
import secrets

import cv2
import numpy as np
import skimage.transform as transform

from dorec.core.utils import TokenParser, save_json

# TODO:
# if: foregroundの一部とbackgourndの最高層領域が重なる -> background側のマスクは全て下層判定
# else: -> backgroundの最高層領域はそのまま残す


def loadMask(filepath, ksize=21):
    """Load mask as RGB image
    Args:
        filepath (str)
        ksize (int)
    Returns:
        mask (np.ndarray): in shape (H, W, 3)
    """
    mask = cv2.imread(filepath)
    mask = cv2.medianBlur(mask, ksize=ksize)
    mask[mask[:, :, 2] > 180] = (0, 0, 255)
    mask[mask[:, :, 1] > 180] = (0, 255, 0)
    mask[mask[:, :, 0] > 180] = (255, 0, 0)
    return mask


def getEdge(mask):
    """Extract edge image with corresponding mask image"""

    if int(cv2.__version__[0]) > 3:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    edge = np.zeros_like(mask).astype(np.uint8)
    edge = cv2.drawContours(edge, contours, -1, 255, 1)

    return edge


def getForegroundMask(mask_path, thresh=100, ksize=21):
    """Load mask image from path"""
    mask = loadMask(mask_path, ksize=ksize)
    _, mask = cv2.threshold(mask, thresh, 255, 0)
    return mask


def getForeground(img, mask):
    """Extract image with corresponding mask image"""
    newImg = cv2.bitwise_and(img.copy(), img, mask=mask.copy())
    return newImg


def composeImg(fg_img, fg_mask, bg_img):
    """Compose foreground and background"""

    assert fg_img.ndim == bg_img.ndim
    fg_mask[fg_mask > 0] = 1

    # resize background
    bg_img = transform.resize(bg_img, fg_img.shape[:2])

    # Substract the forground area from the background
    if len(bg_img.shape) > 2:
        bg_img = bg_img * \
            (1 - fg_mask.reshape(fg_img.shape[0], fg_img.shape[1], 1))
    else:
        bg_img = bg_img * \
            (1 - fg_mask.reshape(fg_img.shape[0], fg_img.shape[1]))

    # Finally, add the foreground
    composed_img = bg_img + fg_img / 255

    return composed_img


def composeMask(fg_mask_img, bg_mask_img, thresh1=100, thresh2=150):
    """
    Args:
        fg_mask_img (np.ndarray): mask for foreground
        bg_mask_img (np.ndarray): mask for background
        thresh1 (int)
        thresh2 (int)
    Returns:
        np.ndarray: composed image
    """

    new_bg_mask_img = np.zeros_like(bg_mask_img).astype(np.uint8)
    if ((fg_mask_img > thresh1) * (bg_mask_img > thresh2)).any():
        new_bg_mask_img[bg_mask_img > thresh1] = 120
    else:
        new_bg_mask_img[((bg_mask_img > thresh1) *
                         (bg_mask_img < thresh2))] = 120
        new_bg_mask_img[bg_mask_img >= thresh2] = 255

    mask = np.zeros_like(fg_mask_img).astype(np.uint8)
    mask[fg_mask_img > thresh1] = 1

    return composeImg(fg_mask_img, mask, new_bg_mask_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, require=True,
                        help="input directory path")
    parser.add_argument("-i2", "--input_dir2", type=str,
                        help="optional input directory")
    parser.add_argument("-o", "--output_dir", type=str, require=True,
                        help="output directory path")
    parser.add_argument("-n", "--num_try", type=int,
                        help="number of try")
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        raise OSError("Output directory aready exists: ", args.output_dir)
    rgb_dir = osp.join(args.output_dir, "images", "rgb")
    mask_dir = osp.join(args.output_dir, "images", "mask")
    edge_dir = osp.join(args.output_dir, "images", "edge")
    anno_dir = osp.join(args.output_dir, "annotations")
    os.makedirs(rgb_dir)
    os.makedirs(mask_dir)
    os.makedirs(edge_dir)
    os.makedirs(anno_dir)

    tp = TokenParser(args.input_dir)
    if args.input_dir2 is not None:
        tp2 = TokenParser(args.input_dir2)
    token_data = {}
    for i in range(args.num_try):
        # Generate unique token
        token = secrets.token_hex(16)

        # Foreground
        img1_path = tp.getFilePath(i, "rgb")
        mask1_path = tp.getFilePath(i, "mask")
        edge1_path = tp.getFilePath(i, "edge")
        # Background
        if args.input_dir2 is not None:
            img2_path = tp2.getFilePath(i + 1, "rgb")
            mask2_path = tp2.getFilePath(i + 1, "mask")
            edge2_path = tp2.getFilePath(i + 1, "edge")
        else:
            img2_path = tp.getFilePath(i + 1, "rgb")
            mask2_path = tp.getFilePath(i + 1, "mask")
            edge2_path = tp.getFilePath(i + 1, "edge")

        mask_img1 = loadMask(mask1_path)
        mask_img2 = loadMask(mask2_path)
        mask1 = getForegroundMask(mask1_path)
        img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
        img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
        edge1 = cv2.imread(edge1_path, 0)
        edge2 = cv2.imread(edge2_path, 0)

        newImg1 = getForeground(img1, mask1)
        newEdge1 = getForeground(edge1, mask1)

        composed_img = composeImg(newImg1, mask1, img2)
        composed_mask = composeMask(mask_img1, mask_img2)
        composed_edge = composeImg(newEdge1, mask1, edge2)

        filename = str(i) + ".png"
        img_fpath = osp.join(rgb_dir, filename)
        mask_fpath = osp.join(mask_dir, filename)
        edge_fpath = osp.join(edge_dir, filename)
        cv2.imwrite(img_fpath, composed_img)
        cv2.imwrite(mask_fpath, composed_mask)
        cv2.imwrite(edge_fpath, composed_edge)
        token_data.update(
            {token: {"rgb": filename, "mask": filename, "edge": filename}})
    token_fpath = osp.join(anno_dir, "token.json")
    save_json(token_fpath, token_data)


if __name__ == "__main__":
    main()
