#!/usr/bin/env python
# -*- coding: utf-8 -*-

from glob import glob
import math
import os.path as osp
import secrets

import cv2
import numpy as np
import PIL

import imgviz
import labelme

from dorec.core.utils import save_json, save_txt, makedirs, load_json, get_logger


logger = get_logger(modname=__name__)

# Default COLORMAP
COLORMAP_DICT = {
    "_background_": (0, 0, 0),
    "top": (255, 0, 0),
    "low": (0, 255, 0),
    "no": (0, 0, 255)
}
COLORMAP = np.array([[0, 0, 0],
                     [255, 0, 0],
                     [0, 255, 0],
                     [0, 0, 255],
                     [255, 255, 255]], dtype=np.uint8)


def labelme2token(indir, outdir, labels="./labels.txt", with_depth=False, noviz=False, num_keypoints=8):
    """Convert annotated data with labelme to our token specified style
    Args:
        indir (str): input directory path
        outputs (str): output directory path
        labels (str, optional): label file path
        with_depth (bool, optional): whether include depth image data (default: False)
        noviz (bool, optional): whether vizualize semantic mask (default: False)
        num_keypoints (int, optional): number of keypoints
    """
    # Make directories
    anno_dir = osp.join(outdir, "annotations")
    rgb_dir = osp.join(outdir, "images", "rgb")
    mask_dir = osp.join(outdir, "images", "mask")
    edge_dir = osp.join(outdir, "images", "edge")
    makedirs(anno_dir)
    makedirs(anno_dir)
    makedirs(rgb_dir)
    makedirs(mask_dir)
    makedirs(edge_dir)
    if with_depth:
        rgb_indir = osp.join(indir, "rgb")
        depth_indir = osp.join(indir, "depth")
        caminfo_indir = osp.join(indir, "camera_info")
        assert osp.exists(depth_indir) and osp.exists(caminfo_indir)
        depth_dir = osp.join(outdir, "images", "depth")
        caminfo_data = {}
        makedirs(depth_dir)
    else:
        rgb_indir = indir

    if not noviz:
        viz_dir = osp.join(outdir, "images", "viz")

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    logger.info("class_names:", class_names)
    logger.info("class_name_to_id:", class_name_to_id)

    token_data = {}
    kpts_data = {}
    for filename in glob(osp.join(rgb_indir, "*.json")):
        logger.info("Generating dataset from:", filename)
        label_file = labelme.LabelFile(filename=filename)
        # Generate unique token
        token = secrets.token_hex(16)

        # Extract keypoint data
        label_file, ith_kpt_data = extract_kpt_data(label_file, num_keypoints)

        kpts_data.update({token: ith_kpt_data})

        out_img_file = osp.join(rgb_dir, label_file.imagePath)
        out_mask_file = osp.join(mask_dir, label_file.imagePath)
        out_edge_file = osp.join(edge_dir, label_file.imagePath)
        if not noviz:
            out_viz_file = osp.join(viz_dir, label_file.imagePath)

        with open(out_img_file, "wb") as f:
            f.write(label_file.imageData)
        img = labelme.utils.img_data_to_arr(label_file.imageData)

        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id
        )
        edge = shapes_to_edge(img_shape=img.shape, shapes=label_file.shapes)
        # labelme.utils.lblsave(out_mask_file, lbl)
        _lblsave(out_mask_file, lbl)
        _lblsave(out_edge_file, edge)

        if not noviz:
            viz = imgviz.label2rgb(
                lbl,
                imgviz.rgb2gray(img),
                font_size=15,
                label_names=class_names,
                loc="rb"
            )
            imgviz.io.imsave(out_viz_file, viz)

        token_data.update(
            {token: {"rgb": label_file.imagePath,
                     "mask": label_file.imagePath,
                     "edge": label_file.imagePath}})
        if with_depth:
            # Save depth
            depth_fname = osp.splitext(osp.basename(filename))[0] + ".png"
            depth = cv2.imread(osp.join(depth_indir, depth_fname))
            depth_savepath = osp.join(depth_dir, depth_fname)
            cv2.imwrite(depth_savepath, depth)
            # Load camera info
            caminfo_fname = osp.basename(filename)
            caminfo = load_json(osp.join(caminfo_indir, caminfo_fname))
            caminfo_data.update({token: caminfo})
            token_data[token].update({"depth": depth_fname})
    save_json(osp.join(anno_dir, "token.json"), token_data)
    save_json(osp.join(anno_dir, "keypoints.json"), kpts_data)

    if len(kpts_data.keys()) > 0:
        names = list(range(num_keypoints))
        save_txt(osp.join(anno_dir, "keypoint_names.txt"), names)

    if with_depth:
        save_json(osp.join(anno_dir, "camera_info.json"), caminfo_data)


def extract_kpt_data(label_file, num_keypoints):
    """Extract and remove keypoint data from label_file
    Args:
        label_file ()
        num_keypoints (int)
    Returns:
        label_file (): label_file object which is extracted and removed keypoint data
        kpt_data (dict): keypoint data
    """
    kpt_data = {}
    while True:
        # Check whether `point` in shapes
        has_point = False
        for shape in label_file.shapes:
            if shape["shape_type"] == "point":
                has_point = True
                break

        if not has_point:
            break

        for i, shape in enumerate(label_file.shapes):
            if shape["shape_type"] == "point":
                label = shape["label"]
                kpt_data.update(
                    {label: {"pos": shape["points"][0], "vis": 1, "in_pic": 1}})
                label_file.shapes.pop(i)

    for i in range(num_keypoints):
        if kpt_data.get(str(i)) is None:
            kpt_data[str(i)] = {"pos": [0, 0], "vis": 0, "in_pic": 0}

    return label_file, kpt_data


def _lblsave(filename, lbl, fill=True):
    """
    from:
    https://github.com/wkentaro/labelme/blob/main/labelme/utils/_io.pym
    Args:
        filename (str): output filename
        lbl (np.ndarray)
    """
    if osp.splitext(filename)[1] != ".png":
        filename += ".png"
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode="L")
        lbl_pil.putpalette(COLORMAP.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            "[%s] Cannot save the pixel-wise class label as PNG. "
            "Please consider using the .npy format." % filename
        )


def shapes_to_edge(img_shape, shapes):
    edge = np.zeros(img_shape[:2], dtype=np.int32)
    for shape in shapes:
        points = shape["points"]
        shape_type = shape.get("shape_type", None)
        mask = _shape_to_edge(img_shape[:2], points, shape_type)
        # NOTE: COLORMAP[4] = (255, 255, 255)
        edge[mask] = 4

    return edge


def _shape_to_edge(img_shape, points, shape_type=None, line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px ** 2) + (cy - py) ** 2)
        draw.elipse([cx - d, cy - d, cx + d, cy + d], outline=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1)

    mask = np.array(mask, dtype=bool)
    return mask
