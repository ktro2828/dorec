#!/usr/bin/env python

import cv2
import numpy as np

CV2_INTEPOLATION = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def imresize(img, size, img_type):
    """Resize image, in case of input image is mask or edge
    Args:
        img (np.ndarray)
        size (tuple)
        img_type (str)
    Returns:
        img (np.ndarray)
    """
    img = cv2.resize(img, size)
    if img_type in ("mask", "edge"):
        img[img < 0.5] = 0.0
        img[img >= 0.5] = 1.0

    return img


def imrotate(img,
             angle,
             center=None,
             scale=1.0,
             border_value=0,
             interpolation='bilinear',
             auto_bound=False):
    """Rotate an image.
    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_value (int): Border value.
        interpolation (str): Same as :func:`resize`.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.
    Returns:
        rotated(np.ndarray): The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(
        img,
        matrix, (w, h),
        flags=CV2_INTEPOLATION[interpolation],
        borderValue=border_value)
    return rotated


def warp_perspective(img, M, size, img_type):
    """Warp image
    Args:
        img ()
        size (tuple): (height, width) order
    """
    img = cv2.warpPerspective(img, M, size)
    if img_type in ("mask", "edge"):
        img[img < 0.5] = 0.0
        img[img >= 0.5] = 1.0

    return img
