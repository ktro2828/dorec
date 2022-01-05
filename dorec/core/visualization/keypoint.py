#!/usr/bin/env python

import cv2
import numpy as np

from dorec.core.ops import imread, COLORMAP, get_max_pred, probalize


def imshow_keypoint(
    img,
    heatmaps=None,
    kpts=None,
    img_ord="chw",
    limbs_pair=None,
    kpt_score_thre=0.0,
    radius=4,
    thickness=1,
    show_keypoint_weight=True,
    show_heatmap=False,
    show_keypoint_name=True
):
    """Draw keypoints and limbs on an image.

    Args:
        image (str, numpy.ndarray, torch.Tensor): image path or image tensor, in shape CxHxW or HxWxC
        heatmaps (torch.Tensor, optional): predicted or GT heatmaps, in shape KxHxW
        kpts (np.ndarray, optional): predicted keypoits positions, in shape Kx2
        img_ord (str, optional): order of image shape(default: 'chw')
        limbs_pair (sequence, optional): list of limbs pair(default: None)
        kpt_score_thre (float, optional): Minimum score of keypoints to be shown. (default: 0.3)
        thickness (int, optional): Thickness of lines(default: 1)
        show_keypoint_weight (bool, optional): indicates whether show keypoint weight respecting to
            score(default: False)
        show_hetmap (bool, optional): indicates whether show heatmap not keypoint position(default: False)
        show_keypoint (bool, optional): indicates whether show name of keypoits as number of keypoint(default: True)

    Raises:
        RuntimeError: at least either heatmaps or kpts must be specified
    """
    img = imread(img, img_ord)
    img_h, img_w = img.shape[:2]

    if heatmaps is None and kpts is None:
        raise RuntimeError(
            "at least either heatmaps or kpts must be specified")

    if heatmaps is not None:
        if show_heatmap:
            heatmaps = probalize(heatmaps)
            heatmaps = heatmaps.cpu().detach().numpy()
            for heatmap in heatmaps:
                hmap = cv2.applyColorMap(
                    np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                img = cv2.addWeighted(img, 0.6, hmap, 0.4, 0)
            return img
        kpts, maxvals = get_max_pred(heatmaps)
    else:
        maxvals = np.ones(kpts.shape[0])

    assert len(kpts) == len(maxvals), \
        "length of kpts and maxvals must be same, but got {} and {}".format(
            len(kpts), len(maxvals))

    for kid, (kpt, kpt_score) in enumerate(zip(kpts, maxvals)):
        x_pos, y_pos = kpt.astype(np.int32)
        kpt_color = COLORMAP[kid % 8]
        if kpt_score > kpt_score_thre:
            if show_keypoint_weight:
                cv2.circle(img, (x_pos, y_pos), radius, kpt_color, -1)
                if show_keypoint_name:
                    cv2.putText(img, str(kid), (x_pos, y_pos),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, kpt_color)
                transparency = max(0, min(1, kpt_score))
                cv2.addWeighted(img, transparency, img, 1 -
                                transparency, 0, dst=img)
            else:
                cv2.circle(img, (x_pos, y_pos), radius, kpt_color, -1)
                if show_keypoint_name:
                    cv2.putText(img, str(kid), (x_pos, y_pos),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, kpt_color)

    # Draw limbs
    if limbs_pair is not None:
        for n, (start, end) in enumerate(limbs_pair):
            line_color = COLORMAP[n % 8]
            cv2.line(img, kpts[start], kpts[end], line_color)

    return img
