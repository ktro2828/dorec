#!/usr/bin/env python

import numpy as np

__all__ = [
    "normalize_pc",
    "shuffle_data",
    "shuffle_points",
    "rotate_point_cloud",
    "rotate_point_cloud_z",
    "rotate_point_cloud_with_normal",
    "rotate_pertubation_point_cloud_with_normal",
    "rotate_point_cloud_by_angle",
    "rotate_pertubation_point_cloud",
    "jitter_point_cloud",
    "shift_point_cloud",
    "random_scale_point_cloud",
    "point_cloud_to_volume_batch"
]


def normalize_pc(batch_pc):
    """Normalize point cloud

    Args:
        batch_pc (np.ndarray): in shape (B, N, C)

    Returns:
        pc_norm (np.ndarray): in shape (B, N, C)
    """
    centroid = np.mean(batch_pc, axis=1, keepdims=True)
    max_dist = np.max(np.linalg.norm(batch_pc, ord=2, axis=2,
                                     keepdims=True), axis=1, keepdims=True)
    pc_norm = batch_pc - centroid / max_dist
    return pc_norm


def shuffle_data(data, labels):
    """Shuffle data and labels

    Args:
        data (np.ndarray): in shape (B, N)
        labels (np.ndarray): in shape (B)
    """
    idx = range(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def shuffle_points(batch_pc):
    """Shuffle orders of points in each point cloud -- changes FPS behavior.
    Use the same shuffling idx for the entire batch.

    Args:
        batch_pc (np.ndarray): in shape (B, N, C)

    Returns:
        (np.ndarray): shuffled, in shape (B, N, C)
    """
    idx = np.arange(batch_pc.shape[1])
    np.random.shuffle(idx)
    return batch_pc[:, idx, :]


def rotate_point_cloud(batch_pc):
    """Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction

    Args:
      BxNx3 array, original batch of point clouds

    Return:
      BxNx3 array, rotated batch of point clouds
    """
    B, _, C = batch_pc.shape
    assert C == 3, "input point must be 3 dimentional, but got {}".format(C)

    rot_angle = np.random.rand(B, 1) * 2 * np.pi
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rot_mat = np.stack(
        (
            np.hstack([cosval, np.zeros((B, 1)), sinval]),
            np.tile([0, 1, 0], (B, 1)),
            np.hstack([-sinval, np.zeros((B, 1)), cosval]),
        ),
        axis=1,
    )

    rotated_pc = np.matmul(batch_pc, rot_mat)

    return rotated_pc


def rotate_point_cloud_z(batch_pc):
    """Randomly rotate the point clouds to augment the dataset
    rotation is per shape based along up direction

    Args:
        batch_pc (np.ndarray): in shape (B, N, C)

    Returns:
        rotated_pc (np.ndarray): rotated,  in shape (B, N, C)

    Raises:
        AssertionError: input point must be 3-dim
    """
    B, _, C = batch_pc.shape
    assert C == 3, "input point must be 3 dimentional, but got {}".format(C)

    rot_angle = np.random.rand(B, 1) * 2 * np.pi
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rot_mat = np.stack(
        (
            np.hstack([cosval, np.zeros((B, 1)), sinval]),
            np.tile([0, 1, 0], (B, 1)),
            np.hstack([-sinval, np.zeros((B, 1)), cosval])
        ),
        axis=1
    )
    rotated_pc = np.matmul(batch_pc, rot_mat)

    return rotated_pc


def rotate_point_cloud_with_normal(batch_xzy_normal):
    """Randomly rotate xyz, normal point cloud.

    Args:
        batch_xyz_normal (np.ndarray): in shape (B, N, 6).
            first 3 channela are xyz, last 3 all normal

    Returns:
        rotated_xyz_normal (np.ndarray): in shape (B, N, 6)

    Raises:
        AssertionError: input point must be 6-dim
    """
    B, _, C = batch_xzy_normal.shape
    assert C == 6, "input point must be 6 dimnetional, but got {}".format(C)

    rot_angle = np.random.rand(B, 1) * 2 * np.pi
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rot_mat = np.stack(
        (
            np.hstack([cosval, np.zeros((B, 1)), sinval]),
            np.tile([0, 1, 0], (B, 1)),
            np.hstack([-sinval, np.zeros((B, 1)), cosval])
        ),
        axis=1,
    )

    batch_xzy_normal[:, :, :3] = np.matmul(batch_xzy_normal[:, :, :3], rot_mat)
    batch_xzy_normal[:, :, 3:] = np.matmul(batch_xzy_normal[:, :, 3:], rot_mat)

    return batch_xzy_normal


def rotate_pertubation_point_cloud_with_normal(batch_pc, angle_sigma=0.06, angle_clip=0.18):
    """Randomly perturb the point clous by small rotations

    Args:
        batch_pc (np.ndarray): in shape (B, N, 6), xyz and normals

    Returns:
        rotated_pc (np.ndarray): rotated, in shape (B, N, 6)

    Raises:
        AssertionError: input point must be 6-dim
    """
    B, _, C = batch_pc.shape
    assert C == 6, "input point must be 6 dimentional, but got {}".format(C)

    angles = np.clip(angle_sigma * np.random.randn(B, 3), -
                     angle_clip, angle_clip)

    Rx = np.stack(
        (
            np.tile([1, 0, 0], (B, 1)),
            np.hstack([np.zeros((B, 1)), np.cos(
                angles[:, 0]), -np.sin(angles[:, 0])]),
            np.hstack([np.zeros((B, 1)), np.sin(
                angles[:, 0]), np.cos(angles[:, 0])])
        ),
        axis=1,
    )

    Ry = np.stack(
        (
            np.hstack([np.cos(angles[:, 1]), np.zeros(
                (B, 1)), np.sin(angles[:, 1])]),
            np.tile([0, 1, 0], (B, 1)),
            np.hstack([-np.sin(angles[:, 1]),
                       np.zeros((B, 1)), np.cos(angles[:, 1])]),
        ),
        axis=1,
    )

    Rz = np.stack(
        (
            np.hstack([np.cos(angles[:, 2]), -
                       np.sin(angles[:, 2]), np.zeros((B, 1))]),
            np.hstack([np.sin(angles[:, 2]), np.cos(
                angles[:, 2]), np.zeros((B, 1))]),
            np.tile([0, 0, 1], (B, 1))
        ),
        axis=1,
    )

    R = np.matmul(Rz, np.matmul(Ry, Rx))
    batch_pc[:, :, :3] = np.matmul(batch_pc[:, :, :3], R)
    batch_pc[:, :, 3:] = np.matmul(batch_pc[:, :, 3:], R)

    return batch_pc


def rotate_point_cloud_by_angle(batch_pc, rotation_angle):
    """Rotate the point cloud along up direction with certain angle.

    Args:
        batch_pc (np.ndarray): in shape (B, N, 3)
        rotation_angle (float): radian value

    Returns:
        rotated_pc (np.ndarray): rotated, in shape (B, N, 3)

    Raises:
        AssertionError: input point must be 3-dim
    """
    B, _, C = batch_pc.shape
    assert C == 3, "input point must be 3 dimentional, but got {}".format(C)

    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rot_mat = np.stack(
        (
            np.hstack([cosval, np.zeros((B, 1)), sinval]),
            np.tile([0, 1, 0], (B, 1)),
            np.hstack([-sinval, np.zeros((B, 1)), cosval])
        ),
        axis=1,
    )

    rotated_pc = np.matmul(batch_pc, rot_mat)

    return rotated_pc


def rotate_pertubation_point_cloud(batch_pc, angle_sigma=0.06, angle_clip=0.18):
    """Randomly perturb the point clouds by small rotations

    Args:
        batch_pc (np.ndarray): in shape (B, N, 3)
        angle_sigma (float):
        angle_clip (float):

    Returns:
        rotated_pc (np.ndarray): rotated, in shape (B, N, 3)
    """
    B, _, C = batch_pc.shape
    assert C == 3, "input point must be 3 dimentional, but got {}".format(C)

    angles = np.clip(angle_sigma * np.random.randn(B, 3), -
                     angle_clip, angle_clip)

    Rx = np.stack(
        (
            np.tile([1, 0, 0], (B, 1)),
            np.hstack([np.zeros((B, 1)), np.cos(
                angles[:, 0]), -np.sin(angles[:, 0])]),
            np.hstack([np.zeros((B, 1)), np.sin(
                angles[:, 0]), np.cos(angles[:, 0])])
        ),
        axis=1,
    )

    Ry = np.stack(
        (
            np.hstack([np.cos(angles[:, 1]), np.zeros(
                (B, 1)), np.sin(angles[:, 1])]),
            np.tile([0, 1, 0], (B, 1)),
            np.hstack([-np.sin(angles[:, 1]),
                       np.zeros((B, 1)), np.cos(angles[:, 1])]),
        ),
        axis=1,
    )

    Rz = np.stack(
        (
            np.hstack([np.cos(angles[:, 2]), -
                       np.sin(angles[:, 2]), np.zeros((B, 1))]),
            np.hstack([np.sin(angles[:, 2]), np.cos(
                angles[:, 2]), np.zeros((B, 1))]),
            np.tile([0, 0, 1], (B, 1))
        ),
        axis=1,
    )

    R = np.matmul(Rz, np.matmul(Ry, Rx))
    rotated_pc = np.matmul(batch_pc, R)

    return rotated_pc


def jitter_point_cloud(batch_pc, sigma=0.01, clip=0.05):
    """Randomly jitter points. jittering is per point.

    Args:
        batch_pc (np.ndarray): in shape (B, N, 3)
        sigma (float)
        clip (float)

    Returns:
        jittered_pc (np.ndarray): jittered, in shape (B, N, 3)
    """
    B, N, C = batch_pc.shape
    assert C == 3, "input point must be 3 dimentional, but got {}".format(C)

    assert clip > 0, "`clip` must be > 0, but got {}".format(clip)

    jittered_pc = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_pc += batch_pc
    return jittered_pc


def shift_point_cloud(batch_pc, shift_range=0.1):
    """Randomly shift point cloud. Shift is per point cloud.

    Args:
        batch_pc (np.ndarray): in shape (B, N, 3)

    Returns:
        (np.ndarray): shifted, in shape (B, N, 3)
    """
    B, N, C = batch_pc.shape
    assert C == 3, "input point must be 3 dimentional, but got {}".format(C)

    shifts = np.random.uniform(-shift_range, shift_range, (B, 1, C))
    batch_pc += shifts
    return batch_pc


def random_scale_point_cloud(batch_pc, scale_low=0.8, scale_high=1.25):
    """Randomly scale the point cloud. Scale is per point cloud.

    Args:
        batch_pc (np.ndarray): in shape (B, N, 3)

    Returns:
        (np.ndarray): scaled, in shape (B, N, 3)
    """
    B, N, C = batch_pc.shape
    assert C == 3, "input point must be 3 dimentional, but got {}".format(C)

    scales = np.random.uniform(scale_low, scale_high, (B, 1, 1))
    batch_pc *= scales
    return batch_pc


# TODO : update random point dropout
def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    """Randomly dropout the point cloud.
    dropouted point cloud is set as same as the first point

    Args:
        batch_pc (np.ndarray): in shape (B, N, 3)

    Returns:
        dropped_pc (np.ndarray): dropped, in shape (B, N, 3)"""
    B, N, C = batch_pc.shape
    assert C == 3, "input point must be 3 dimentional, but got {}".format(C)

    drop_ratio = np.random.rand(B, N, 1) * max_dropout_ratio
    np.where(np.random.rand(B, N, 1) <= drop_ratio)


# Ref: https://github.com/charlesq34/pointnet/blob/539db60eb63335ae00fe0da0c8e38c791c764d2b/utils/pc_util.py


def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """
    Args:
        point_clouds (np.ndarray): in shape (B, N, 3)
        vsize (int, optional): size of volume
        radius (float, optional): length of radius
        flatten (bool, optional): indicates whether flatten points

    Returns:
        (np.ndarray): in shape (B, vsize, vsize, vsize)
    """
    vol_list = []
    B = point_clouds.shape[0]
    for i in range(B):
        vol = point_cloud_to_volume(np.squeeze(
            point_clouds[i, :, :]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)


def point_cloud_to_volume(points, vsize, radius=1.0):
    """Convert points to volume, assumes points are in range [-radius, radius]

    Args:
        points (np.ndarray): in shape (N, 3)
        vsize (int): size of volume
        radius (float, optional): length of radius

    Returns:
        (np.ndarray): in shape (visze, vsize, vsize)
    """
    assert points.shape[1] == 3, "input points must be 3-dimentional, but got {}".format(
        points.shape[1])
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol


def volume_to_point_cloud(vol):
    """
    Args:
    vol (np.ndarray): vol is occupancy grid (value = 0 or 1) of size (vsize, vsize, vsize)

    Returns:
        points (np.ndarray): in shape (N, 3)
    """
    vsize = vol.shape[0]
    assert (
        vol.shape[1] == vsize and vol.shape[2] == vsize
    ), "`vol` must be in shape (vsize, vsize, vsize), but got {}".format(vol.shape)
