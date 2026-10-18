import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from scipy.signal import savgol_filter


def smoothen_poses(poses, window_length=9):
    """Smooth joint angles. Poses and global_root_orient should be given as rotation vectors."""
    batch_size, n_joints = poses.shape[0:2]
    if batch_size < 3:
        return poses
    elif batch_size < window_length:
        window_length = batch_size if batch_size % 2 == 1 else batch_size - 1
    poses = poses.reshape((batch_size, n_joints * 3))
    qs = matrix_to_quaternion(axis_angle_to_matrix(torch.FloatTensor(poses).view(-1, 3))).numpy()
    qs = qs.reshape((-1, n_joints, 4))
    qs = fix_quaternions(qs)
    qs_smooth = []
    for j in range(n_joints):
        qss = savgol_filter(qs[:, j], window_length=window_length, polyorder=2, axis=0)
        qs_smooth.append(qss[:, np.newaxis])
    qs_clean = np.concatenate(qs_smooth, axis=1)
    qs_clean = qs_clean / (np.linalg.norm(qs_clean, axis=-1, keepdims=True) + 1e-07)
    ps_clean = matrix_to_axis_angle(
        quaternion_to_matrix(torch.FloatTensor(qs_clean).view(-1, 4))
    ).numpy()
    ps_clean = np.reshape(ps_clean, [batch_size, n_joints, 3])
    qs_clean = np.nan_to_num(qs_clean)
    return ps_clean


def smoothen_coefficients(coefficents, window_length=9):
    batch_size, n_coefs = coefficents.shape[0:2]
    if batch_size < 3:
        return coefficents
    elif batch_size < window_length:
        window_length = batch_size if batch_size % 2 == 1 else batch_size - 1
    qs_smooth = []
    for j in range(n_coefs):
        qss = savgol_filter(coefficents[:, j], window_length=window_length, polyorder=2, axis=0)
        qs_smooth.append(qss[:, np.newaxis])
    qs_clean = np.concatenate(qs_smooth, axis=1)
    qs_clean = qs_clean / (np.linalg.norm(qs_clean, axis=-1, keepdims=True) + 1e-07)
    qs_clean = np.reshape(qs_clean, [batch_size, n_coefs])
    qs_clean = np.nan_to_num(qs_clean)
    return qs_clean


def fix_quaternions(quats):
    """
    From https://github.com/facebookresearch/QuaterNet/blob/ce2d8016f749d265da9880a8dcb20a9be1a6d69c/common/quaternion.py

    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    :param quats: A numpy array of shape (F, N, 4).
    :return: A numpy array of the same shape.
    """
    assert len(quats.shape) == 3
    assert quats.shape[-1] == 4
    result = quats.copy()
    dot_products = np.sum(quats[1:] * quats[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def moving_average_smooth(x, window_size=5, dim=-1):
    kernel_smooth = torch.ones(window_size).float() / window_size
    kernel_smooth = kernel_smooth[None, None].to(x)
    rad = kernel_smooth.size(-1) // 2
    x = x.transpose(dim, -1)
    x_shape = x.shape[:-1]
    x = rearrange(x, "... f -> (...) 1 f")
    x = F.pad(x[None], (rad, rad, 0, 0), mode="replicate")[0]
    x = F.conv1d(x, kernel_smooth)
    x = x.squeeze(1).reshape(*x_shape, -1)
    x = x.transpose(-1, dim)
    return x
