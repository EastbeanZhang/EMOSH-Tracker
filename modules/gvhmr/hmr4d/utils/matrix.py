import numpy as np
import torch


def get_mat_BtoA(matA, matB):
    """
        return matrix B in the coordinate of A

    Args:
        matA (tensor): [4, 4] world matrix
        matB (tensor): [4, 4] world matrix
    """
    if isinstance(matA, torch.Tensor):
        matA_inv = torch.inverse(matA)
    elif isinstance(matA, np.ndarray):
        matA_inv = np.linalg.inv(matA)
    else:
        raise ValueError
    matA_inv = normalized_matrix(matA_inv)
    if isinstance(matA, torch.Tensor):
        mat_BtoA = torch.matmul(matA_inv, matB)
    elif isinstance(matA, np.ndarray):
        mat_BtoA = np.matmul(matA_inv, matB)
    mat_BtoA = normalized_matrix(mat_BtoA)
    return mat_BtoA


def get_mat_BfromA(matA, matBtoA):
    """
        return world matrix B given matrix A and mat B realtive to A

    Args:
        matA (_type_): [4, 4] world matrix
        matBtoA (_type_): [4, 4] matrix B relative to A
    """
    if isinstance(matA, torch.Tensor):
        matB = torch.matmul(matA, matBtoA)
    if isinstance(matA, np.ndarray):
        matB = np.matmul(matA, matBtoA)
    matB = normalized_matrix(matB)
    return matB


def get_relative_position_to(pos, mat):
    """_summary_

    Args:
        pos (_type_): [N, M, 3] or [N, 3]
        mat (_type_): [N, 4, 4] or [4, 4]

    Returns:
        _type_: _description_
    """
    if isinstance(mat, torch.Tensor):
        mat_inv = torch.inverse(mat)
    elif isinstance(mat, np.ndarray):
        mat_inv = np.linalg.inv(mat)
    else:
        raise ValueError
    mat_inv = normalized_matrix(mat_inv)
    if isinstance(mat, torch.Tensor):
        rot_pos = torch.matmul(mat_inv[..., :-1, :-1], pos.transpose(-1, -2)).transpose(-1, -2)
    elif isinstance(mat, np.ndarray):
        rot_pos = np.matmul(mat_inv[..., :-1, :-1], pos.swapaxes(-1, -2)).swapaxes(-1, -2)
    world_pos = rot_pos + mat_inv[..., None, :-1, 3]
    return world_pos


def get_rotation(mat):
    """_summary_

    Args:
        mat (_type_): [..., 4, 4]

    Returns:
        _type_: _description_
    """
    return mat[..., :-1, :-1]


def set_position(mat, pos):
    """_summary_

    Args:
        mat (_type_): [..., 4, 4]

    Returns:
        _type_: _description_
    """
    mat[..., :-1, 3] = pos
    return mat


def get_position(mat):
    """_summary_

    Args:
        mat (_type_): [..., 4, 4]

    Returns:
        _type_: _description_
    """
    return mat[..., :-1, 3]


def get_position_from(pos, mat):
    """_summary_

    Args:
        pos (_type_): [N, M, 3] or [N, 3]
        mat (_type_): [N, 4, 4] or [4, 4]

    Returns:
        _type_: _description_
    """
    if isinstance(mat, torch.Tensor):
        rot_pos = torch.matmul(mat[..., :-1, :-1], pos.transpose(-1, -2)).transpose(-1, -2)
    elif isinstance(mat, np.ndarray):
        rot_pos = np.matmul(mat[..., :-1, :-1], pos.swapaxes(-1, -2)).swapaxes(-1, -2)
    else:
        raise ValueError
    world_pos = rot_pos + mat[..., None, :-1, 3]
    return world_pos


def normalized_matrix(mat):
    if mat.shape[-1] == 4:
        rot_mat = mat[..., :-1, :-1]
    else:
        rot_mat = mat
    if isinstance(mat, torch.Tensor):
        rot_mat_norm = rot_mat / (rot_mat.norm(2, dim=-2, keepdim=True) + 1e-09)
        norm_mat = torch.zeros_like(mat)
    elif isinstance(mat, np.ndarray):
        rot_mat_norm = rot_mat / (np.linalg.norm(rot_mat, ord=2, axis=-2, keepdims=True) + 1e-09)
        norm_mat = np.zeros_like(mat)
    else:
        raise ValueError
    if mat.shape[-1] == 4:
        norm_mat[..., :-1, :-1] = rot_mat_norm
        norm_mat[..., :-1, -1] = mat[..., :-1, -1]
        norm_mat[..., -1, -1] = 1.0
    else:
        norm_mat = rot_mat_norm
    return norm_mat


def get_TRS(rot_mat, pos):
    """_summary_

    Args:
        rot_mat (tensor): [N, 3, 3]
        pos (tensor): [N, 3]

    Returns:
        mat (tensor): [N, 4, 4]
    """
    if isinstance(rot_mat, torch.Tensor):
        mat = torch.eye(4, device=pos.device).repeat(pos.shape[:-1] + (1, 1))
    elif isinstance(rot_mat, np.ndarray):
        mat = np.eye(4, dtype=np.float32)
        for _ in range(len(pos.shape) - 1):
            mat = mat[None]
        mat = np.tile(mat, pos.shape[:-1] + (1, 1))
    else:
        raise ValueError
    mat[..., :3, :3] = rot_mat
    mat[..., :3, 3] = pos
    mat = normalized_matrix(mat)
    return mat


def quat_xyzw2wxyz(quat):
    new_quat = torch.cat([quat[..., 3:4], quat[..., :3]], dim=-1)
    return new_quat


def quat_wxyz2xyzw(quat):
    new_quat = torch.cat([quat[..., 1:4], quat[..., :1]], dim=-1)
    return new_quat


def normalize(x, eps: float = 1e-09):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


def quat_unit(a):
    return normalize(a)


def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * torch.sin(theta.clone())
    w = torch.cos(theta.clone())
    return quat_unit(torch.cat([xyz, w], dim=-1))


def calc_heading(q, head_ind=0, gravity_axis="z"):
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., head_ind] = 1
    shape = ref_dir.shape[:-1]
    q = q.reshape((-1, 4))
    ref_dir = ref_dir.reshape(-1, 3)
    rot_dir = quat_rotate(q, ref_dir)
    rot_dir = rot_dir.reshape(shape + (3,))
    if gravity_axis == "z":
        heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    elif gravity_axis == "y":
        heading = torch.atan2(rot_dir[..., 0], rot_dir[..., 2])
    elif gravity_axis == "x":
        heading = torch.atan2(rot_dir[..., 2], rot_dir[..., 1])
    return heading


def calc_heading_quat(q, head_ind=0, gravity_axis="z"):
    heading = calc_heading(q, head_ind, gravity_axis=gravity_axis)
    axis = torch.zeros_like(q[..., 0:3])
    if gravity_axis == "z":
        g_axis = 2
    elif gravity_axis == "y":
        g_axis = 1
    elif gravity_axis == "x":
        g_axis = 0
    axis[..., g_axis] = 1
    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q


def forward_kinematics(mat, parent):
    """_summary_

    Args:
        mat ([..., N, 3, 3]): _description_
        parent (): _description_
    """
    if isinstance(mat, torch.Tensor):
        rotations = torch.eye(mat.shape[-1], device=mat.device)
        rotations = rotations.repeat(mat.shape[:-2] + (1, 1))
    else:
        rotations = np.eye(mat.shape[-1], dtype=np.float32)
        rotations = np.tile(rotations, mat.shape[:-2] + (1, 1))
    for i in range(mat.shape[-3]):
        if parent[i] != -1:
            if isinstance(mat, torch.Tensor):
                new_mat = get_mat_BfromA(rotations[..., parent[i], :, :], mat[..., i, :, :])
                rotations = torch.cat(
                    (
                        rotations[..., :i, :, :],
                        new_mat[..., None, :, :],
                        rotations[..., i + 1 :, :, :],
                    ),
                    dim=-3,
                )
            else:
                rotations[..., i, :, :] = get_mat_BfromA(
                    rotations[..., parent[i], :, :], mat[..., i, :, :]
                )
        elif isinstance(mat, torch.Tensor):
            rotations = torch.cat((mat[..., : i + 1, :, :], rotations[..., i + 1 :, :, :]), dim=-3)
        else:
            rotations[..., i, :, :] = mat[..., i, :, :]
    return rotations
