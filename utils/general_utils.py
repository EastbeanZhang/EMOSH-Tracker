import numpy as np
import roma
import torch


def covert_smpl_params_gvhmr(param_dict, image_idx):
    smplx_params = {}
    device = param_dict["smpl_params_incam"]["transl"].device
    smplx_params["pred_cam_t"] = param_dict["smpl_params_incam"]["transl"][
        image_idx : image_idx + 1
    ]
    smplx_params["K_intrinsic"] = param_dict["K_fullimg"][image_idx : image_idx + 1]
    smplx_params["focal_px"] = torch.tensor(
        param_dict["K_fullimg"][image_idx, 0, 0].item()
    ).reshape(1, 1)
    smplx_params["focal_py"] = torch.tensor(
        param_dict["K_fullimg"][image_idx, 1, 1].item()
    ).reshape(1, 1)
    smplx_params["shape"] = torch.zeros((1, 300), dtype=torch.float32).to(device)
    smplx_params["shape"][:, :10] = param_dict["smpl_params_incam"]["betas"][
        image_idx : image_idx + 1, :10
    ]
    smplx_params["body_pose"] = param_dict["smpl_params_incam"]["body_pose"][
        image_idx : image_idx + 1, :
    ].reshape(1, 21, 3)
    smplx_params["global_pose"] = param_dict["smpl_params_incam"]["global_orient"][
        image_idx : image_idx + 1, :
    ].reshape(1, 1, 3)
    smplx_params["w2c_RT"] = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    smplx_params["w2c_RT"][:, :3, 3] = smplx_params["pred_cam_t"]
    return smplx_params


def convert_mano_params(left_hand_pose, right_hand_pose):
    if isinstance(left_hand_pose, np.ndarray):
        left_hand_pose = torch.from_numpy(left_hand_pose).to(torch.float32)
        right_hand_pose = torch.from_numpy(right_hand_pose).to(torch.float32)
    (b, n) = left_hand_pose.shape[:2]
    left_hand_pose = roma.rotmat_to_rotvec(left_hand_pose.flatten(0, 1)).reshape(b, n * 3)
    right_hand_pose = roma.rotmat_to_rotvec(right_hand_pose.flatten(0, 1)).reshape(b, n, 3)
    left_hand_pose[:, 1::3] *= -1
    left_hand_pose[:, 2::3] *= -1
    left_hand_pose = left_hand_pose.reshape(b, n, 3)
    return (left_hand_pose, right_hand_pose)


def transform_hand_vertex_to_image(
    hand_vertices,
    hand_params,
    transform_matrix,
    image_size: int = 256,
    is_left: bool = False,
):
    device = hand_vertices.device
    batch_size = hand_vertices.shape[0]
    points = hand_vertices.clone()
    w2c_RT = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    w2c_RT[:, :3, 3] = hand_params["pred_cam_t"]
    points_cam = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    points_cam = torch.einsum("bij,bnj->bni", w2c_RT, points_cam)
    focalxy = hand_params["focal_length"][:, None, :]
    points_img = points_cam[..., :2] * focalxy / (points_cam[..., 2:3] + 1e-08)
    points_img[..., 0:1] = points_img[..., 0:1] + image_size / 2
    points_img[..., 1:2] = points_img[..., 1:2] + image_size / 2
    points_img = torch.cat([points_img, points_cam[..., 2:3]], dim=-1)
    p = points_img.clone()
    if is_left:
        p[..., 0] = image_size - 1 - p[..., 0]
    device = p.device
    M = transform_matrix.to(device)
    batch_size = M.shape[0]
    T = torch.eye(3, device=device).repeat(batch_size, 1, 1)
    T[:, :2, :2] = M[:, :2, :2]
    # Image resizing changes pixel coordinates; depth remains in model units.
    t = torch.zeros(batch_size, 3, device=device)
    t[:, :2] = M[:, :2, 2]
    final_points = torch.bmm(p, T.mT) + t.unsqueeze(1)
    return final_points
