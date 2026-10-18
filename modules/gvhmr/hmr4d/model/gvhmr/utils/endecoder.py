import torch
import torch.nn as nn
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix

from .....hmr4d.configs import MainStore, builds
from .....hmr4d.utils import matrix
from .....hmr4d.utils.pylogger import Log
from .....hmr4d.utils.smplx_utils import make_smplx
from . import stats_compose


class EnDecoder(nn.Module):
    def __init__(self, stats_name="DEFAULT_01", noise_pose_k=10):
        super().__init__()
        stats = getattr(stats_compose, stats_name)
        Log.info(f"[EnDecoder] Use {stats_name} for statistics!")
        self.register_buffer("mean", torch.tensor(stats["mean"]).float(), False)
        self.register_buffer("std", torch.tensor(stats["std"]).float(), False)
        self.noise_pose_k = noise_pose_k
        self.smplx_model = make_smplx("supermotion_v437coco17")
        parents = self.smplx_model.parents[:22]
        self.register_buffer("parents_tensor", parents, False)
        self.parents = parents.tolist()

    def fk_v2(self, body_pose, betas, global_orient=None, transl=None, get_intermediate=False):
        """
        Args:
            body_pose: (B, L, 63)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
        Returns:
            joints: (B, L, 22, 3)
        """
        (B, L) = body_pose.shape[:2]
        if global_orient is None:
            global_orient = torch.zeros((B, L, 3), device=body_pose.device)
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)
        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)
        if transl is not None:
            local_skeleton[..., 0, :] += transl
        mat = matrix.get_TRS(rotmat, local_skeleton)
        fk_mat = matrix.forward_kinematics(mat, self.parents)
        joints = matrix.get_position(fk_mat)
        if not get_intermediate:
            return joints
        else:
            return (joints, mat, fk_mat)

    def decode(self, x_norm):
        """x_norm: (B, L, C)"""
        (B, L, C) = x_norm.shape
        x = x_norm * self.std + self.mean
        body_pose_r6d = x[:, :, :126]
        betas = x[:, :, 126:136]
        global_orient_r6d = x[:, :, 136:142]
        global_orient_gv_r6d = x[:, :, 142:148]
        local_transl_vel = x[:, :, 148:151]
        body_pose = matrix_to_axis_angle(rotation_6d_to_matrix(body_pose_r6d.reshape(B, L, -1, 6)))
        body_pose = body_pose.flatten(-2)
        global_orient_c = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_r6d))
        global_orient_gv = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_gv_r6d))
        output = {
            "body_pose": body_pose,
            "betas": betas,
            "global_orient": global_orient_c,
            "global_orient_gv": global_orient_gv,
            "local_transl_vel": local_transl_vel,
        }
        return output


group_name = "endecoder/gvhmr"
cfg_base = builds(EnDecoder, populate_full_signature=True)
MainStore.store(name="v1_no_stdmean", node=cfg_base, group=group_name)
MainStore.store(name="v1", node=cfg_base(stats_name="MM_V1"), group=group_name)
MainStore.store(
    name="v1_amass_local_bedlam_cam",
    node=cfg_base(stats_name="MM_V1_AMASS_LOCAL_BEDLAM_CAM"),
    group=group_name,
)
MainStore.store(name="v2", node=cfg_base(stats_name="MM_V2"), group=group_name)
MainStore.store(name="v2_1", node=cfg_base(stats_name="MM_V2_1"), group=group_name)
