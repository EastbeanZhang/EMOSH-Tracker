import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from roma import rotvec_to_unitquat
from tqdm import tqdm

from ehm_tracker.optimizer import BodyOptimizer
from ehm_tracker.parameters import (
    squeeze_tensor_recursive,
    to_numpy_array_recursive,
    to_tensor_recursive,
)
from utils.crop_utils import get_square_box_tensor, transform_points
from utils.smooth_utils import smoothen_coefficients, smoothen_poses


class HeadOptimizer(BodyOptimizer):
    def check_face_visibility(self, face_mesh, leye, reye):
        center = face_mesh.mean(1)
        eye = (leye + reye) / 2.0
        eye_vec = center - eye
        cam_vec = center - 0
        eye_vec = F.normalize(torch.stack((eye_vec[:, 0], eye_vec[:, 2]), 1), p=2, dim=1)
        cam_vec = F.normalize(torch.stack((cam_vec[:, 0], cam_vec[:, 2]), 1), p=2, dim=1)
        dot_prod = torch.sum(eye_vec * cam_vec, 1)
        face_valid = dot_prod > -math.cos(math.pi / 180 * 115)
        return face_valid

    def get_init_hand_kpts(
        self, smplx_params, flame_params, id_params, current_landmarks_whole_img, batch_size
    ):
        self.init_w2c_RT = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        self.init_w2c_RT[:, :3, 3] = smplx_params["w2c_RT"][:, :3, 3].detach().clone()
        self.init_result_dict = self.ehm(
            smplx_params, flame_params, id_params, return_landmarks=True
        )
        init_landmarks_whole = self.init_result_dict["landmarks_whole"]
        (landmarks_whole_img, landmarks_whole_cam) = self.perspective_projection(
            self.init_w2c_RT, smplx_params, init_landmarks_whole
        )
        (self.init_vertices_img, self.init_vertices_cam) = self.perspective_projection(
            self.init_w2c_RT, smplx_params, self.init_result_dict["vertices"]
        )
        landmarks_whole_img = landmarks_whole_img + (
            current_landmarks_whole_img[:, -70:, :] - landmarks_whole_img[:, -70:, :]
        ).mean(dim=1, keepdim=True)
        self.init_result_dict["landmarks_whole_img"] = landmarks_whole_img
        self.init_lhand_kpts = landmarks_whole_img[:, 25:45, :]
        self.init_rhand_kpts = landmarks_whole_img[:, 45:65, :]
        self.init_lhand_box = get_square_box_tensor(self.init_lhand_kpts, scale_factor=1.2)
        self.init_rhand_box = get_square_box_tensor(self.init_rhand_kpts, scale_factor=1.2)

    def check_hand_validity_head(
        self,
        smplx_params,
        flame_params,
        id_params,
        current_landmarks_whole_img,
        batch_size,
        lmk_score,
    ):
        smplx_params = deepcopy(smplx_params)
        smplx_params["left_hand_pose"] = smplx_params["left_hand_pose"].clone()
        smplx_params["right_hand_pose"] = smplx_params["right_hand_pose"].clone()
        self.get_init_hand_kpts(
            smplx_params, flame_params, id_params, current_landmarks_whole_img, batch_size
        )
        lhand_score = lmk_score[:, 25:45].mean(dim=1, keepdim=True)
        rhand_score = lmk_score[:, 45:65].mean(dim=1, keepdim=True)
        lhand_invalid = lhand_score < 0.4
        rhand_invalid = rhand_score < 0.4
        landmarks_whole_img = self.init_result_dict["landmarks_whole_img"]
        init_lhand_kpts = landmarks_whole_img[:, 25:45, :]
        init_rhand_kpts = landmarks_whole_img[:, 45:65, :]
        init_lhand_kpts_ymin = init_lhand_kpts[:, :, 1].min(dim=1, keepdim=True)[0]
        init_rhand_kpts_ymin = init_rhand_kpts[:, :, 1].min(dim=1, keepdim=True)[0]
        init_lhand_kpts_ymove = (
            torch.relu(self.image_size[1] - init_lhand_kpts_ymin) + self.image_size[1] / 8
        )
        init_rhand_kpts_ymove = (
            torch.relu(self.image_size[1] - init_rhand_kpts_ymin) + self.image_size[1] / 8
        )
        init_lhand_kpts[:, :, 1:2] = init_lhand_kpts[:, :, 1:2] + init_lhand_kpts_ymove[:, None]
        init_rhand_kpts[:, :, 1:2] = init_rhand_kpts[:, :, 1:2] + init_rhand_kpts_ymove[:, None]
        init_lhand_kpts[:, :, 1:2] = init_lhand_kpts[:, :, 1:2].mean(dim=1, keepdim=True)
        init_rhand_kpts[:, :, 1:2] = init_rhand_kpts[:, :, 1:2].mean(dim=1, keepdim=True)
        self.init_lhand_kpts = init_lhand_kpts
        self.init_rhand_kpts = init_rhand_kpts
        lhand_valid = 1 - lhand_invalid[:, 0].float()
        rhand_valid = 1 - rhand_invalid[:, 0].float()
        return (lhand_valid, rhand_valid)

    def get_dwpose_weights(self, lmk_score, face_valid, lhand_valid=None, rhand_valid=None):
        weight = lmk_score.clone()
        weight[lmk_score <= 0.4] *= 0.0
        weight[:, 1:7] = (lmk_score[:, 1:7] > 0.4) * weight[:, 1:7] * 20
        weight[:, 8:10] = (lmk_score[:, 8:10] > 0.4) * weight[:, 8:10] * 20
        weight[:, 14:20] = (lmk_score[:, 14:20] > 0.6) * weight[:, 14:20] * 6
        weight[:, 25:65] = (lmk_score[:, 25:65] > 0.6) * weight[:, 25:65] * 3
        weight[:, 10:14] = lmk_score[:, 10:14] * weight[:, 10:14] * 10
        if lhand_valid is not None and rhand_valid is not None:
            weight[:, 25:45] = (
                weight[:, 25:45] * lhand_valid[:, None] + (1 - lhand_valid[:, None]) * 0.5
            )
            weight[:, 45:65] = (
                weight[:, 45:65] * rhand_valid[:, None] + (1 - rhand_valid[:, None]) * 0.5
            )
        return weight

    def get_optimizer_one(
        self,
        smplx_params,
        flame_params,
        id_params,
        cam_R_6d,
        cam_T_xy,
        cam_T_z,
        batch_idx=0,
        lr=0.001,
    ):
        decay = float(batch_idx > 0)
        optimizer = torch.optim.AdamW(
            [
                {"params": smplx_params["body_pose"], "lr": lr * 1.0},
                {"params": smplx_params["global_pose"], "lr": lr * 0.1},
                {"params": smplx_params["jaw_pose"], "lr": lr},
                {"params": smplx_params["right_hand_pose"], "lr": lr * 0.5},
                {"params": smplx_params["left_hand_pose"], "lr": lr * 0.5},
                {"params": smplx_params["eye_pose"], "lr": lr},
                {"params": cam_R_6d, "lr": lr * 0.5},
                {"params": cam_T_xy, "lr": lr * 1},
                {"params": cam_T_z, "lr": lr * 5.0},
                {"params": flame_params["expression_params"], "lr": lr},
                {"params": flame_params["eyelid_params"], "lr": lr},
                {"params": id_params["smplx_shape"], "lr": lr * 0.1 * (1 - decay)},
                {"params": id_params["flame_shape"], "lr": lr * 0.1 * (1 - decay)},
                {"params": id_params["head_scale"], "lr": lr * 0.1 * (1 - decay)},
                {"params": id_params["hand_scale"], "lr": lr * 0.1 * (1 - decay)},
                {"params": id_params["joints_offset"], "lr": lr * 0.01 * (1 - decay)},
            ]
        )
        return optimizer

    def perspective_projection(self, w2c_RT, cam_params, points):
        points_cam = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        points_cam = torch.einsum("bij,bnj->bni", w2c_RT, points_cam)
        focalxy = torch.cat(
            [cam_params["focal_px"][..., None, 0:1], cam_params["focal_py"][..., None, 0:1]], dim=-1
        )
        points_img = points_cam[..., :2] * focalxy / (points_cam[..., 2:3] + 1e-08)
        points_img[..., 0:1] = points_img[..., 0:1] + self.image_size[0] / 2
        points_img[..., 1:2] = points_img[..., 1:2] + self.image_size[1] / 2
        return (points_img, points_cam)

    def tracking(self, batch_params, id_params, batch_names, batch_idx=0):
        batch_size = len(batch_names)
        batch_params = squeeze_tensor_recursive(batch_params, dim=1)
        batch_params["flame_params"]["expression_params"] = batch_params["flame_params"][
            "expression_params"
        ].clamp(-3, 3)
        batch_params["flame_params"]["expression_params"] = torch.tensor(
            smoothen_coefficients(
                batch_params["flame_params"]["expression_params"].cpu().numpy(), window_length=15
            ),
            device=self.device,
        )
        batch_params["smplx_params"]["jaw_pose"] = torch.tensor(
            smoothen_poses(
                batch_params["smplx_params"]["jaw_pose"].cpu().numpy(), window_length=15
            ),
            device=self.device,
        )
        init_body_pose = batch_params["smplx_params"]["body_pose"].clone().to(self.device)
        init_global_pose = batch_params["smplx_params"]["global_pose"].clone().to(self.device)
        init_lhand_pose = batch_params["smplx_params"]["left_hand_pose"].clone().to(self.device)
        init_rhand_pose = batch_params["smplx_params"]["right_hand_pose"].clone().to(self.device)
        (init_body_pose_quad, init_global_pose_quad) = (
            rotvec_to_unitquat(init_body_pose),
            rotvec_to_unitquat(init_global_pose),
        )
        (init_lhand_pose_quad, init_rhand_pose_quad) = (
            rotvec_to_unitquat(init_lhand_pose),
            rotvec_to_unitquat(init_rhand_pose),
        )
        batch_params = to_tensor_recursive(batch_params, False, self.device)
        smplx_params = to_tensor_recursive(batch_params["smplx_params"], True, self.device)
        flame_params = to_tensor_recursive(batch_params["flame_params"], True, self.device)
        id_params = to_tensor_recursive(id_params, True, self.device)
        face_valid_o = batch_params["face_valid"]
        cam_R_6d = matrix_to_rotation_6d(smplx_params["w2c_RT"][:, :3, :3].detach().clone())
        cam_T = smplx_params["w2c_RT"][:, :3, 3].detach().clone()
        cam_T_z = cam_T[:, 2:3].detach().clone()
        cam_T_xy = cam_T[:, 0:2].detach().clone()
        (cam_T.requires_grad, cam_R_6d.requires_grad) = (True, True)
        (cam_T_z.requires_grad, cam_T_xy.requires_grad) = (True, True)
        dwpose_res = to_tensor_recursive(batch_params["dwpose_res"], False, self.device)
        full_crop_mats = batch_params["full_crop"]
        head_crop_mats = batch_params["head_crop"]
        M_head_c2f = full_crop_mats["M_o2c-full"] @ head_crop_mats["M_c2o"]
        lmk203 = transform_points(batch_params["lmk203"][..., :2], M_head_c2f)
        lmkmp = transform_points(batch_params["lmkmp"][..., :2], M_head_c2f)
        with torch.no_grad():
            (lhand_valid, rhand_valid) = self.check_hand_validity_head(
                smplx_params,
                flame_params,
                id_params,
                dwpose_res["keypoints"],
                batch_size,
                dwpose_res["scores"],
            )
            dwpose_res["keypoints"][:, 25:45, :] = dwpose_res["keypoints"][
                :, 25:45, :
            ] * lhand_valid[:, None, None] + self.init_lhand_kpts.clone() * (
                1 - lhand_valid[:, None, None]
            )
            dwpose_res["keypoints"][:, 45:65, :] = dwpose_res["keypoints"][
                :, 45:65, :
            ] * rhand_valid[:, None, None] + self.init_rhand_kpts.clone() * (
                1 - rhand_valid[:, None, None]
            )
        body_ver_idx = np.setdiff1d(
            np.arange(self.ehm.v_template.shape[0]), self.ehm.smplx.smplx2flame_ind
        )
        dwpose_weight = self.get_dwpose_weights(
            dwpose_res["scores"], face_valid_o, lhand_valid, rhand_valid
        )
        optimizer_one = self.get_optimizer_one(
            smplx_params, flame_params, id_params, cam_R_6d, cam_T_xy, cam_T_z, batch_idx
        )
        all_parameters = []
        for param_group in optimizer_one.param_groups:
            all_parameters.extend(param_group["params"])
        tqdm_bar = tqdm(range(self.steps), desc="optimizing EHM parameters")
        lambda_smooth = 1.0
        for step_idx in tqdm_bar:
            result_dict = self.ehm(smplx_params, flame_params, id_params, return_landmarks=True)
            cam_R_mat = rotation_6d_to_matrix(cam_R_6d)
            w2c_RT = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
            w2c_RT[:, :3, :3] = cam_R_mat
            cam_T = torch.cat([cam_T_xy, cam_T_z], dim=-1)
            w2c_RT[:, :3, 3] = cam_T
            (loss_kpt_dict, loss_reg, loss_smooth) = ({}, {}, {})
            landmarks_whole = result_dict["landmarks_whole"]
            (landmarks_whole_img, landmarks_whole_cam) = self.perspective_projection(
                w2c_RT, smplx_params, landmarks_whole
            )
            (vertice_img, vertice_cam) = self.perspective_projection(
                w2c_RT, smplx_params, result_dict["vertices"]
            )
            head_vertice_img = vertice_img[:, self.ehm.smplx.smplx2flame_ind, :]
            face_valid = self.check_face_visibility(
                vertice_cam[:, self.ehm.smplx.smplx2flame_ind, :],
                landmarks_whole_cam[:, self.ehm.smplx.kpt_dict["name"].index("L_Eye"), :],
                landmarks_whole_cam[:, self.ehm.smplx.kpt_dict["name"].index("R_Eye"), :],
            )
            face_valid = (face_valid_o[:, 0] * face_valid)[:, None, None]
            loss_kpt_dict["whole_lmk_loss"] = (
                torch.abs(landmarks_whole_img - dwpose_res["keypoints"]) * dwpose_weight[..., None]
            ).mean()
            (lmkmp_img, lmk203_img) = (None, None)
            face_valid = face_valid.float()
            if lmkmp is not None:
                (lmkmp_img, _) = self.perspective_projection(
                    w2c_RT, smplx_params, result_dict["landmarks_mp"]
                )
                loss_kpt_dict["lmkmp_loss"] = (
                    torch.abs(lmkmp[:, self.ehm.flame.lmk_mp_indices, :] - lmkmp_img) * face_valid
                ).mean() * 1
                loss_kpt_dict["gaze_loss"] = (
                    torch.abs(lmkmp[:, 468:478, :] - head_vertice_img[:, self.ehm.eye_mp_idx])
                    * face_valid
                ).mean()
            if lmk203 is not None:
                (lmk203_img, _) = self.perspective_projection(
                    w2c_RT, smplx_params, result_dict["landmarks_203"]
                )
                loss_kpt_dict["lmk203_loss"] = (
                    self.ehm.flame.cal_lmk203_loss(
                        lmk203_img * face_valid, lmk203 * face_valid, flame_params["pose_params"]
                    )
                    * 5
                )
            (body_pose_quad, global_pose_quad) = (
                rotvec_to_unitquat(smplx_params["body_pose"]),
                rotvec_to_unitquat(smplx_params["global_pose"]),
            )
            (lhand_pose_quad, rhand_pose_quad) = (
                rotvec_to_unitquat(smplx_params["left_hand_pose"]),
                rotvec_to_unitquat(smplx_params["right_hand_pose"]),
            )
            loss_reg["body_pose_reg1"] = (
                F.mse_loss(body_pose_quad, init_body_pose_quad) * 10.0
                + F.mse_loss(smplx_params["body_pose"], smplx_params["body_pose"] * 0) * 2
                + F.mse_loss(global_pose_quad, init_global_pose_quad) * 10.0
            )
            loss_reg["hand_pose_reg"] = (
                F.mse_loss(lhand_pose_quad, init_lhand_pose_quad) * 1
                + F.mse_loss(smplx_params["left_hand_pose"], smplx_params["left_hand_pose"] * 0) * 1
                + F.mse_loss(rhand_pose_quad, init_rhand_pose_quad) * 1
                + F.mse_loss(smplx_params["right_hand_pose"], smplx_params["right_hand_pose"] * 0)
                * 1
            )
            loss_reg["jaw_reg"] = torch.square(smplx_params["jaw_pose"][..., 1:]).mean() * 130
            loss_reg["shape_reg"] = (
                torch.square(id_params["smplx_shape"]).mean() * 0.35
                + torch.square(id_params["flame_shape"]).mean() * 0.15
            )
            loss_reg["expression_reg"] = (
                torch.square(flame_params["expression_params"]).mean() * 0.06
            )
            loss_reg["expression_reg_2"] = (
                torch.relu(torch.abs(flame_params["expression_params"]) - 4.0).mean() * 100
            )
            loss_reg["scale_reg"] = ((id_params["head_scale"] - 1) ** 2).mean() * 1000.0 + (
                (id_params["hand_scale"] - 1) ** 2
            ).mean() * 10000.0
            loss_reg["joints_offset_reg"] = (id_params["joints_offset"] ** 2).mean() * 200000.0
            loss_reg["vertices_z_reg"] = (
                torch.abs(vertice_cam[..., 2] - self.init_vertices_cam[..., 2]).mean() * 5
            )
            if batch_size > 1:
                loss_smooth["cam_T_smooth"] = (
                    F.mse_loss(cam_T[1:, 0:2], cam_T[:-1, 0:2]).mean() * 5000.0 * lambda_smooth
                )
                loss_smooth["cam_Tz_smooth"] = (
                    F.mse_loss(cam_T[1:, 2], cam_T[:-1, 2]).mean() * 5000.0 * lambda_smooth
                )
                loss_smooth["cam_R_smooth"] = (
                    F.mse_loss(cam_R_6d[1:], cam_R_6d[:-1]).mean() * 100000.0 * lambda_smooth
                )
                loss_smooth["global_pose_smooth"] = (
                    F.mse_loss(global_pose_quad[1:], global_pose_quad[:-1]).mean() * 200.0
                )
                loss_smooth["body_pose_smooth"] = (
                    F.mse_loss(body_pose_quad[1:], body_pose_quad[:-1]).mean()
                    * 20000.0
                    * lambda_smooth
                )
                loss_smooth["jaw_pose_smooth"] = (
                    F.mse_loss(smplx_params["jaw_pose"][1:], smplx_params["jaw_pose"][:-1]).mean()
                    * 400.0
                )
                loss_smooth["hand_pose_smooth"] = (
                    F.mse_loss(lhand_pose_quad[1:], lhand_pose_quad[:-1]).mean() * 400.0
                    + F.mse_loss(rhand_pose_quad[1:], rhand_pose_quad[:-1]).mean() * 400.0
                )
                loss_smooth["vertice_smooth"] = (
                    F.mse_loss(vertice_cam[1:, body_ver_idx], vertice_cam[:-1, body_ver_idx]).mean()
                    * 50
                )
                loss_smooth["expression_smooth"] = (
                    F.mse_loss(
                        flame_params["expression_params"][1:],
                        flame_params["expression_params"][:-1],
                    ).mean()
                    * 1
                )
            loss_kpt_val = sum(loss_kpt_dict.values()) * self.pixel_scale
            loss_reg_val = sum(loss_reg.values())
            loss_smooth_val = (
                sum(loss_smooth.values()) if batch_size > 1 else loss_kpt_val.new_zeros(())
            )
            total_loss = loss_kpt_val + loss_reg_val + loss_smooth_val
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite tracking loss at step {step_idx}")
            optimizer_one.zero_grad()
            total_loss.backward()
            for param in all_parameters:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        raise FloatingPointError(f"Non-finite tracking gradient at step {step_idx}")
            optimizer_one.step()
            tqdm_bar.set_postfix(
                loss_kpt=loss_kpt_val.item(),
                loss_reg=loss_reg_val.item(),
                loss_smooth=loss_smooth_val.item(),
                batch_idx=batch_idx,
            )
            if step_idx == 1000:
                optimizer_one.param_groups[7]["lr"] = 1e-05
                optimizer_one.param_groups[8]["lr"] = 5e-05
                lambda_smooth = max(loss_kpt_val.item() / (loss_smooth_val.item() + 1e-06), 1.0)
        tqdm_bar.close()
        cam_R_mat = rotation_6d_to_matrix(cam_R_6d)
        cam_T = torch.cat([cam_T_xy, cam_T_z], dim=-1)
        w2c_RT = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        w2c_RT[:, :3, :3] = cam_R_mat
        w2c_RT[:, :3, 3] = cam_T
        smplx_params["w2c_RT"] = w2c_RT
        smplx_params = to_numpy_array_recursive(smplx_params)
        flame_params = to_numpy_array_recursive(flame_params)
        batch_params = to_numpy_array_recursive(batch_params)
        id_params = to_numpy_array_recursive(id_params)
        return (
            self.get_tracking_results(batch_names, smplx_params, flame_params, batch_params),
            id_params,
        )
