import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from roma import rotvec_to_unitquat
from tqdm import tqdm

from ehm_tracker.parameters import (
    squeeze_tensor_recursive,
    to_numpy_array_recursive,
    to_tensor_recursive,
)
from ehm_tracker.rendering import MeshRenderer
from modules.human_model.ehm import EHM
from utils.crop_utils import get_iou2_tensor, get_square_box_tensor, transform_points
from utils.general_utils import transform_hand_vertex_to_image
from utils.smooth_utils import smoothen_coefficients, smoothen_poses


class BodyOptimizer:
    def __init__(self, config, model_path, device="cuda:0"):
        self.image_size = config.target_size
        self.device = device
        self.dtype = torch.float32
        # Preserve the pixel-loss tuning when the canvas is resized.
        self.pixel_scale = 512.0 / max(self.image_size)
        self.ehm = EHM(model_path).to(self.device)
        self.renderer = MeshRenderer(self.ehm.smplx.faces_tensor)
        self.steps = config.steps
        self.hand_iou2_threshold = 0.0

    def check_face_visibility(self, face_mesh, leye, reye):
        face_dir = face_mesh[:, 3557, :3] - face_mesh[:, 3520, :3]
        face_dir = F.normalize(face_dir, p=2, dim=1)
        face_valid = face_dir[:, 0].abs() < 1 - math.cos(math.pi / 180 * 70)
        face_valid = face_valid & (face_dir[:, 1].abs() < 0.3)
        face_valid = face_valid & (face_dir[:, 2].abs() > 0.65)
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
        (self.init_landmarks_img, self.init_landmarks_cam) = (
            landmarks_whole_img,
            landmarks_whole_cam,
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

    def check_hand_validity(
        self,
        smplx_params,
        flame_params,
        id_params,
        batch_size,
        current_landmarks_whole_img,
        bbx_xyxy,
    ):
        smplx_params = deepcopy(smplx_params)
        smplx_params["left_hand_pose"] = smplx_params["left_hand_pose"].clone()
        smplx_params["right_hand_pose"] = smplx_params["right_hand_pose"].clone()
        self.get_init_hand_kpts(
            smplx_params, flame_params, id_params, current_landmarks_whole_img, batch_size
        )
        self.init_result_dict["landmarks_whole_gt"] = current_landmarks_whole_img
        lhand_iou = get_iou2_tensor(self.init_lhand_box, bbx_xyxy)
        rhand_iou = get_iou2_tensor(self.init_rhand_box, bbx_xyxy)
        lhand_invalid = lhand_iou <= self.hand_iou2_threshold
        rhand_invalid = rhand_iou <= self.hand_iou2_threshold
        self.init_lhand_kpts = self.init_result_dict["landmarks_whole_img"][:, 25:45, :]
        self.init_rhand_kpts = self.init_result_dict["landmarks_whole_img"][:, 45:65, :]
        lhand_valid = 1 - lhand_invalid.float()
        rhand_valid = 1 - rhand_invalid.float()
        return (lhand_valid, rhand_valid)

    def get_ref_hand_vertices(self, batch_params):
        (
            left_hand_params,
            left_hand_crop_info,
            right_hand_params,
            right_hand_crop_info,
            full_crop_info,
        ) = (
            batch_params["left_hand_params"],
            batch_params["left_hand_crop"],
            batch_params["right_hand_params"],
            batch_params["right_hand_crop"],
            batch_params["full_crop"],
        )
        img_size = 256
        batch_size = left_hand_params["hand_pose"].shape[0]
        left_hand_params["hand_pose"] = (
            left_hand_params["hand_pose"].clone().reshape(batch_size, -1)
        )
        left_hand_params["hand_pose"][:, 1::3] *= -1
        left_hand_params["hand_pose"][:, 2::3] *= -1
        left_hand_params["hand_pose"] = left_hand_params["hand_pose"].reshape(batch_size, -1, 3)
        left_hand_res = self.ehm.mano(left_hand_params, pose_type="aa")
        right_hand_res = self.ehm.mano(right_hand_params, pose_type="aa")
        left_hand_vertices = transform_hand_vertex_to_image(
            left_hand_res["vertices"],
            left_hand_params,
            full_crop_info["M_o2c-full"] @ left_hand_crop_info["M_c2o"],
            img_size,
            True,
        )
        left_hand_joints = transform_hand_vertex_to_image(
            left_hand_res["joints"],
            left_hand_params,
            full_crop_info["M_o2c-full"] @ left_hand_crop_info["M_c2o"],
            img_size,
            True,
        )
        left_hand_vertices[..., 2] = left_hand_vertices[..., 2] - left_hand_joints[:, 0:1, 2]
        right_hand_vertices = transform_hand_vertex_to_image(
            right_hand_res["vertices"],
            right_hand_params,
            full_crop_info["M_o2c-full"] @ right_hand_crop_info["M_c2o"],
            img_size,
        )
        right_hand_joints = transform_hand_vertex_to_image(
            right_hand_res["joints"],
            right_hand_params,
            full_crop_info["M_o2c-full"] @ right_hand_crop_info["M_c2o"],
            img_size,
        )
        right_hand_vertices[..., 2] = right_hand_vertices[..., 2] - right_hand_joints[:, 0:1, 2]
        return (left_hand_vertices, right_hand_vertices)

    def get_dwpose_weights(self, lmk_score, face_valid, lhand_valid=None, rhand_valid=None):
        weight = lmk_score.clone()
        weight[lmk_score <= 0.4] *= 0.0
        weight[:, 1:7] = (lmk_score[:, 1:7] > 0.4) * weight[:, 1:7] * 20
        weight[:, 8:10] = (lmk_score[:, 8:10] > 0.4) * weight[:, 8:10] * 20
        weight[:, 14:20] = (lmk_score[:, 14:20] > 0.6) * weight[:, 14:20] * 6
        weight[:, 25:65] = (lmk_score[:, 25:65] > 0.6) * weight[:, 25:65] * 3
        weight[:, 10:14] = lmk_score[:, 10:14] * weight[:, 10:14] * 20
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
                {"params": smplx_params["global_pose"], "lr": lr * 0.1 * 0.1},
                {"params": smplx_params["jaw_pose"], "lr": lr * 0.5},
                {"params": smplx_params["right_hand_pose"], "lr": lr * 0.1},
                {"params": smplx_params["left_hand_pose"], "lr": lr * 0.1},
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
        points_img = points_cam[..., :2] * focalxy / (points_cam[..., 2:3] + 1e-07)
        points_img[..., 0:1] = points_img[..., 0:1] + self.image_size[0] / 2
        points_img[..., 1:2] = points_img[..., 1:2] + self.image_size[1] / 2
        return (points_img, points_cam)

    def get_tracking_results(self, batch_names, smplx_params, flame_params, init_params):
        tracking_results = {}
        for idx, name in enumerate(batch_names):
            tracking_results[name] = {
                "smplx_params": {
                    "global_pose": smplx_params["global_pose"][idx : idx + 1],
                    "body_pose": smplx_params["body_pose"][idx : idx + 1],
                    "jaw_pose": smplx_params["jaw_pose"][idx : idx + 1],
                    "left_hand_pose": smplx_params["left_hand_pose"][idx : idx + 1],
                    "right_hand_pose": smplx_params["right_hand_pose"][idx : idx + 1],
                    "eye_pose": smplx_params["eye_pose"][idx : idx + 1],
                    "w2c_RT": smplx_params["w2c_RT"][idx : idx + 1],
                    "focal_px": smplx_params["focal_px"][idx : idx + 1],
                    "focal_py": smplx_params["focal_py"][idx : idx + 1],
                },
                "flame_params": {
                    "expression_params": flame_params["expression_params"][idx : idx + 1],
                    "eyelid_params": flame_params["eyelid_params"][idx : idx + 1],
                },
                "full_crop": {
                    "M_o2c-full": init_params["full_crop"]["M_o2c-full"][idx : idx + 1],
                    "M_c2o-full": init_params["full_crop"]["M_c2o-full"][idx : idx + 1],
                    "bbx_xys": init_params["full_crop"]["bbx_xys"][idx : idx + 1],
                    "bbx_xyxy": init_params["full_crop"]["bbx_xyxy"][idx : idx + 1],
                    "image_box_xyxy": init_params["full_crop"]["image_box_xyxy"][idx : idx + 1],
                },
                "head_crop": {
                    "M_o2c": init_params["head_crop"]["M_o2c"][idx : idx + 1],
                    "M_c2o": init_params["head_crop"]["M_c2o"][idx : idx + 1],
                },
                "left_hand_crop": {
                    "M_o2c": init_params["left_hand_crop"]["M_o2c"][idx : idx + 1],
                    "M_c2o": init_params["left_hand_crop"]["M_c2o"][idx : idx + 1],
                },
                "right_hand_crop": {
                    "M_o2c": init_params["right_hand_crop"]["M_o2c"][idx : idx + 1],
                    "M_c2o": init_params["right_hand_crop"]["M_c2o"][idx : idx + 1],
                },
            }
        return tracking_results

    def tracking(self, batch_params, id_params, batch_names, batch_idx=0):
        batch_size = len(batch_names)
        batch_params = squeeze_tensor_recursive(batch_params, dim=1)
        batch_params["smplx_params"]["left_hand_pose"] = torch.tensor(
            smoothen_poses(
                batch_params["smplx_params"]["left_hand_pose"].cpu().numpy(), window_length=29
            ),
            device=self.device,
        )
        batch_params["smplx_params"]["right_hand_pose"] = torch.tensor(
            smoothen_poses(
                batch_params["smplx_params"]["right_hand_pose"].cpu().numpy(), window_length=29
            ),
            device=self.device,
        )
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
                batch_params["smplx_params"]["jaw_pose"].cpu().numpy(), window_length=29
            ),
            device=self.device,
        )
        batch_params["smplx_params"]["body_pose"] = (
            batch_params["smplx_params"]["body_pose"].detach().clone()
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
        cam_T = torch.cat([cam_T_xy, cam_T_z], dim=-1)
        smplx_params["w2c_RT"] = smplx_params["w2c_RT"].detach().clone()
        smplx_params["w2c_RT"][:, :3, 3] = cam_T.detach().clone()
        cam_R_6d.requires_grad = True
        (cam_T_z.requires_grad, cam_T_xy.requires_grad) = (True, True)
        dwpose_res = to_tensor_recursive(batch_params["dwpose_res"], False, self.device)
        full_crop_mats = batch_params["full_crop"]
        head_crop_mats = batch_params["head_crop"]
        M_head_c2f = full_crop_mats["M_o2c-full"] @ head_crop_mats["M_c2o"]
        lmk203 = transform_points(batch_params["lmk203"][..., :2], M_head_c2f)
        lmkmp = transform_points(batch_params["lmkmp"][..., :2], M_head_c2f)
        with torch.no_grad():
            (lhand_valid, rhand_valid) = self.check_hand_validity(
                smplx_params,
                flame_params,
                id_params,
                batch_size,
                dwpose_res["keypoints"],
                batch_params["full_crop"]["image_box_xyxy"],
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
            (left_hand_ref_vertices, right_hand_ref_vertices) = self.get_ref_hand_vertices(
                batch_params
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
        lambda_smooth = 1.0
        tqdm_bar = tqdm(range(self.steps), desc="optimizing EHM parameters")
        for step_idx in tqdm_bar:
            result_dict = self.ehm(smplx_params, flame_params, id_params, return_landmarks=True)
            cam_R_mat = rotation_6d_to_matrix(cam_R_6d)
            cam_T = torch.cat([cam_T_xy, cam_T_z], dim=-1)
            w2c_RT = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
            w2c_RT[:, :3, :3] = cam_R_mat
            w2c_RT[:, :3, 3] = cam_T
            (loss_kpt_dict, loss_reg, loss_smooth, loss_3d) = ({}, {}, {}, {})
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
            rhand_valid_2 = (dwpose_res["scores"][:, 45:65] * rhand_valid[:, None]).mean(
                dim=1
            ) >= 0.7
            lhand_valid_2 = (dwpose_res["scores"][:, 25:45] * lhand_valid[:, None]).mean(
                dim=1
            ) >= 0.7
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
                ).mean() * 1.0
                loss_kpt_dict["gaze_loss"] = (
                    torch.abs(lmkmp[:, 468:478, :] - head_vertice_img[:, self.ehm.eye_mp_idx])
                    * face_valid
                ).mean() * 1.0
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
            vertice_img_3d = torch.cat([vertice_img, vertice_cam[..., 2:3]], dim=-1)
            joints_img_3d = torch.cat([landmarks_whole_img, landmarks_whole_cam[..., 2:3]], dim=-1)
            left_hand_pred_vertices = vertice_img_3d[:, self.ehm.smplx.smplx2mano_ind["left_hand"]]
            right_hand_pred_vertices = vertice_img_3d[
                :, self.ehm.smplx.smplx2mano_ind["right_hand"]
            ]
            left_hand_pred_vertices[..., 2] = (
                left_hand_pred_vertices[..., 2] - joints_img_3d[:, 12:13, 2]
            )
            right_hand_pred_vertices[..., 2] = (
                right_hand_pred_vertices[..., 2] - joints_img_3d[:, 13:14, 2]
            )
            hand_residual_scale = vertice_img.new_tensor([self.pixel_scale, self.pixel_scale, 1.0])
            loss_3d["left_hand"] = (
                (
                    torch.abs(
                        left_hand_pred_vertices[:, self.ehm.mano.selected_vert_ids]
                        - left_hand_ref_vertices[:, self.ehm.mano.selected_vert_ids]
                    )
                    * hand_residual_scale
                    * lhand_valid_2[:, None, None]
                ).mean()
                * 0.5
                * 2
            )
            loss_3d["right_hand"] = (
                (
                    torch.abs(
                        right_hand_pred_vertices[:, self.ehm.mano.selected_vert_ids]
                        - right_hand_ref_vertices[:, self.ehm.mano.selected_vert_ids]
                    )
                    * hand_residual_scale
                    * rhand_valid_2[:, None, None]
                ).mean()
                * 0.5
                * 2
            )
            loss_3d["left_hand"] += (
                (
                    torch.abs(
                        left_hand_pred_vertices[:, self.ehm.mano.selected_vert_ids, 2]
                        - left_hand_ref_vertices[:, self.ehm.mano.selected_vert_ids, 2]
                    )
                    * lhand_valid_2[:, None]
                ).mean()
                * 0.5
                * 2
                * 10
            )
            loss_3d["right_hand"] += (
                (
                    torch.abs(
                        right_hand_pred_vertices[:, self.ehm.mano.selected_vert_ids, 2]
                        - right_hand_ref_vertices[:, self.ehm.mano.selected_vert_ids, 2]
                    )
                    * rhand_valid_2[:, None]
                ).mean()
                * 0.5
                * 2
                * 10
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
            loss_reg["scale_reg"] = ((id_params["head_scale"] - 1) ** 2).mean() * 500.0 + (
                (id_params["hand_scale"] - 1) ** 2
            ).mean() * 10000.0
            loss_reg["joints_offset_reg"] = (id_params["joints_offset"] ** 2).mean() * 200000.0
            loss_reg["vertices_z_reg"] = (
                torch.abs(vertice_cam[..., 2] - self.init_vertices_cam[..., 2]).mean() * 5
                + torch.abs(landmarks_whole_cam[..., 2] - self.init_landmarks_cam[..., 2]).mean()
                * 2.5
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
                    * 10000.0
                    * lambda_smooth
                )
                loss_smooth["body_pose_smooth"] += (
                    F.l1_loss(smplx_params["body_pose"][1:], smplx_params["body_pose"][:-1]).mean()
                    * 100.0
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
                loss_smooth["hand_pose_smooth"] += (
                    F.mse_loss(
                        smplx_params["left_hand_pose"][1:], smplx_params["left_hand_pose"][:-1]
                    ).mean()
                    * 200.0
                    + F.mse_loss(
                        smplx_params["right_hand_pose"][1:], smplx_params["right_hand_pose"][:-1]
                    ).mean()
                    * 200.0
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
            loss_3d_val = sum(loss_3d.values())
            total_loss = loss_kpt_val + loss_reg_val + loss_smooth_val + loss_3d_val
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
                loss_3d=loss_3d_val.item(),
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
