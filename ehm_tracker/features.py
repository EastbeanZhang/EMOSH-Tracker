import numpy as np
import roma
import torch
from tqdm import tqdm

from utils.crop_utils import (
    crop_and_align_image,
    crop_image_by_bbox,
    get_batch_gvhmr,
    get_bbx_xys_from_xyxy,
    get_bounding_box,
    parse_bbox_from_landmark_lite,
    transform_points,
)
from utils.general_utils import convert_mano_params, covert_smpl_params_gvhmr
from utils.smooth_utils import moving_average_smooth


class FeatureExtractor:
    @torch.no_grad()
    def run_gvhmr(self, body_images, bbx_xys=None):
        num_images = len(body_images)
        if bbx_xys is None:
            bbx_xys = np.array(
                [
                    self.target_size[0] / 2,
                    self.target_size[1] / 2,
                    min(self.target_size[0], self.target_size[1]),
                ]
            )
            bbx_xys = bbx_xys[None].repeat(num_images, axis=0)
        body_images = torch.as_tensor(body_images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        bbx_xys = torch.tensor(bbx_xys, dtype=torch.float32, device=self.device)
        if num_images == 1:
            result = self.gvhmr_model(body_images.repeat(2, 1, 1, 1), bbx_xys.repeat(2, 1))
            return {
                key: {k: v[:1] for k, v in value.items()} if isinstance(value, dict) else value[:1]
                for key, value in result.items()
                if key != "net_outputs"
            }
        return self.gvhmr_model(body_images, bbx_xys)

    @torch.no_grad()
    def crop_images(self, image, box=None, square_box=None):
        image_result = {"ori_image": image}
        init_result = {}
        (dwpose_result, dwpose_raw_result) = self.dwpose_detector(image)
        dwpose_raw_result["keypoints"] = dwpose_raw_result["keypoints_135"]
        dwpose_raw_result["scores"] = dwpose_raw_result["scores_135"]
        dwpose_raw_result.pop("keypoints_135")
        dwpose_raw_result.pop("scores_135")
        if len(dwpose_raw_result["bbox"]) == 0:
            print("dwpose do not detect any body, skip")
            return (None, None)
        ori_box = dwpose_raw_result["bbox"].reshape(2, 2)
        if box is not None:
            (dwpose_raw_result["bbox"], dwpose_result["bbox"]) = (box, box)
        if square_box is not None:
            dwpose_result["square_bbox"] = square_box
        crop_full_res = crop_and_align_image(
            image, dwpose_result["bbox"], output_size=self.target_size
        )
        ori_box_body = transform_points(ori_box, crop_full_res["M_o2c"]).reshape(1, 4)
        bbx_xys_body = get_bbx_xys_from_xyxy(torch.tensor(ori_box_body), base_enlarge=1.2).numpy()
        image_box_xyxy = np.array([[0, 0], [image.shape[1], image.shape[0]]], dtype=np.float32)
        image_box_xyxy = transform_points(image_box_xyxy, crop_full_res["M_o2c"]).reshape(1, 4)
        init_result["dwpose_raw_res"] = dwpose_raw_result
        init_result["dwpose_res"] = {
            "keypoints": transform_points(dwpose_raw_result["keypoints"], crop_full_res["M_o2c"]),
            "faces": dwpose_result["faces"],
            "hands": dwpose_result["hands"],
            "scores": dwpose_raw_result["scores"],
        }
        init_result["full_crop"] = {
            "M_o2c-full": crop_full_res["M_o2c"],
            "M_c2o-full": crop_full_res["M_c2o"],
            "bbx_xys": bbx_xys_body,
            "bbx_xyxy": ori_box_body,
            "image_box_xyxy": image_box_xyxy,
        }
        full_crop_image = crop_full_res["crop_image"].copy()
        image_result["full_crop_image"] = full_crop_image
        face_box = get_bounding_box(dwpose_result["faces"], 1.6)
        (x1, y1, x2, y2) = face_box["bbox"]
        crop_head_res = crop_and_align_image(image, face_box["bbox"], output_size=[256, 256])
        if x1 >= x2 - 4 or y1 >= y2 - 4:
            crop_head_res["crop_image"] = np.zeros((256, 256, 3), dtype=np.uint8)
            crop_head_res["crop_image"][0, 0, :] = 255
        init_result["head_crop"] = {
            "M_o2c": crop_head_res["M_o2c"],
            "M_c2o": crop_head_res["M_c2o"],
        }
        image_result["head_crop_image"] = crop_head_res["crop_image"]
        hands_lmk = dwpose_result["hands"]
        left_hand_lmk = hands_lmk[0]
        right_hand_lmk = hands_lmk[1]
        left_hand_box = parse_bbox_from_landmark_lite(left_hand_lmk)
        right_hand_box = parse_bbox_from_landmark_lite(right_hand_lmk)
        crop_left_hand_res = crop_image_by_bbox(
            image, left_hand_box["bbox"], lmk=left_hand_lmk, scale=1.3, dsize=256
        )
        crop_right_hand_res = crop_image_by_bbox(
            image, right_hand_box["bbox"], lmk=right_hand_lmk, scale=1.3, dsize=256
        )
        image_result["left_hand_crop_image"] = crop_left_hand_res["crop_image"]
        image_result["right_hand_crop_image"] = crop_right_hand_res["crop_image"]
        init_result.update(
            {
                "left_hand_crop": {
                    "M_o2c": crop_left_hand_res["M_o2c"],
                    "M_c2o": crop_left_hand_res["M_c2o"],
                }
            }
        )
        init_result.update(
            {
                "right_hand_crop": {
                    "M_o2c": crop_right_hand_res["M_o2c"],
                    "M_c2o": crop_right_hand_res["M_c2o"],
                }
            }
        )
        (x1, y1, x2, y2) = left_hand_box["bbox"]
        if x1 >= x2 - 4 or y1 >= y2 - 4:
            image_result["left_hand_crop_image"] = np.zeros((256, 256, 3), dtype=np.uint8)
            image_result["left_hand_crop_image"][[0, 0, -1, -1], [0, -1, 0, -1], :] = 123
        (x1, y1, x2, y2) = right_hand_box["bbox"]
        if x1 >= x2 - 4 or y1 >= y2 - 4:
            image_result["right_hand_crop_image"] = np.zeros((256, 256, 3), dtype=np.uint8)
            image_result["right_hand_crop_image"][[0, 0, -1, -1], [0, -1, 0, -1], :] = 123
        return (image_result, init_result)

    @torch.no_grad()
    def extract_body_pose(self, bbx_xys_list, full_images, head_images, lhand_images, rhand_images):
        num_images = len(full_images)
        batch_size = self.args.inference_batch_size
        boxes = torch.from_numpy(np.asarray(bbx_xys_list, dtype=np.float32)).squeeze(1)
        bbx_xys = moving_average_smooth(boxes, window_size=5, dim=0)
        bbx_xys = moving_average_smooth(bbx_xys, window_size=5, dim=0)
        (body_images, bbx_xys_final) = get_batch_gvhmr(np.stack(full_images, axis=0), bbx_xys)
        print("Extracting body pose ...")
        body_params = self.run_gvhmr(body_images, bbx_xys_final.numpy())
        lhand_image = np.stack(lhand_images, axis=0)
        rhand_image = np.stack(rhand_images, axis=0)
        head_image = np.stack(head_images, axis=0)
        lhand_image = torch.tensor(lhand_image, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        rhand_image = torch.tensor(rhand_image, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        head_image = torch.tensor(head_image, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        lhand_images = torch.flip(lhand_image, dims=[-1])
        (lhand_out, rhand_out) = (
            {
                "hand_pose": [],
                "hand_betas": [],
                "hand_orient": [],
                "hand_cam_t": [],
                "hand_focal_length": [],
            },
            {
                "hand_pose": [],
                "hand_betas": [],
                "hand_orient": [],
                "hand_cam_t": [],
                "hand_focal_length": [],
            },
        )
        teaser_out = {}
        lmk203_out = []
        for j in tqdm(range(0, num_images, batch_size), desc="head hand pose"):
            lhand_temp_out = self.hamer_encoder(lhand_images[j : j + batch_size].to(self.device))
            rhand_temp_out = self.hamer_encoder(rhand_image[j : j + batch_size].to(self.device))
            teaser_temp_out = self.teaser_encoder(head_image[j : j + batch_size].to(self.device))
            lmk203_temp_out = self.lmk203_model(head_image[j : j + batch_size].to(self.device))
            lhand_out["hand_pose"].append(lhand_temp_out["pred_mano_params"]["hand_pose"])
            rhand_out["hand_pose"].append(rhand_temp_out["pred_mano_params"]["hand_pose"])
            lhand_out["hand_betas"].append(lhand_temp_out["pred_mano_params"]["betas"])
            rhand_out["hand_betas"].append(rhand_temp_out["pred_mano_params"]["betas"])
            lhand_out["hand_orient"].append(lhand_temp_out["pred_mano_params"]["global_orient"])
            rhand_out["hand_orient"].append(rhand_temp_out["pred_mano_params"]["global_orient"])
            lhand_out["hand_cam_t"].append(lhand_temp_out["pred_cam_t"])
            rhand_out["hand_cam_t"].append(rhand_temp_out["pred_cam_t"])
            lhand_out["hand_focal_length"].append(lhand_temp_out["focal_length"])
            rhand_out["hand_focal_length"].append(rhand_temp_out["focal_length"])
            lmk203_out.append(lmk203_temp_out)
            for key in teaser_temp_out.keys():
                if key not in teaser_out:
                    teaser_out[key] = []
                teaser_out[key].append(teaser_temp_out[key])
        lhand_out["hand_pose"] = torch.cat(lhand_out["hand_pose"], dim=0)
        rhand_out["hand_pose"] = torch.cat(rhand_out["hand_pose"], dim=0)
        lhand_out["hand_betas"] = torch.cat(lhand_out["hand_betas"], dim=0)
        rhand_out["hand_betas"] = torch.cat(rhand_out["hand_betas"], dim=0)
        lhand_out["hand_orient"] = torch.cat(lhand_out["hand_orient"], dim=0)
        rhand_out["hand_orient"] = torch.cat(rhand_out["hand_orient"], dim=0)
        lhand_out["hand_cam_t"] = torch.cat(lhand_out["hand_cam_t"], dim=0)
        rhand_out["hand_cam_t"] = torch.cat(rhand_out["hand_cam_t"], dim=0)
        lhand_out["hand_focal_length"] = torch.cat(lhand_out["hand_focal_length"], dim=0)
        rhand_out["hand_focal_length"] = torch.cat(rhand_out["hand_focal_length"], dim=0)
        lmk203_out = torch.cat(lmk203_out, dim=0)
        for key in teaser_out.keys():
            teaser_out[key] = torch.cat(teaser_out[key], dim=0)
        return (body_params, bbx_xys, teaser_out, lhand_out, rhand_out, lmk203_out)

    @torch.no_grad()
    def extract_other_pose(
        self,
        image_name,
        image_idx,
        head_crop_image,
        left_hand_crop_image,
        right_hand_crop_image,
        body_params,
        head_params,
        lhand_params,
        rhand_params,
        lmk203_out,
    ):
        init_result = {}
        id_share_result = {}
        smplx_params = covert_smpl_params_gvhmr(body_params, image_idx)
        init_result["smplx_params"] = smplx_params
        id_share_result["smplx_shape"] = smplx_params["shape"]
        init_result["smplx_params"].pop("shape")
        crop_head_image_np = head_crop_image
        face_exist = True
        lmkmp = self.lmkmp_model(crop_head_image_np, True)
        lmk203 = lmk203_out[image_idx]
        teaser_out = {
            key: head_params[key][image_idx : image_idx + 1] for key in head_params.keys()
        }
        init_result["face_valid"] = np.array([[1]], dtype=np.float32)
        if lmkmp is None or lmk203 is None or (not face_exist):
            init_result["face_valid"] = init_result["face_valid"] * 0.0
            teaser_out = {key: teaser_out[key] * 0.0 for key in teaser_out.keys()}
        if lmk203 is not None:
            lmk203 = lmk203.cpu().numpy()
        else:
            lmk203 = np.zeros((1, 203, 2), dtype=np.float32)
        if lmkmp is not None:
            lmkmp = lmkmp[None]
        else:
            lmkmp = np.zeros((1, 478, 3), dtype=np.float32)
        init_result["flame_params"] = teaser_out
        id_share_result["flame_shape"] = teaser_out["shape_params"]
        init_result["flame_params"].pop("shape_params")
        init_result.update({"lmk203": lmk203, "lmkmp": lmkmp})
        lhand_image_np = left_hand_crop_image
        rhand_image_np = right_hand_crop_image
        lh_out = lhand_params["hand_pose"][image_idx : image_idx + 1]
        rh_out = rhand_params["hand_pose"][image_idx : image_idx + 1]
        (lhand_pose, rhand_pose) = convert_mano_params(lh_out, rh_out)
        lhand_oreint = roma.rotmat_to_rotvec(
            lhand_params["hand_orient"][image_idx : image_idx + 1].flatten(0, 1)
        ).reshape(1, 1, 3)
        rhand_orient = roma.rotmat_to_rotvec(
            rhand_params["hand_orient"][image_idx : image_idx + 1].flatten(0, 1)
        ).reshape(1, 1, 3)
        lhand_save_params = {
            "hand_pose": lhand_pose,
            "betas": lhand_params["hand_betas"][image_idx : image_idx + 1],
            "global_orient": lhand_oreint,
            "pred_cam_t": lhand_params["hand_cam_t"][image_idx : image_idx + 1],
            "focal_length": lhand_params["hand_focal_length"][image_idx : image_idx + 1],
        }
        rhand_save_params = {
            "hand_pose": rhand_pose,
            "betas": rhand_params["hand_betas"][image_idx : image_idx + 1],
            "global_orient": rhand_orient,
            "pred_cam_t": rhand_params["hand_cam_t"][image_idx : image_idx + 1],
            "focal_length": rhand_params["hand_focal_length"][image_idx : image_idx + 1],
        }
        if (lhand_image_np[[0, 0, -1, -1], [0, -1, 0, -1], :] == 123).sum() == 4 * 3:
            lhand_pose *= 0.0
        if (rhand_image_np[[0, 0, -1, -1], [0, -1, 0, -1], :] == 123).sum() == 4 * 3:
            rhand_pose *= 0.0
        init_result["smplx_params"]["left_hand_pose"] = lhand_pose
        init_result["smplx_params"]["right_hand_pose"] = rhand_pose
        init_result["smplx_params"]["eye_pose"] = torch.zeros((1, 2, 3), dtype=torch.float32)
        init_result["smplx_params"]["jaw_pose"] = init_result["flame_params"]["jaw_params"][None]
        init_result["left_hand_params"] = lhand_save_params
        init_result["right_hand_params"] = rhand_save_params
        return (init_result, id_share_result)
