"""Inference models used to initialize body, face, and hand parameters."""

import numpy as np
import torch
from torch import nn

from ehm_tracker.assets import asset_path


class HamerInitializer(nn.Module):
    def __init__(self):
        super().__init__()
        from modules.hamer.configs import get_config
        from modules.hamer.models.hamer import HAMER

        cfg = get_config(str(asset_path("hamer/hamer_ckpts/model_config.yaml")))
        cfg.defrost()
        cfg.MANO.MEAN_PARAMS = str(asset_path("hamer/mano_mean_params.npz"))
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
        cfg.freeze()
        self.model = HAMER(cfg)
        state = torch.load(
            asset_path("hamer/hamer_ckpts/checkpoints/hamer.ckpt"),
            map_location="cpu",
            weights_only=False,
        )["state_dict"]
        state = {k: v for k, v in state.items() if k.startswith(("backbone.", "mano_head."))}
        self.model.load_state_dict(state, strict=True)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def forward(self, images):
        return self.model({"img": (images - self.mean) / self.std})


class FaceLandmarks(nn.Module):
    def __init__(self):
        super().__init__()
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        self.detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(asset_path("mediapipe/face_landmarker.task"))
                ),
                num_faces=1,
                min_face_detection_confidence=0.1,
                min_face_presence_confidence=0.1,
            )
        )

    def forward(self, image, rgb=True):
        result = self.detector.detect(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))
        )
        if not result.face_landmarks:
            return None
        return np.array(
            [[p.x * image.shape[1], p.y * image.shape[0], p.z] for p in result.face_landmarks[0]],
            dtype=np.float32,
        )

    def close(self):
        self.detector.close()


class DenseFaceLandmarks(nn.Module):
    def __init__(self):
        super().__init__()
        import onnxruntime as ort

        # This small ONNX model avoids CUDA/cuDNN coupling by running on the CPU.
        self.session = ort.InferenceSession(
            str(asset_path("lmk_203/landmark.onnx")), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def forward(self, images):
        # The model was exported for single-image input.
        import torch.nn.functional as F

        resized = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        points = []
        for image in resized:
            outputs = self.session.run(None, {self.input_name: image[None].cpu().numpy()})
            landmarks = np.asarray(outputs[2]).reshape(1, 203, 2).copy()
            landmarks[..., 0] *= images.shape[-1]
            landmarks[..., 1] *= images.shape[-2]
            points.append(landmarks)
        return torch.from_numpy(np.concatenate(points)).to(images.device)


class BodyInitializer(nn.Module):
    def __init__(self, image_size, device, batch_size=4):
        super().__init__()
        from hydra import compose, initialize_config_module

        from modules.gvhmr.hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
        from modules.gvhmr.hmr4d.model.gvhmr.utils import endecoder as endecoder
        from modules.gvhmr.hmr4d.network.gvhmr import (
            relative_transformer as relative_transformer,
        )
        from modules.gvhmr.hmr4d.network.hmr2 import load_hmr2
        from modules.gvhmr.hmr4d.utils.preproc.vitpose_pytorch import build_model

        with initialize_config_module(
            version_base="1.3", config_module="modules.gvhmr.hmr4d.configs"
        ):
            cfg = compose(config_name="demo", overrides=[])
        self.model = DemoPL(cfg.pipeline)
        self.model.load_pretrained_model(
            str(asset_path("gvhmr/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"))
        )
        self.pose = build_model(
            "ViTPose_huge_coco_256x192",
            str(asset_path("gvhmr/checkpoints/vitpose/vitpose-h-multi-coco.pth")),
        )
        self.extractor = load_hmr2()
        self.image_size, self.device, self.batch_size = image_size, device, batch_size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, images, boxes):
        from modules.gvhmr.hmr4d.utils.geo.flip_utils import flip_heatmap_coco17
        from modules.gvhmr.hmr4d.utils.geo.hmr_cam import estimate_K
        from modules.gvhmr.hmr4d.utils.geo_transform import compute_cam_angvel
        from modules.gvhmr.hmr4d.utils.kpts.kp2d_utils import keypoints_from_heatmaps

        keypoints, features = [], []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size].to(self.device)
            normalized = (chunk - self.mean) / self.std
            pose_input = normalized[:, :, :, 32:224]
            heatmap, flipped = self.pose(torch.cat([pose_input, pose_input.flip(-1)])).chunk(2)
            heatmap = (heatmap + flip_heatmap_coco17(flipped)) * 0.5
            xy, confidence = keypoints_from_heatmaps(
                heatmap.cpu().numpy(),
                boxes[start : start + len(chunk)].cpu().numpy()[:, :2],
                boxes[start : start + len(chunk)].cpu().numpy()[:, 2:]
                * np.array([0.75, 1.0])
                / 200,
                use_udp=True,
            )
            keypoints.append(torch.from_numpy(np.concatenate([xy, confidence], axis=-1)))
            features.append(self.extractor({"img": normalized}).cpu())
        length = len(images)
        data = {
            "length": torch.tensor(length, device=self.device),
            "bbx_xys": boxes.to(self.device),
            "kp2d": torch.cat(keypoints).to(self.device),
            "f_imgseq": torch.cat(features).to(self.device),
            "K_fullimg": estimate_K(*self.image_size).repeat(length, 1, 1).to(self.device),
            "cam_angvel": compute_cam_angvel(torch.eye(3).repeat(length, 1, 1)).to(self.device),
        }
        return self.model.predict(data, static_cam=True)


def load_initializers(device, image_size, batch_size):
    from modules.dwpose import DWposeDetector
    from modules.teaser import TeaserEncoder

    body = BodyInitializer(image_size, device, batch_size).to(device).eval()
    hands = HamerInitializer().to(device).eval()
    face = TeaserEncoder()
    state = torch.load(asset_path("teaser/Teaser.pt"), map_location="cpu", weights_only=True)
    prefix = "teaser_encoder."
    state = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    face.load_state_dict(state, strict=True)
    return dict(
        dwpose_detector=DWposeDetector(str(asset_path("dwpose")), use_gpu=False),
        gvhmr_model=body,
        hamer_encoder=hands,
        teaser_encoder=face.to(device).eval(),
        lmkmp_model=FaceLandmarks(),
        lmk203_model=DenseFaceLandmarks(),
    )
