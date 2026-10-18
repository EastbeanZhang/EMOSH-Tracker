"""Required files are checked before importing GPU models."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

REQUIRED_FILES = {
    "EHM body": [
        "smplx/SMPLX_NEUTRAL_2020.npz",
        "smplx/SMPL-X__FLAME_vertex_ids.npy",
        "smplx/MANO_SMPLX_vertex_ids.pkl",
        "smplx/smplx_extra_joints.yaml",
        "smplx/SMPLX_to_J14.pkl",
    ],
    "EHM face": [
        "flame/generic_model.pkl",
        "flame/landmark_embedding.npy",
        "flame/mediapipe_landmark_embedding.npz",
        "flame/203_landmark_embeding.npz",
        "flame/l_eyelid.npy",
        "flame/r_eyelid.npy",
    ],
    "EHM hand": ["mano/MANO_RIGHT.pkl", "mano/selected_hand_ver.npy"],
    "Body initialization": [
        "gvhmr/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
        "gvhmr/checkpoints/vitpose/vitpose-h-multi-coco.pth",
        "gvhmr/checkpoints/hmr2/epoch=10-step=25000.ckpt",
        "gvhmr/checkpoints/hmr2/smpl_mean_params.npz",
        "gvhmr/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
    ],
    "Hand initialization": [
        "hamer/hamer_ckpts/checkpoints/hamer.ckpt",
        "hamer/hamer_ckpts/model_config.yaml",
        "hamer/mano_mean_params.npz",
    ],
    "Face initialization": [
        "teaser/Teaser.pt",
        "lmk_203/landmark.onnx",
        "mediapipe/face_landmarker.task",
    ],
    "Keypoints": ["dwpose/yolox_l.onnx", "dwpose/dw-ll_ucoco_384.onnx"],
}


def asset_path(relative=""):
    root = Path(os.environ.get("EHM_ASSET_DIR", Path(__file__).resolve().parents[1] / "assets"))
    return (root / relative).resolve()


def check_assets(root=None):
    root = asset_path() if root is None else Path(root)
    missing = [
        relative
        for files in REQUIRED_FILES.values()
        for relative in files
        if not (root / relative).is_file() or (root / relative).stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            "Required model files are missing or empty:\n"
            + "\n".join(f"  {root / item}" for item in missing)
            + "\nSee docs/models.md for download and placement instructions."
        )
    for files in REQUIRED_FILES.values():
        for relative in files:
            validate_model_file(root / relative)
    return sum(map(len, REQUIRED_FILES.values()))


def validate_model_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty model file: {path}")
    with path.open("rb") as stream:
        signature = stream.read(4)
    if path.suffix == ".npz" or signature.startswith(b"PK"):
        if not zipfile.is_zipfile(path):
            raise ValueError(f"Incomplete model archive: {path}")
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"Corrupt model archive member {bad}: {path}")
