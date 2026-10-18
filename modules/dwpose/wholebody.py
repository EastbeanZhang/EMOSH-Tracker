import os.path as osp

import numpy as np
import onnxruntime as ort

from .onnxdet import inference_detector
from .onnxpose import inference_pose
from .util import dst_dwkpt_name, src_dwkpt_name

dst_name_to_idx = {name: i for i, name in enumerate(dst_dwkpt_name)}
# Find the source and destination indices for names that exist in both lists
src_dw_indices_to_move = []
dst_dw_indices_to_fill = []
for i, name in enumerate(src_dwkpt_name):
    if name in dst_name_to_idx:
        src_dw_indices_to_move.append(i)
        dst_dw_indices_to_fill.append(dst_name_to_idx[name])


def change_kpt_name(src_kpt):
    """
    Efficiently reorders keypoints using pre-calculated indices.
    """

    dst_kpt_num = len(dst_dwkpt_name)
    new_kpt = np.zeros(
        (
            (
                src_kpt.shape[0],
                dst_kpt_num,
            )
            + src_kpt.shape[2:]
        ),
        dtype=np.float32,
    )
    new_kpt[:, dst_dw_indices_to_fill] = src_kpt[:, src_dw_indices_to_move]

    return new_kpt


class Wholebody:
    def __init__(self, ckpt_dir: str, use_gpu=True):
        providers = ["CPUExecutionProvider"] if not use_gpu else ["CUDAExecutionProvider"]
        onnx_det = osp.join(ckpt_dir, "yolox_l.onnx")
        onnx_pose = osp.join(ckpt_dir, "dw-ll_ucoco_384.onnx")

        self.session_det = ort.InferenceSession(path_or_bytes=onnx_det, providers=providers)
        self.session_pose = ort.InferenceSession(path_or_bytes=onnx_pose, providers=providers)

    def __call__(self, oriImg):
        det_result = inference_detector(self.session_det, oriImg)
        if det_result is None or len(det_result) == 0:
            return (
                np.empty((0, 134, 2), dtype=np.float32),
                np.empty((0, 134), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 135, 2), dtype=np.float32),
                np.empty((0, 135), dtype=np.float32),
            )
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        new_keypoints_info_2 = change_kpt_name(keypoints_info.copy())

        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)

        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3
        ).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]
        keypoints_135, scores_135 = (
            new_keypoints_info_2[..., :2],
            new_keypoints_info_2[..., 2],
        )

        return keypoints, scores, det_result, keypoints_135, scores_135
