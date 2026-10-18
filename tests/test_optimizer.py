"""Model-free checks of the actual optimizer's pixel projection and smoothing."""

import numpy as np
import pytest
import torch

pytest.importorskip("pytorch3d")
from ehm_tracker.head_optimizer import HeadOptimizer
from ehm_tracker.optimizer import BodyOptimizer
from utils.smooth_utils import smoothen_coefficients, smoothen_poses


@pytest.mark.parametrize("optimizer_type", [BodyOptimizer, HeadOptimizer])
@pytest.mark.parametrize("size", [(512, 512), (640, 360), (360, 640), (641, 479), (1920, 1080)])
def test_optimizer_projection_uses_width_and_height(optimizer_type, size):
    optimizer = optimizer_type.__new__(optimizer_type)
    optimizer.image_size = size
    camera = {"focal_px": torch.tensor([[100.0]]), "focal_py": torch.tensor([[120.0]])}
    points = torch.tensor([[[0.1, 0.2, 2.0]]])
    projected, _ = optimizer.perspective_projection(torch.eye(4)[None], camera, points)
    torch.testing.assert_close(projected, torch.tensor([[[size[0] / 2 + 5, size[1] / 2 + 12]]]))


@pytest.mark.parametrize("count", [1, 2, 3, 8, 16])
def test_short_sequence_smoothing_stays_finite(count):
    assert np.isfinite(smoothen_poses(np.zeros((count, 1, 3), dtype=np.float32))).all()
    assert np.isfinite(smoothen_coefficients(np.zeros((count, 50), dtype=np.float32))).all()


def test_completely_outside_hand_uses_prior():
    optimizer = BodyOptimizer.__new__(BodyOptimizer)
    optimizer.image_size = (640, 360)
    optimizer.hand_iou2_threshold = 0.0
    optimizer.device = "cpu"
    optimizer.dtype = torch.float32
    optimizer.init_lhand_box = torch.tensor([[700.0, 10.0, 740.0, 50.0]])
    optimizer.init_rhand_box = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    optimizer.init_result_dict = {"landmarks_whole_img": torch.zeros(1, 135, 2)}
    optimizer.get_init_hand_kpts = lambda *args: None
    params = {"left_hand_pose": torch.zeros(1, 15, 3), "right_hand_pose": torch.zeros(1, 15, 3)}
    left, right = optimizer.check_hand_validity(
        params, {}, {}, 1, torch.zeros(1, 135, 2), torch.tensor([[0.0, 0.0, 640.0, 360.0]])
    )
    assert left.item() == 0
    assert right.item() == 1


def test_single_image_body_context_is_truncated_back_to_one():
    from types import SimpleNamespace

    from ehm_tracker.features import FeatureExtractor

    extractor = FeatureExtractor()
    extractor.device = "cpu"
    extractor.target_size = (641, 479)
    extractor.args = SimpleNamespace()

    def body_model(images, boxes):
        assert len(images) == len(boxes) == 2
        return {
            "smpl_params_incam": {"transl": torch.zeros(2, 3)},
            "K_fullimg": torch.eye(3)[None].repeat(2, 1, 1),
            "net_outputs": {},
        }

    extractor.gvhmr_model = body_model
    result = extractor.run_gvhmr(torch.zeros(1, 256, 256, 3), np.array([[320.5, 239.5, 400.0]]))
    assert result["K_fullimg"].shape == (1, 3, 3)
    assert result["smpl_params_incam"]["transl"].shape == (1, 3)


def test_fitting_windows_cover_every_frame_and_share_identity():
    from types import SimpleNamespace

    from ehm_tracker.pipeline import Tracker

    tracker = Tracker.__new__(Tracker)
    tracker.args = SimpleNamespace(optimization_batch_size=2)
    calls = []

    class Optimizer:
        def tracking(self, batch_params, id_params, batch_names, batch_idx=0):
            calls.append((batch_idx, list(batch_names), id_params["value"]))
            return {name: {"body": np.zeros(1)} for name in batch_names}, {"value": batch_idx + 1}

    tracker.ehm_tracker = Optimizer()
    initial = {f"frame_{i:06d}": {"a": np.ones(1)} for i in range(5)}
    results, identity = tracker.optimize(initial, {"value": 0})
    assert list(results) == list(initial)
    assert [call[2] for call in calls] == [0, 1, 2]
    assert identity["value"] == 3
