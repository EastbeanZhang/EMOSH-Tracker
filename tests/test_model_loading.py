"""Decoder integration check when the licensed model assets are available."""

import pytest
import torch

from ehm_tracker.assets import REQUIRED_FILES, asset_path


@pytest.mark.models
def test_ehm_assets_load_and_backpropagate():
    required = [
        path for group in ("EHM body", "EHM face", "EHM hand") for path in REQUIRED_FILES[group]
    ]
    if any(not asset_path(path).is_file() for path in required):
        pytest.skip("Licensed EHM model assets are not installed")
    pytest.importorskip("chumpy")
    from modules.human_model.ehm import EHM

    model = EHM(str(asset_path()))
    body = {
        name: torch.zeros(2, count, 3)
        for name, count in [
            ("global_pose", 1),
            ("body_pose", 21),
            ("jaw_pose", 1),
            ("eye_pose", 2),
            ("left_hand_pose", 15),
            ("right_hand_pose", 15),
        ]
    }
    expression = torch.zeros(2, 50, requires_grad=True)
    shape = torch.zeros(1, 300, requires_grad=True)
    identity = {
        "smplx_shape": shape,
        "flame_shape": torch.zeros(1, 300),
        "joints_offset": torch.zeros(1, 55, 3),
    }
    result = model(body, {"expression_params": expression}, identity, return_landmarks=True)
    assert result["vertices"].shape == (2, 10475, 3)
    assert result["landmarks_203"].shape == (2, 203, 3)
    assert all(torch.isfinite(value).all() for value in result.values())
    result["vertices"].square().mean().backward()
    for gradient in [shape.grad, expression.grad]:
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.norm() > 0
