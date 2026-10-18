import runpy
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize(
    "source",
    ["modules/hamer/utils/geometry.py", "modules/gvhmr/hmr4d/network/hmr2/utils/geometry.py"],
)
@pytest.mark.parametrize("batch", [1, 3, 4])
def test_rotation_conversion_is_independent_of_batch_axis(source, batch):
    # Load the pure geometry helper without constructing pretrained networks.
    convert = runpy.run_path(str(Path(__file__).resolve().parents[1] / source))["rot6d_to_rotmat"]
    identity_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).repeat(batch, 1)
    torch.testing.assert_close(convert(identity_6d), torch.eye(3)[None].repeat(batch, 1, 1))
