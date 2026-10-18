"""Validate frame coverage, cameras, finite parameters, and preview media."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from ehm_tracker.media import read_media
from ehm_tracker.parameters import assert_finite


def verify(folder):
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata["completed"] is True
    results = np.load(folder / "optim_tracking.npy", allow_pickle=True).item()
    identity = np.load(folder / "id_share_params.npy", allow_pickle=True).item()
    assert_finite(results)
    assert_finite(identity)
    assert list(results) == metadata["frames_keys"]
    assert len(results) == metadata["frame_count"]
    assert metadata["source_frame_indices"] == list(range(len(results)))
    width, height = metadata["tracking_size"]
    for frame in results.values():
        camera = frame["smplx_params"]
        transform = frame["full_crop"]["M_o2c-full"]
        inverse = frame["full_crop"]["M_c2o-full"]
        np.testing.assert_allclose(transform @ inverse, np.eye(3)[None], atol=2e-4)
        np.testing.assert_allclose(
            transform @ camera["K_original"], camera["K_intrinsic"], atol=5e-3
        )
        assert camera["focal_px"].item() > 0 and camera["focal_py"].item() > 0
    if metadata["fps"] is not None:
        frames, fps = read_media(folder / "tracking.mp4")
        assert len(frames) == len(results)
        assert abs(fps - metadata["fps"]) < 0.02
        assert all(frame.shape[:2] == (height, width * 3) for frame in frames)
    else:
        with Image.open(folder / "tracking.png") as image:
            assert image.size == (width * 3, height)
    for subfolder in ["input", "mesh", "overlay"]:
        images = sorted((folder / subfolder).glob("*.png"))
        assert len(images) == len(results)
        for path in images:
            with Image.open(path) as image:
                assert image.size == (width, height)
    return metadata["frame_count"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(f"Verified {verify(args.result)} frames")
