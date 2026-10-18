import subprocess
import sys

import pytest

from ehm_tracker.assets import check_assets
from ehm_tracker.cli import build_parser


def test_help_without_gpu_or_model_files():
    result = subprocess.run([sys.executable, "track.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--target-size" in result.stdout


def test_missing_assets_are_reported_together(tmp_path):
    with pytest.raises(FileNotFoundError, match="smplx/SMPLX_NEUTRAL_2020.npz") as error:
        check_assets(tmp_path)
    assert "hamer.ckpt" in str(error.value)
    assert "docs/models.md" in str(error.value)


def test_invalid_frame_limit_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--max-frames", "0"])


def test_truncated_pytorch_archive_is_rejected(tmp_path):
    from ehm_tracker.assets import validate_model_file

    path = tmp_path / "weights.pt"
    path.write_bytes(b"PK\x03\x04" + b"broken archive")
    with pytest.raises(ValueError, match="Incomplete model archive"):
        validate_model_file(path)
