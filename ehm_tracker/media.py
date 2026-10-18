"""RGB media input and dimension-preserving video output."""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def read_media(path, max_frames=None):
    path = Path(path)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        with Image.open(path) as image:
            frames = [np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()]
        return frames, None
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported input format: {path.suffix}")
    frames = []
    with imageio.get_reader(path) as reader:
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps", 0))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Missing or invalid frame rate in {path}")
        for frame in reader:
            frames.append(frame[..., :3])
            if max_frames is not None and len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError(f"No decodable frames in {path}")
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("Changing frame dimensions within a video are unsupported")
    return frames, fps


def encoder_options(size):
    width, height = size
    # 4:2:0 requires even dimensions; 4:4:4 preserves an odd-sized canvas.
    return dict(
        codec="libx264",
        macro_block_size=1,
        pixelformat="yuv444p" if width % 2 or height % 2 else "yuv420p",
    )


@contextmanager
def video_writer(path, fps, size):
    with imageio.get_writer(path, fps=fps, quality=7, **encoder_options(size)) as writer:
        yield writer
