"""Pixel geometry. Sizes are (width, height); arrays are (height, width, channels)."""

from __future__ import annotations

import numpy as np


def validate_size(size):
    if len(size) != 2 or any(isinstance(x, bool) or int(x) != x or x < 1 for x in size):
        raise ValueError("Image size must contain two positive integers: width height")
    return tuple(int(x) for x in size)


def fit_transform(source_size, output_size, box=None):
    """Fit a rectangle without stretching or cropping, adding centered padding."""
    sw, sh = validate_size(source_size)
    ow, oh = validate_size(output_size)
    x1, y1, x2, y2 = (0, 0, sw, sh) if box is None else np.asarray(box, dtype=float)
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box must be finite and have positive area")
    scale = min(ow / (x2 - x1), oh / (y2 - y1))
    return np.array(
        [
            [scale, 0, ow / 2 - scale * (x1 + x2) / 2],
            [0, scale, oh / 2 - scale * (y1 + y2) / 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def transform_points(points, matrix):
    """Apply an affine transform, preserving NumPy or Torch batches and gradients."""
    if points is None:
        return None
    if isinstance(points, np.ndarray):
        matrix = np.asarray(matrix, dtype=points.dtype)
        homogeneous = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
        return np.matmul(homogeneous, np.swapaxes(matrix, -1, -2))[..., :2]
    import torch

    matrix = torch.as_tensor(matrix, dtype=points.dtype, device=points.device)
    homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    return torch.matmul(homogeneous, matrix.transpose(-1, -2))[..., :2]


def crop_and_align_image(
    source_image,
    bounding_box,
    points=None,
    output_size=(512, 512),
    padding_scale=1.0,
    **kwargs,
):
    import cv2

    if not np.isfinite(padding_scale) or padding_scale <= 0:
        raise ValueError("padding_scale must be positive")
    box = np.asarray(bounding_box, dtype=np.float32).reshape(2, 2)
    center = box.mean(axis=0)
    box = (center + (box - center) * padding_scale).reshape(4)
    matrix = fit_transform(source_image.shape[1::-1], output_size, box)
    image = cv2.warpAffine(
        source_image,
        matrix[:2],
        validate_size(output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return {
        "crop_image": image,
        "crop_lmk": transform_points(points, matrix),
        "M_o2c": matrix,
        "M_c2o": np.linalg.inv(matrix),
    }


def get_bounding_box(points, padding=1.4, as_square=True):
    points = np.asarray(points).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1) & (points >= 0).all(axis=1)]
    if len(points) < 2:
        return {"bbox": np.array([0, 0, 1, 1], dtype=np.float32)}
    low, high = points.min(axis=0), points.max(axis=0)
    center, extent = (low + high) / 2, np.maximum(high - low, 1)
    if as_square:
        extent[:] = extent.max()
    extent *= padding
    return {"bbox": np.concatenate([center - extent / 2, center + extent / 2]).astype(np.float32)}


def project_points(points, intrinsics, world_to_camera=None):
    """Reference pinhole projection used by the geometry regression tests."""
    points = np.asarray(points)
    if world_to_camera is not None:
        homogeneous = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
        points = np.matmul(homogeneous, np.swapaxes(world_to_camera, -1, -2))[..., :3]
    pixels = np.matmul(points, np.swapaxes(intrinsics, -1, -2))
    return pixels[..., :2] / pixels[..., 2:3]
