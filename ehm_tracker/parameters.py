"""Conversions at the CPU storage / GPU optimization boundary."""

import numpy as np
import torch


def to_tensor_recursive(value, require_grad=False, device="cuda:0"):
    if isinstance(value, dict):
        return {k: to_tensor_recursive(v, require_grad, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_tensor_recursive(v, require_grad, device) for v in value)
    if isinstance(value, (np.ndarray, torch.Tensor)):
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device).detach().clone()
        return tensor.requires_grad_(require_grad)
    return value


def to_numpy_array_recursive(value):
    if isinstance(value, dict):
        return {k: to_numpy_array_recursive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_numpy_array_recursive(v) for v in value)
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value


def squeeze_tensor_recursive(value, dim=1):
    if isinstance(value, dict):
        return {k: squeeze_tensor_recursive(v, dim) for k, v in value.items()}
    return value.squeeze(dim) if isinstance(value, torch.Tensor) else value


def assert_finite(value, path="parameters"):
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}/{key}")
    elif isinstance(value, (np.ndarray, torch.Tensor)):
        finite = (
            torch.isfinite(value).all().item()
            if isinstance(value, torch.Tensor)
            else np.isfinite(value).all()
        )
        if not finite:
            raise FloatingPointError(f"Non-finite values in {path}")
