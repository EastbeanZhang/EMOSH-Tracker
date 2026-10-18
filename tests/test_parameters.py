import numpy as np
import pytest

from ehm_tracker.parameters import (
    assert_finite,
    to_numpy_array_recursive,
    to_tensor_recursive,
)


def test_recursive_device_and_gradient_flags():
    value = {"outer": {"a": np.ones((2, 3)), "b": [np.zeros(2)]}}
    result = to_tensor_recursive(value, require_grad=True, device="cpu")
    assert result["outer"]["a"].device.type == "cpu"
    assert result["outer"]["a"].requires_grad
    assert result["outer"]["b"][0].requires_grad
    np.testing.assert_equal(to_numpy_array_recursive(result)["outer"]["a"], value["outer"]["a"])


def test_nonfinite_parameters_fail_with_location():
    with pytest.raises(FloatingPointError, match="parameters/body"):
        assert_finite({"body": np.array([np.nan])})
