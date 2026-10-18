import numpy as np
import pytest
import torch

from ehm_tracker.geometry import (
    crop_and_align_image,
    fit_transform,
    project_points,
    transform_points,
    validate_size,
)

SIZES = [(512, 512), (640, 360), (360, 640), (832, 480), (641, 479), (1920, 1080)]


@pytest.mark.parametrize("source_size", SIZES)
@pytest.mark.parametrize("output_size", SIZES)
def test_fit_inverse_and_no_stretch(source_size, output_size):
    width, height = source_size
    matrix = fit_transform(source_size, output_size)
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
    fitted = transform_points(corners, matrix)
    np.testing.assert_allclose(transform_points(fitted, np.linalg.inv(matrix)), corners, atol=3e-4)
    np.testing.assert_allclose(matrix[0, 0], matrix[1, 1])
    assert fitted.min() >= -1e-4
    assert (fitted <= np.array(output_size) + 1e-4).all()
    np.testing.assert_allclose(fitted.mean(axis=0), np.array(output_size) / 2, atol=1e-4)


@pytest.mark.parametrize("output_size", SIZES)
def test_intrinsic_transfer_commutes_with_projection(output_size):
    matrix = fit_transform((1024, 768), output_size)
    intrinsics = np.array([[900, 0, 512], [0, 850, 384], [0, 0, 1]], dtype=float)
    points = np.array([[-0.2, 0.1, 2], [0.4, -0.1, 4], [0, 0, 1]], dtype=float)
    expected = transform_points(project_points(points, intrinsics), matrix)
    np.testing.assert_allclose(project_points(points, matrix @ intrinsics), expected, atol=1e-6)
    np.testing.assert_allclose(np.linalg.inv(matrix) @ (matrix @ intrinsics), intrinsics, atol=2e-4)


@pytest.mark.parametrize("output_size", SIZES)
def test_real_warp_has_requested_dimensions(output_size):
    image = np.zeros((93, 157, 3), dtype=np.uint8)
    image[35:55, 60:90] = 255
    result = crop_and_align_image(image, [0, 0, 157, 93], output_size=output_size)
    assert result["crop_image"].shape == (output_size[1], output_size[0], 3)
    np.testing.assert_allclose(result["M_c2o"] @ result["M_o2c"], np.eye(3), atol=3e-5)


def test_batched_torch_transform_preserves_gradients():
    points = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True)
    matrices = torch.tensor(
        [[[2.0, 0, 10], [0, 2, 20], [0, 0, 1]], [[3.0, 0, -1], [0, 3, -2], [0, 0, 1]]]
    )
    output = transform_points(points, matrices)
    np.testing.assert_allclose(
        output.detach().numpy(),
        transform_points(points.detach().numpy(), matrices.numpy()),
    )
    output.sum().backward()
    torch.testing.assert_close(
        points.grad, torch.tensor([[[2.0, 2.0], [2.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]])
    )


@pytest.mark.parametrize("size", [(0, 1), (-3, 512), (2.5, 3), (True, 7)])
def test_invalid_sizes_fail(size):
    with pytest.raises(ValueError):
        validate_size(size)


def test_degenerate_crop_fails():
    with pytest.raises(ValueError):
        fit_transform((100, 100), (200, 200), [0, 0, 0, 20])


def test_hand_projection_resizing_preserves_metric_depth():
    from utils.general_utils import transform_hand_vertex_to_image

    vertices = torch.tensor([[[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]])
    camera = {
        "pred_cam_t": torch.tensor([[0.0, 0.0, 2.0]]),
        "focal_length": torch.tensor([[100.0, 100.0]]),
    }
    first = transform_hand_vertex_to_image(vertices, camera, torch.eye(3)[None])
    second = transform_hand_vertex_to_image(
        vertices, camera, torch.diag(torch.tensor([2.0, 2.0, 1.0]))[None]
    )
    torch.testing.assert_close(second[..., :2], first[..., :2] * 2)
    torch.testing.assert_close(second[..., 2], first[..., 2])
