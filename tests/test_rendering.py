"""CUDA silhouette checks, independent of pretrained model weights."""

import numpy as np
import pytest
import torch

pytest.importorskip("pytorch3d")
from ehm_tracker.rendering import MeshRenderer


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA renderer validation requires a GPU")
@pytest.mark.parametrize(
    "size", [(512, 512), (640, 360), (360, 640), (832, 480), (641, 479), (1920, 1080)]
)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_rendered_silhouette_matches_pinhole_projection(size, batch_size):
    width, height = size
    vertices = torch.tensor(
        [[[-0.4, -0.2, 2.0], [0.2, -0.2, 2.0], [-0.1, 0.4, 2.0]]], device="cuda"
    ).repeat(batch_size, 1, 1)
    faces = torch.tensor([[0, 1, 2]], device="cuda")
    renderer = MeshRenderer(faces)
    camera = {
        "w2c_RT": torch.eye(4, device="cuda")[None].repeat(batch_size, 1, 1),
        "image_size": size,
        "focal_length": torch.tensor([[200.0, 300.0]], device="cuda").repeat(batch_size, 1),
        "princpt": [width / 2, height / 2],
    }
    images = renderer.render_mesh(vertices, camera)
    assert images.shape == (batch_size, height, width, 3)
    projected = np.array(
        [
            [width / 2 - 40, height / 2 - 30],
            [width / 2 + 20, height / 2 - 30],
            [width / 2 - 10, height / 2 + 60],
        ]
    )
    y, x = np.mgrid[:height, :width]
    pixels = np.stack([x + 0.5, y + 0.5], axis=-1)
    distances = []
    for start, end in zip(projected, np.roll(projected, -1, axis=0)):
        edge = end - start
        offset = pixels - start
        distances.append(
            (edge[0] * offset[..., 1] - edge[1] * offset[..., 0]) / np.linalg.norm(edge)
        )
    signed_distance = np.min(distances, axis=0)
    # A sharp tip need not contain a pixel center. Compare the whole silhouette,
    # allowing only the one-pixel edge band affected by rasterization/blending.
    for image in images:
        foreground = image.max(axis=-1) > 0
        assert foreground.any()
        assert foreground[signed_distance > 1].all()
        assert not foreground[signed_distance < -1].any()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA renderer validation requires a GPU")
def test_dense_faces_do_not_leave_rasterization_gaps(capfd):
    vertices = torch.tensor(
        [
            [
                [-0.01, -0.01, 2.0],
                [0.01, -0.01, 2.0],
                [0.0, 0.01, 2.0],
                [-1.0, -1.0, 3.0],
                [1.0, -1.0, 3.0],
                [0.0, 1.0, 3.0],
            ]
        ],
        device="cuda",
    )
    faces = torch.cat(
        [
            torch.tensor([[0, 1, 2]], device="cuda").repeat(20000, 1),
            torch.tensor([[3, 4, 5]], device="cuda"),
        ]
    )
    camera = {
        "w2c_RT": torch.eye(4, device="cuda")[None],
        "image_size": (128, 128),
        "focal_length": torch.tensor([[100.0, 100.0]], device="cuda"),
        "princpt": [64.0, 64.0],
    }
    image = MeshRenderer(faces).render_mesh(vertices, camera)[0]
    # The large last triangle covers this region; tiny faces contain no pixel centers.
    assert (image[60:68, 60:68].max(axis=-1) > 0).all()
    captured = capfd.readouterr()
    assert "overflow" not in (captured.out + captured.err).lower()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA renderer validation requires a GPU")
def test_batched_rotated_cameras_keep_individual_intrinsics():
    vertices = np.array([[-0.4, -0.2, 2.0], [0.2, -0.2, 2.2], [-0.1, 0.4, 2.1]])
    rotations = np.array(
        [
            [[0.8660254, -0.5, 0], [0.5, 0.8660254, 0], [0, 0, 1]],
            [[0.8660254, 0.5, 0], [-0.5, 0.8660254, 0], [0, 0, 1]],
        ]
    )
    translations = np.array([[0.1, -0.1, 0.4], [-0.2, 0.15, 0.8]])
    focal = np.array([[300, 400], [450, 270]])
    principal = np.array([[217, 190], [400, 220]])
    transforms = np.tile(np.eye(4), (2, 1, 1))
    transforms[:, :3, :3], transforms[:, :3, 3] = rotations, translations
    camera = {
        "w2c_RT": torch.tensor(transforms, device="cuda", dtype=torch.float32),
        "image_size": (641, 479),
        "focal_length": torch.tensor(focal, device="cuda", dtype=torch.float32),
        "princpt": torch.tensor(principal, device="cuda", dtype=torch.float32),
    }
    images = MeshRenderer(torch.tensor([[0, 1, 2]], device="cuda")).render_mesh(
        torch.tensor(vertices, device="cuda", dtype=torch.float32)[None].repeat(2, 1, 1), camera
    )
    y, x = np.mgrid[:479, :641]
    pixels = np.stack([x + 0.5, y + 0.5], axis=-1)
    for index, image in enumerate(images):
        points = vertices @ rotations[index].T + translations[index]
        projected = points[:, :2] / points[:, 2:] * focal[index] + principal[index]
        distances = []
        for start, end in zip(projected, np.roll(projected, -1, axis=0)):
            edge, offset = end - start, pixels - start
            distances.append(
                (edge[0] * offset[..., 1] - edge[1] * offset[..., 0]) / np.linalg.norm(edge)
            )
        signed_distance = np.min(distances, axis=0)
        foreground = image.max(axis=-1) > 0
        assert foreground.any()
        assert foreground[signed_distance > 1].all()
        assert not foreground[signed_distance < -1].any()
