import numpy as np
import pytest
from PIL import Image

from ehm_tracker.media import read_media, video_writer


@pytest.mark.parametrize("size", [(512, 512), (640, 360), (360, 640), (641, 479)])
def test_video_dimensions_frame_count_and_fps(tmp_path, size):
    path = tmp_path / "input.mp4"
    with video_writer(path, 23.976, size) as writer:
        for index in range(3):
            frame = np.full((size[1], size[0], 3), index * 40, dtype=np.uint8)
            writer.append_data(frame)
    frames, fps = read_media(path)
    assert len(frames) == 3
    assert all(frame.shape == (size[1], size[0], 3) for frame in frames)
    assert abs(fps - 23.976) < 0.01
    assert len(read_media(path, max_frames=2)[0]) == 2


def test_image_rgb_and_dimensions(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (71, 97), 120).save(path)
    frames, fps = read_media(path)
    assert frames[0].shape == (97, 71, 3)
    assert fps is None
