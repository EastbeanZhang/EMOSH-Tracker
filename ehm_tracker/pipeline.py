"""Image/video tracking with explicit frame and camera metadata."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ehm_tracker.features import FeatureExtractor
from ehm_tracker.geometry import fit_transform
from ehm_tracker.media import read_media, video_writer
from ehm_tracker.models import load_initializers
from ehm_tracker.parameters import (
    assert_finite,
    to_numpy_array_recursive,
    to_tensor_recursive,
)


class Tracker(FeatureExtractor):
    def __init__(self, args, image_size):
        self.args, self.device = args, args.device
        self.target_size, self.body_size = image_size, (256, 256)
        self.__dict__.update(load_initializers(args.device, image_size, args.inference_batch_size))
        from ehm_tracker.head_optimizer import HeadOptimizer
        from ehm_tracker.optimizer import BodyOptimizer

        optimizer_type = HeadOptimizer if args.head_only else BodyOptimizer
        config = SimpleNamespace(target_size=image_size, steps=args.steps, device=args.device)
        self.ehm_tracker = optimizer_type(config, str(args.assets), device=args.device)

    def close(self):
        self.lmkmp_model.close()

    def initialize(self, frames):
        images, initial, boxes = {}, {}, []
        for index, image in enumerate(tqdm(frames, desc="Detect and crop")):
            key = f"frame_{index:06d}"
            height, width = image.shape[:2]
            cropped, parameters = self.crop_images(
                image, box=np.array([0, 0, width, height], dtype=np.float32)
            )
            if cropped is None:
                raise RuntimeError(
                    f"No person detected in source frame {index}; no frames were dropped. Try a clip with one visible subject."
                )
            images[key], initial[key] = cropped, parameters
            boxes.append(parameters["full_crop"]["bbx_xys"])
        full = [value["full_crop_image"] for value in images.values()]
        heads = [value["head_crop_image"] for value in images.values()]
        left = [value["left_hand_crop_image"] for value in images.values()]
        right = [value["right_hand_crop_image"] for value in images.values()]
        body, boxes, face, left_params, right_params, landmarks = self.extract_body_pose(
            boxes, full, heads, left, right
        )
        identities = {}
        for index, key in enumerate(initial):
            initial[key]["full_crop"]["bbx_xys"] = boxes[index].cpu().numpy()
            parameters, identity = self.extract_other_pose(
                key,
                index,
                heads[index],
                left[index],
                right[index],
                body,
                face,
                left_params,
                right_params,
                landmarks,
            )
            initial[key].update(parameters)
            for name, value in identity.items():
                identities.setdefault(name, []).append(to_numpy_array_recursive(value))
        identity = {
            name: np.concatenate(value, axis=0).mean(axis=0, keepdims=True)
            for name, value in identities.items()
        }
        identity.update(
            head_scale=np.ones((1, 1, 3), dtype=np.float32),
            hand_scale=np.ones((1, 2, 3), dtype=np.float32),
            joints_offset=np.zeros((1, 55, 3), dtype=np.float32),
        )
        return images, to_numpy_array_recursive(initial), identity

    def optimize(self, initial, identity):
        keys = list(initial)
        results = {}
        # The body initializer sees the complete clip; EHM fitting uses bounded windows.
        for start in range(0, len(keys), self.args.optimization_batch_size):
            batch_keys = keys[start : start + self.args.optimization_batch_size]
            batch = torch.utils.data.default_collate([initial[key] for key in batch_keys])
            tracked, identity = self.ehm_tracker.tracking(
                batch,
                identity,
                batch_keys,
                batch_idx=start // self.args.optimization_batch_size,
            )
            assert_finite(tracked)
            assert_finite(identity)
            results.update(tracked)
        return results, identity

    @torch.no_grad()
    def visualize(self, images, results, identity, output, fps):
        identity = to_tensor_recursive(identity, device=self.device)
        width, height = self.target_size
        writer_context = (
            video_writer(output / "tracking.mp4", fps, (width * 3, height)) if fps else None
        )
        writer = writer_context.__enter__() if writer_context else None
        try:
            for index, (key, result) in enumerate(tqdm(results.items(), desc="Render")):
                params = to_tensor_recursive(result, device=self.device)
                camera = {
                    "w2c_RT": params["smplx_params"]["w2c_RT"],
                    "image_size": self.target_size,
                    "focal_length": torch.cat(
                        [
                            params["smplx_params"]["focal_px"],
                            params["smplx_params"]["focal_py"],
                        ],
                        dim=-1,
                    ),
                    "princpt": [width / 2, height / 2],
                }
                vertices = self.ehm_tracker.ehm(
                    params["smplx_params"], params["flame_params"], identity
                )["vertices"]
                mesh = self.ehm_tracker.renderer.render_mesh(vertices, camera)[0]
                source = images[key]["full_crop_image"]
                overlay = ((source.astype(np.float32) + mesh) / 2).round().astype(np.uint8)
                preview = np.concatenate([source, mesh, overlay], axis=1)
                if writer:
                    writer.append_data(preview)
                else:
                    Image.fromarray(preview).save(output / "tracking.png")
                if self.args.save_images:
                    for name, frame in [
                        ("input", source),
                        ("mesh", mesh),
                        ("overlay", overlay),
                    ]:
                        folder = output / name
                        folder.mkdir(exist_ok=True)
                        Image.fromarray(frame).save(folder / f"{index:06d}.png")
        finally:
            if writer_context:
                writer_context.__exit__(None, None, None)


def track_file(path, args):
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(args.device)
    frames, fps = read_media(path, args.max_frames)
    source_size = frames[0].shape[1::-1]
    size = tuple(args.target_size or source_size)
    destination = args.output / Path(path).stem
    if destination.exists():
        raise FileExistsError(
            f"Output already exists: {destination}; choose a new output directory"
        )
    destination.mkdir(parents=True)
    tracker = None
    try:
        tracker = Tracker(args, size)
        images, initial, identity = tracker.initialize(frames)
        results, identity = tracker.optimize(initial, identity)
        inverse = np.linalg.inv(fit_transform(source_size, size))
        for result in results.values():
            camera = result["smplx_params"]
            intrinsics = np.array(
                [
                    [camera["focal_px"].item(), 0, size[0] / 2],
                    [0, camera["focal_py"].item(), size[1] / 2],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            camera["K_intrinsic"] = intrinsics[None]
            camera["K_original"] = (inverse @ intrinsics)[None]
        assert_finite(results)
        np.save(destination / "optim_tracking.npy", results)
        np.save(destination / "id_share_params.npy", identity)
        np.save(destination / "init_tracking.npy", initial)
        if args.visualize:
            tracker.visualize(images, results, identity, destination, fps)
        metadata = {
            "format_version": 2,
            "input": Path(path).name,
            "source_size": list(source_size),
            "tracking_size": list(size),
            "fps": fps,
            "frame_count": len(frames),
            "source_frame_indices": list(range(len(frames))),
            "frames_keys": list(results),
            "mode": "head" if args.head_only else "body",
            "steps": args.steps,
            "optimization_batch_size": args.optimization_batch_size,
            "completed": True,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "gpu_memory_bytes": {
                "peak_allocated": torch.cuda.max_memory_allocated(args.device),
                "peak_reserved": torch.cuda.max_memory_reserved(args.device),
            },
        }
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return destination
    except Exception as error:
        (destination / "error.txt").write_text(f"{type(error).__name__}: {error}\n")
        raise
    finally:
        if tracker is not None:
            tracker.close()
        torch.cuda.empty_cache()
