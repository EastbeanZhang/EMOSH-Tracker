"""Command-line interface for EMOSH-Tracker."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ehm_tracker.assets import asset_path, check_assets
from ehm_tracker.geometry import validate_size


def positive_integer(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="EMOSH-Tracker: expressive body, hand and face tracking"
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        help="Images, videos, or directories (one subject per input)",
    )
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument(
        "--assets",
        type=Path,
        default=asset_path(),
        help="Root directory of downloaded model files",
    )
    parser.add_argument(
        "--check-assets",
        action="store_true",
        help="Check required files without loading models",
    )
    parser.add_argument(
        "--target-size",
        type=positive_integer,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Fit full input into this canvas; default: original size",
    )
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="Use portrait/upper-body fitting for cropped or unseen limbs",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Save input / mesh / overlay preview"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Also save per-frame PNGs; implies --visualize",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device; CUDA_VISIBLE_DEVICES can select a physical GPU",
    )
    parser.add_argument(
        "--steps",
        type=positive_integer,
        default=1251,
        help="Optimization steps per fitting window",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=positive_integer,
        default=4,
        help="Head, hand and body feature batch size",
    )
    parser.add_argument(
        "--optimization-batch-size",
        type=positive_integer,
        default=32,
        help="Frames per EHM fitting window",
    )
    parser.add_argument(
        "--max-frames",
        type=positive_integer,
        help="Process at most this many video frames",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.assets = args.assets.expanduser().resolve()
    os.environ["EHM_ASSET_DIR"] = str(args.assets)
    if args.target_size:
        validate_size(args.target_size)
    try:
        count = check_assets(args.assets)
        if args.check_assets:
            print(f"Found all {count} required files under {args.assets}")
            return 0
        if not args.input:
            parser.error("--input is required unless --check-assets is used")
        from ehm_tracker.media import IMAGE_SUFFIXES, VIDEO_SUFFIXES

        paths = []
        for item in args.input:
            if item.is_dir():
                paths.extend(
                    p
                    for p in sorted(item.iterdir())
                    if p.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES
                )
            elif item.is_file():
                paths.append(item)
            else:
                raise FileNotFoundError(f"Input does not exist: {item}")
        paths = sorted(set(p.resolve() for p in paths))
        if not paths:
            raise ValueError("No supported inputs found")
        if len({p.stem for p in paths}) != len(paths):
            raise ValueError("Input stems must be unique within a run to avoid output collisions")
        import torch

        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(
                "Tracking requires a CUDA GPU; geometry and media tests run on a CPU"
            )
        torch.cuda.set_device(args.device)
        args.save_images = bool(args.save_images)
        args.visualize |= args.save_images
        from ehm_tracker.pipeline import track_file

        for path in paths:
            print(f"Saved {track_file(path, args)}")
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        RuntimeError,
        FloatingPointError,
    ) as error:
        parser.exit(1, f"{error}\n")
    return 0
