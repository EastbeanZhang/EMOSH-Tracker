"""Run the actual CUDA tracker over the release validation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES = [(512, 512), (640, 360), (360, 640), (832, 480), (641, 479), (1920, 1080)]
CASES = [
    ("full_body.jpg", False),
    ("upper_body.jpg", True),
    ("head.jpg", True),
    ("head_motion.mp4", True),
    ("full_body_motion.mp4", False),
    ("full_body_motion_2.mp4", False),
    ("upper_body_motion.mp4", True),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--examples", type=Path, default=ROOT / "assets/examples")
    parser.add_argument("--output", type=Path, default=ROOT / "validation_runs")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1251)
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed case")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for width, height in SIZES:
        for filename, head in CASES:
            output = args.output / f"{width}x{height}"
            source = args.examples / filename
            source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            cmd = [
                sys.executable,
                str(ROOT / "track.py"),
                "--input",
                str(source.resolve()),
                "--assets",
                str(args.assets.resolve()),
                "--output",
                str(output.resolve()),
                "--target-size",
                str(width),
                str(height),
                "--max-frames",
                str(args.max_frames),
                "--steps",
                str(args.steps),
                "--visualize",
                "--save-images",
            ]
            if head:
                cmd.append("--head-only")
            started = time.monotonic()
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            log = args.output / f"{width}x{height}_{Path(filename).stem}.log"
            log.write_text(result.stdout + "\n" + result.stderr)
            item = {
                "case": filename,
                "input_sha256": source_sha256,
                "size": [width, height],
                "steps": args.steps,
                "returncode": result.returncode,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "log": log.name,
            }
            if result.returncode == 0:
                # Verification errors also make this matrix fail.
                verified = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/verify_result.py"),
                        str(output / Path(filename).stem),
                    ],
                    capture_output=True,
                    text=True,
                )
                item["verification_returncode"] = verified.returncode
                metadata_path = output / Path(filename).stem / "metadata.json"
                if metadata_path.is_file():
                    metadata = json.loads(metadata_path.read_text())
                    item["gpu_memory_bytes"] = metadata.get("gpu_memory_bytes")
                    item["source_size"] = metadata["source_size"]
                    item["frame_count"] = metadata["frame_count"]
                    item["fps"] = metadata["fps"]
                with log.open("a") as stream:
                    stream.write("\n" + verified.stdout + verified.stderr)
            report.append(item)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"{filename} {width}x{height}: exit {item.get('verification_returncode', item['returncode'])}",
                flush=True,
            )
            if args.fail_fast and (item["returncode"] or item.get("verification_returncode", 0)):
                return 1
    return int(any(item["returncode"] or item.get("verification_returncode", 0) for item in report))


if __name__ == "__main__":
    raise SystemExit(main())
