#!/usr/bin/env bash
set -euo pipefail
python track.py --input assets/examples/full_body.jpg assets/examples/full_body_motion.mp4 --output results/examples --visualize
python track.py --input assets/examples/head.jpg assets/examples/head_motion.mp4 assets/examples/upper_body.jpg assets/examples/upper_body_motion.mp4 --output results/examples --head-only --visualize
