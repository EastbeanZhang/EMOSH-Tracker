# Validation

Validated on 18 October 2026 in a clean Linux environment with no inherited system site packages: Python 3.10.18, PyTorch 2.9.1+cu130, CUDA toolkit 13.0, and PyTorch3D 0.7.9 (`33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`). Exact installed versions are recorded in [tested-environment.txt](tested-environment.txt).

## Regression and model checks

`python -m pytest -q tests` passed **105 tests with no skips** in 6.74 seconds. One warning comes from a legacy model pickle referring to SciPy's deprecated sparse namespace. Ruff checks and formatting also passed.

The suite covers image fitting and inverse transforms across 36 source/canvas combinations, transfer of camera intrinsics, body/head projection and gradients, short-sequence smoothing, frame-window coverage, metric hand depth under resizing, hand visibility gating, singleton body initialization, grayscale decoding, exact odd-sized video output, frame counts, and frame rates. Fourteen CUDA rendering cases check analytic pinhole coverage and dense-mesh rasterization, including rotated cameras with separate intrinsics for each batch item. A real EHM decoder test checks 10,475 vertices, 203 facial landmarks, and shape/expression gradients.

All 26 required model files load during real tracking. Complete official TEASER downloads replaced truncated copies in the supplied snapshot; [file hashes](../assets/face_model_manifest.json) identify the tested versions. Initialization checkpoints and their sources are recorded in [network_model_manifest.json](../assets/network_model_manifest.json). Licensed geometry was recovered from unchanged buffers in the author's GUAVA checkpoint for these private tests; [geometry_asset_validation.json](geometry_asset_validation.json) records the exact array checks. Model weights are excluded from the release. Users should obtain licensed models through [the model instructions](models.md).

## End-to-end tracking

The original five research samples passed **30 complete runs**: three images and two eight-frame video excerpts, each at 512×512, 640×360, 360×640, 832×480, 641×479, and 1920×1080. Every run used the default **1,251 fitting steps**. Each result passed finite-parameter, camera-transform, frame-coverage, FPS, and exact-export-size checks.

The final bundled examples also passed **42/42 runs**, with no failures. These use the public-media hashes in [the example manifest](../assets/examples/manifest.json): ordinary body fitting for the full-body cases, and `--head-only` for the portrait and cropped upper-body cases. The four full videos each passed at **48 frames**, preserving the source decoder rates: 30 FPS for both full-body videos, 25 FPS for the upper-body video, and 23.98 FPS for the portrait; their default 32-frame fitting window was followed by a 16-frame final window. The second full-body demonstration tracks the other performer in the same licensed dance recording, with a separately cropped 384×640 input. All three images also passed at their native dimensions (736×1024, 320×240, and 545×660).

[resolution_validation.json](resolution_validation.json) records all 49 final runs, including file hashes, modes, dimensions, frame counts, frame rates, fitting steps, elapsed time, and allocator peaks. Representative input/mesh/overlay frames were reviewed for all 42 matrix cases. Native video previews were also inspected around the fitting-window boundary. Face, visible-hand, and body alignment remained consistent across the tested canvases. Unobserved limbs remain ambiguous: the cropped upper-body example produced an incorrect extra visible arm under ordinary body fitting, so its documented command uses the existing portrait objective with out-of-frame hand priors.

Peak allocated PyTorch GPU memory ranged from **7.93 to 8.23 GiB**; peak reserved memory ranged from **8.16 to 8.50 GiB**. The largest values were 8,833,624,576 and 9,130,999,808 bytes, recorded for the two full-body video excerpts at a 1920×1080 tracking canvas. Each excerpt contained eight frames at 30 FPS (0.27 s), with 1,251 optimization steps. Their source dimensions were 460×640 and 384×640; the canvas used aspect-preserving scaling and padding. The four native-resolution full-video tests each contained 48 frames (1.60–2.00 s), using fitting windows of 32 and 16 frames. These are process-local allocator measurements for the tested images and clips, not a universal VRAM minimum. They exclude other processes and non-PyTorch allocations. Timing was collected under concurrent workloads and is not a speed benchmark.

The renderer uses unbinned rasterization to avoid dropping triangles when a dense body mesh exceeds coarse-bin capacity. Previews were inspected against the input after that change.

## Long video memory test

A separate **15.00-second, 450-frame test at 30 FPS** used native **1920×1080 input and tracking dimensions**, with `--head-only`, an inference batch size of four, and the default **32-frame fitting window / 1,251 steps per window**. All 15 windows completed: fourteen windows of 32 frames and a final two-frame window. Initialization processed the complete clip; measurement included model loading, fitting, visualization and per-frame export in a fresh process.

| Measurement | Peak |
| --- | ---: |
| PyTorch allocated | **8.22 GiB** |
| PyTorch reserved | **8.51 GiB** |
| Sampled process memory, including other CUDA allocations | approximately **10.70 GiB** |

All **450 frames** passed finite-parameter, positive-focal-length, camera-transform, frame-coverage, 30 FPS and exact-export-dimension checks. Input, mesh and overlay PNGs were verified at 1920×1080; the three-panel video was verified at 5760×1080. [Machine-readable results and hashes](long_clip_memory.json) record the input, output, measurements and sampling conditions.

The input is a continuous 15-second excerpt beginning at 20.02 seconds of [NASA's Kelsey Young interview](https://svs.gsfc.nasa.gov/14934), retaining the source's native dimensions and resampling approximately 23.98 FPS to 30 FPS. The source is public-domain NASA media. The test uses the portrait objective; other scene types and settings can have different peaks. PyTorch allocator peaks exclude other processes and non-PyTorch allocations. Process-memory sampling includes non-PyTorch allocations but its approximately one-second interval can miss brief peaks. Keep additional GPU memory headroom. Runtime under concurrent workloads is not a speed benchmark.

To reproduce in the documented environment:

```bash
curl -L 'https://svs.gsfc.nasa.gov/vis/a010000/a014900/a014934/Kelsey_Young__SOTs_Artemis_II_1.webm' -o kelsey.webm
tracker_ffmpeg="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
"$tracker_ffmpeg" -ss 20.02 -i kelsey.webm -an -vf fps=30 \
  -frames:v 450 -c:v libx264 -crf 18 -pix_fmt yuv420p portrait_15s_30fps.mp4
python track.py --input portrait_15s_30fps.mp4 --output results/memory_15s \
  --target-size 1920 1080 --head-only --steps 1251 \
  --optimization-batch-size 32 --inference-batch-size 4 --visualize --save-images
python scripts/verify_result.py results/memory_15s/portrait_15s_30fps
```

The result's `metadata.json` records `gpu_memory_bytes.peak_allocated` and `peak_reserved` in bytes. To sample total process GPU memory, run the following in a second terminal while tracking is active:

```bash
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv --loop=1
```

Record the maximum for the tracker Python process's PID. Do not sum other processes; the reported MiB value divided by 1,024 gives GiB. A fresh output directory and a separate process keep the measurement independent of earlier clips.

## Reproduce

```bash
python -m pip install -r requirements-dev.txt
python track.py --check-assets
python -m pytest -q tests
ruff check .
ruff format --check ehm_tracker utils modules tests scripts track.py
python scripts/validate_resolutions.py --max-frames 8 --steps 1251 --fail-fast
bash scripts/run_examples.sh
```

The matrix validates its exports automatically. To check a manually run case, include `--visualize --save-images`, then run `python scripts/verify_result.py /path/to/result/input_stem`.

Successful execution and consistent camera geometry do not guarantee accurate motion for every image. Small faces/hands, occlusion, back views, motion blur, and ambiguous body shape can still degrade estimates. Fitting windows share identity but have no temporal loss across their boundaries; a small pose discontinuity can remain there. Long-clip memory use also depends on the complete-sequence body initializer. No calibrated tracking-accuracy or speed benchmark is claimed.
