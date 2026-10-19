# Runtime and outputs

`--inference-batch-size` controls feature extraction batches (default 4). `--optimization-batch-size` controls EHM fitting windows (default 32). Reduce these when memory is limited. GVHMR sees the complete clip; use `--max-frames` or shorter source clips for long inputs. Shape is fitted in the first optimization window and shared with subsequent windows. Temporal losses do not cross window boundaries. GVHMR uses static-camera initialization; the release does not recover a camera-motion-compensated world trajectory.

Each input writes a directory under `--output`:

```text
results/videos/full_body_motion/
├── metadata.json          # Completion marker, source dimensions, FPS and frame indices
├── optim_tracking.npy     # EHM motion, cameras and crop transforms per frame
├── id_share_params.npy    # Shared body/head shape and scale parameters
├── init_tracking.npy      # Initial estimates and detected keypoints
├── tracking.mp4           # Input | mesh | overlay, when --visualize is set
├── input/                 # Per-frame PNGs, when --save-images is set
├── mesh/
└── overlay/
```

Image previews are saved as `tracking.png`. An output is complete only when `metadata.json` contains `completed: true`. Detection or optimization failures stop the run and leave an `error.txt`; frames are not silently dropped or renumbered. Audio is not copied to preview videos. Inputs with variable frame timing are decoded into a constant-FPS preview using the decoder's reported rate.

`smplx_params.K_intrinsic` refers to the tracking canvas; `K_original` refers to the source image. Crop transforms `M_o2c-full` and `M_c2o-full` map between those coordinate systems. See [output conventions](outputs.md). NumPy dictionary files use pickle and should only be loaded from trusted runs.

Use a fresh output directory for each run. Input stems must be unique. `--max-frames 8 --steps 10` provides a quick execution check; use the default 1,251 steps to assess tracking quality.

## Memory usage

A **15-second portrait video**, with **450 frames at 30 FPS** and native **1920×1080** input and tracking dimensions, reached approximately **10.70 GiB sampled process GPU memory**. The run used `--head-only`, an inference batch size of four, and 32-frame fitting windows with 1,251 steps per window. It included initialization, fitting, visualization and per-frame PNG export.

PyTorch recorded **8.22 GiB allocated / 8.51 GiB reserved**. The process measurement also includes other CUDA allocations; its one-second sampling interval can miss brief peaks. These values describe this configuration, and are not a minimum GPU capacity. Leave headroom for different scenes, clip lengths and settings.

The result's `metadata.json` records allocator peaks under `gpu_memory_bytes`. To sample process memory while tracking runs:

```bash
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv --loop=1
```

## Limitations

Small faces and hands, occlusion, back views and motion blur can degrade estimates. Use `--head-only` for portraits and upper-body crops with unseen limbs. The body objective can inherit an incorrect arm pose in tightly cropped inputs. Fitting windows share identity, but a small pose discontinuity can remain at their boundaries.
