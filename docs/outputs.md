# Output conventions

The release keeps the research result names `optim_tracking.npy`, `id_share_params.npy`, and `init_tracking.npy`. These are NumPy dictionary containers, loaded with `np.load(path, allow_pickle=True).item()`. Only load trusted files.

A result contains one `frame_000000` entry per decoded source frame, in source order. Detection failures stop a run instead of removing frames. `metadata.json` records the source dimensions, tracking canvas, frame count, FPS, source indices, fitting mode, and completion status. Image runs have `fps: null`.

`elapsed_seconds` measures the run through the requested exports. `gpu_memory_bytes.peak_allocated` and `peak_reserved` record this process's PyTorch CUDA allocator peaks; they exclude CPU models and other processes on the GPU.

## Coordinates and cameras

All image sizes use `[width, height]`. NumPy images use `[height, width, channels]`, RGB. Pixel coordinates use X right and Y down. The camera looks along positive Z. Model geometry and camera translations use the parametric models' spatial units; pixel resizing does not rescale mesh depth.

`smplx_params` contains axis-angle `global_pose`, `body_pose`, `jaw_pose`, `left_hand_pose`, `right_hand_pose`, `eye_pose`, the world-to-camera transform `w2c_RT`, and pixel focal lengths `focal_px`, `focal_py`. `flame_params` contains expression and eyelid coefficients. Shared shape, head/hand scales, and joint offsets live in `id_share_params.npy`.

For a model point `X`, projection is:

```text
X_camera = w2c_RT @ [X, 1]
p_tracking ~ K_intrinsic @ X_camera[:3]
p_original ~ K_original @ X_camera[:3]
K_original = inverse(M_o2c-full) @ K_intrinsic
```

`K_intrinsic` and `K_original` have shape `[1, 3, 3]` in each frame. `full_crop.M_o2c-full` maps original pixels to the tracking canvas. `M_c2o-full` is its inverse. Head/hand crop matrices refer to the original image; compose them with `M_o2c-full` when moving landmarks to the canvas.

The tracking camera's principal point is the canvas center. The initializer estimates focal length from image geometry; it is not a calibrated physical camera. Original intrinsics are derived consistently from the fitted camera and image transform.

## Compatibility

The EHM pose/shape groups retain the research parameter layout. New camera matrices and `metadata.json` are additional fields. The release does not create the old image LMDB or training-dataset index. Consumers that assumed 512×512 crops, constant 30 FPS, or an LMDB must use the recorded dimensions/FPS and optional PNG exports instead. Dropping frames in downstream applications requires keeping the source frame-index mapping.

## Completion and previews

`metadata.json` with `completed: true` is written only after requested exports succeed. A failed run may contain intermediate arrays and `error.txt`, but has no completion marker. Existing output directories are not reused automatically.

`tracking.png` / `tracking.mp4` shows input, mesh and overlay side by side. Each panel has exactly the requested tracking dimensions. Odd dimensions use H.264 4:4:4; even dimensions use 4:2:0. Per-frame PNGs preserve exact canvas dimensions. Previews do not carry input audio, and variable-frame-rate timestamps are not retained.
