# From tracked videos to EMOSH training data

EMOSH-Tracker exports the RGB frames, motion, expression, identity and camera parameters consumed by [EMOSH](https://github.com/EastbeanZhang/EMOSH). Dataset packing and training run in the EMOSH environment; no training packages are needed in the Tracker environment.

## Track

After completing the [installation](installation.md) and [model preparation](models.md), run in the Tracker repository:

```bash
python track.py --input /path/to/video.mp4 --output results \
  --visualize --save-images
```

This keeps the video's original dimensions. Add `--head-only` for portraits or cropped upper bodies. To choose another fitting canvas, use the optional `--target-size WIDTH HEIGHT`.

Inspect `results/video/tracking.mp4`. Keep `input/`, `optim_tracking.npy`, `id_share_params.npy` and `metadata.json` together.

## Pack and train

Install [EMOSH](https://github.com/EastbeanZhang/EMOSH/blob/main/docs/installation.md), activate its environment and run from its repository:

```bash
python prepare_dataset.py --tracking /path/to/tracker/results/video \
  --output data/train --id clip_0001
python train.py --dataset data/train --model-path models/wan-base \
  --assets assets --output outputs/train --checkpoint-offload
```

The converter uses the original fixed training prompt. Repeat the packing command with a unique ID for each video; add `--split valid` for held-out records. The standard training clip contains 77 frames. Keep at least nine consecutive frames for temporal conditioning.

EMOSH accepts native dimensions, Wan-Animate pixel area budgets and custom canvases. Its loader transforms RGB frames, camera intrinsics and crop matrices together, so changing the training size does not require tracking the video again.

See the [complete data workflow and format](https://github.com/EastbeanZhang/EMOSH/blob/main/docs/data.md) and [training and inference commands](https://github.com/EastbeanZhang/EMOSH/blob/main/docs/training.md).
