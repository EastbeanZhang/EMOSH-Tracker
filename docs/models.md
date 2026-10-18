# Model preparation

Paths below are relative to `--assets` (default: `assets/` at the repository root). Directory names are lowercase and case-sensitive.

```text
assets/
├── smplx/
│   ├── SMPLX_NEUTRAL_2020.npz
│   ├── SMPL-X__FLAME_vertex_ids.npy
│   ├── MANO_SMPLX_vertex_ids.pkl
│   ├── smplx_extra_joints.yaml
│   └── SMPLX_to_J14.pkl
├── flame/
│   ├── generic_model.pkl
│   ├── landmark_embedding.npy
│   ├── mediapipe_landmark_embedding.npz
│   ├── 203_landmark_embeding.npz
│   ├── l_eyelid.npy
│   └── r_eyelid.npy
├── mano/
│   ├── MANO_RIGHT.pkl
│   └── selected_hand_ver.npy
├── gvhmr/checkpoints/
│   ├── gvhmr/gvhmr_siga24_release.ckpt
│   ├── vitpose/vitpose-h-multi-coco.pth
│   ├── hmr2/epoch=10-step=25000.ckpt
│   ├── hmr2/smpl_mean_params.npz
│   └── body_models/smplx/SMPLX_NEUTRAL.npz
├── hamer/
│   ├── hamer_ckpts/model_config.yaml
│   ├── hamer_ckpts/checkpoints/hamer.ckpt
│   └── mano_mean_params.npz
├── teaser/Teaser.pt
├── lmk_203/landmark.onnx
├── mediapipe/face_landmarker.task
└── dwpose/
    ├── yolox_l.onnx
    └── dw-ll_ucoco_384.onnx
```

## Parametric models and EHM data

1. Obtain **SMPL-X 2020 neutral**, with the full shape basis, from the [official downloads](https://smpl-x.is.tue.mpg.de/download.php). Store it as `smplx/SMPLX_NEUTRAL_2020.npz`. The EHM decoder uses 300 body shape coefficients; a reduced 10-coefficient archive is insufficient. The same licensed neutral model is used by GVHMR: copy it to `gvhmr/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz`.
2. Download **FLAME 2020** from [FLAME](https://flame.is.tue.mpg.de/download.php) and place its neutral `generic_model.pkl` under `flame/`.
3. Download **MANO v1.2** from [MANO](https://mano.is.tue.mpg.de/download.php), and place `MANO_RIGHT.pkl` under `mano/`. Left-hand model weights are not required; the tracker mirrors right-hand predictions.
4. Run `python scripts/download_support.py`. It downloads the ten EHM landmark, joint and correspondence files listed in `assets/support_manifest.json`, checks their Git blob hashes, and uses a fixed [EHM-Tracker revision](https://github.com/Pixel-Talk/EHM-Tracker/tree/176f4d88a4fa9e73f00d102a8e924be5209b0d0c). It does not download licensed body/face/hand model weights.

Legacy FLAME/MANO pickles may refer to `chumpy`. The tested Python 3.10 environment pins NumPy below 1.24 to retain the aliases used by those legacy files. Install `chumpy==0.70` after NumPy with `--no-build-isolation` as shown in the README. Only load model pickles/checkpoints from trusted sources.

## Initialization networks

- **GVHMR:** use the [official installation page](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md) and its [checkpoint folder](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link). This tracker uses the released GVHMR checkpoint, ViTPose-H, and **HMR2.0a `epoch=10-step=25000.ckpt`**. It does not need DPVO, YOLOv8, SMPL gender models, or training data. `smpl_mean_params.npz` is the HMR2 decoder's standard initialization file; use the `data/smpl_mean_params.npz` from the [4DHumans model distribution](https://github.com/shubham-goel/4D-Humans#download-models).
- **HaMeR:** obtain the [official demo archive](https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz), also linked by the [official download script](https://github.com/geopavlakos/hamer/blob/main/fetch_demo_data.sh). Keep its `hamer_ckpts/model_config.yaml`, checkpoint, and `data/mano_mean_params.npz`; copy the latter to `hamer/mano_mean_params.npz`. EHM supplies the MANO geometry separately, so the inference-only HaMeR wrapper does not load its training losses or MANO decoder.
- **TEASER:** download [Teaser.pt](https://drive.google.com/file/d/1SjYuCfDf5ElxK3VNvcF7KPBR57EIRDSY/view) from the [official pretrained model folder](https://drive.google.com/drive/folders/1WhTjAZIQBCZqDRziu8_ZBtMC736K9T2A?usp=drive_link) and store it as `teaser/Teaser.pt`. The complete file is **360,042,145 bytes**. Its 974 `teaser_encoder.` state entries load strictly into the released MobileNet encoder with `timm==0.9.12`; a two-image forward pass has been verified. The loader uses `weights_only=True`.
- **203-point face landmarks:** download [landmark.onnx](https://drive.google.com/file/d/1Zu9Vj4f14l7RGD5I5G3iHsL5vEc9YkLW/view) from the same folder and store it as `lmk_203/landmark.onnx`. The complete file is **114,666,491 bytes**. It accepts one RGB `[1, 3, 224, 224]` float image in `[0, 1]`; the third output is `[1, 406]`, representing 203 normalized XY coordinates. The wrapper scales X by crop width and Y by crop height. Real inference and rectangular-batch output have been verified with `onnxruntime==1.20.1` on the CPU. Do not use an arbitrary 68/106/478-point detector as a replacement.
- **DWPose:** download `yolox_l.onnx` and `dw-ll_ucoco_384.onnx` from the [official ONNX branch](https://github.com/IDEA-Research/DWPose/tree/onnx). Both files must match the ONNX interface in that release. DWPose runs on the CPU to keep CUDA/cuDNN dependencies confined to PyTorch.
- **MediaPipe:** download [face_landmarker.task, version 1](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task), and place it under `mediapipe/`.

## Reuse existing downloads

The importer copies required files from extracted archives or an existing model directory; it does not modify source directories or overwrite existing destination files. Ambiguous filenames require manual placement. It rejects incomplete ZIP-format checkpoints before copying them.

```bash
python scripts/import_assets.py --source /path/to/extracted/models /path/to/existing/assets
python track.py --check-assets
```

The published v1 [EHM-Tracker checkpoint archive](https://drive.google.com/file/d/1g_4YKQvLSWo8yzYHgNstr91RCD4rne8p/view?usp=sharing) is an additional source for shared files, but it is not a complete v2 bundle. GVHMR and its feature weights must be prepared separately. Renaming a checkpoint does not convert its architecture.

## What is checked

`--check-assets` validates presence, nonempty files, and the integrity of ZIP-format `.pt`, `.ckpt`, and `.npz` archives. Network constructors validate the actual model interfaces at load time. A successful file check alone does not establish that tracking works.

Exact sizes, source links and SHA-256 hashes for these two downloads are recorded in [face_model_manifest.json](../assets/face_model_manifest.json). Check your files with:

```bash
sha256sum assets/teaser/Teaser.pt assets/lmk_203/landmark.onnx
```

Complete tracking was validated with all 26 required files in the clean environment documented in [validation.md](validation.md). The original research snapshot's truncated TEASER files were replaced with the complete official downloads; source files were preserved.

[Network checkpoint hashes and source URLs](../assets/network_model_manifest.json) record the six large initialization files used for validation. HaMeR and DWPose came from author-hosted files; the GVHMR files came from the listed public mirror of its distribution. Prefer the official download links above when available. Hashes identify tested files, not a separate license grant.

The previous EHM-Tracker uses an exported `pretrained/teaser/TEASER_lmk_2.onnx` model. This release loads the original PyTorch encoder checkpoint above; the old export is not a replacement for `Teaser.pt`.
