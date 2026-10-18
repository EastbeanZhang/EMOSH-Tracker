# Third-party notices

The root Apache-2.0 license covers the project-authored tracker integration. It does not replace the terms of embedded components, pretrained weights, parametric models, or media.

| Component | Source | Terms / retained notices |
| --- | --- | --- |
| EHM and tracking integration | [EHM-Tracker](https://github.com/Pixel-Talk/EHM-Tracker) and the EMOSH research implementation | Root Apache-2.0 for project-authored integration; embedded model code retains its headers |
| GVHMR, its HMR2/ViTPose helpers and body-model support data | [GVHMR](https://github.com/zju3dv/GVHMR) | [Research/non-commercial license](modules/gvhmr/LICENSE); upstream acknowledgements and embedded notices also apply |
| HaMeR inference components | [HaMeR](https://github.com/geopavlakos/hamer) | [MIT](modules/hamer/LICENSE) |
| DWPose ONNX helpers | [DWPose](https://github.com/IDEA-Research/DWPose) | [Apache-2.0, including OpenMMLab notices](modules/dwpose/LICENSE) |
| TEASER encoder | [TEASER](https://github.com/Pixel-Talk/TEASER) | Author-provided encoder from the EMOSH implementation; model distribution terms apply independently |
| SMPL-X / FLAME / MANO model code and data | [SMPL-X](https://smpl-x.is.tue.mpg.de/), [FLAME](https://flame.is.tue.mpg.de/), [MANO](https://mano.is.tue.mpg.de/) | Retained MPG notices in `modules/human_model`; separately registered model agreements apply |
| Quaternion helpers within GVHMR | [QuaterNet](https://github.com/facebookresearch/QuaterNet) | Facebook copyright notices retained; [upstream license](https://github.com/facebookresearch/QuaterNet/blob/main/LICENSE) |
| MediaPipe, ONNX Runtime, timm, PyTorch3D and other installed libraries | Their respective upstream distributions | Installed under their own licenses; not relicensed by this repository |

Inference adaptation changes include removal of training/demo paths, explicit asset roots, lazy renderer imports, model-loading checks, consistent coordinate handling, and unused-import cleanup. Files under `modules/` are adapted source rather than unmodified upstream snapshots.

The ten auxiliary EHM files downloaded by `scripts/download_support.py` come from the fixed EHM-Tracker revision in the manifest. The script validates upstream Git blob hashes. These files remain subject to their original source/model terms.

Example inputs and paper figures retain their original media rights. See [asset provenance](assets/PROVENANCE.md). Pretrained SMPL-X, FLAME, MANO, HaMeR, GVHMR and TEASER weights are excluded from the source distribution.
