# What changed from EHM-Tracker

[EMOSH-Tracker](https://github.com/EastbeanZhang/EMOSH-Tracker) is the tracker developed for [EMOSH](https://arxiv.org/abs/2606.28026), building on the EHM representation and the [GUAVA / EHM-Tracker](https://github.com/Pixel-Talk/EHM-Tracker) workflow. EHM still combines SMPL-X body geometry with FLAME facial shape and expression; it is not a new parametric model.

## Changes described in the paper

The main paper, “Expressive Motion Representation,” contrasts GUAVA's cascaded facial-then-body fitting with EMOSH's unified optimization of body, hand, face, identity shape, and camera parameters. GVHMR, HaMeR, and TEASER supply coarse priors for body, hands, and face. DWPose and dense facial landmarks provide 2D guidance.

The confidence-aware Validity Gating mechanism suppresses unreliable guidance instead of forcing alignment to uncertain detections. Hand gating uses keypoint confidence and projected 3D hand/image overlap, with fallback priors for missing hands. Face gating masks facial losses under large rotations or back views. The objective also includes 3D hand alignment, parameter regularization, and temporal smoothness.

The supplementary “Tracking Comparison” section compares the trackers in the same full-body, half-body, and head-only scenarios. It reports fewer foot/hand deviations, fewer hallucinated visible arms when hands are occluded, and better stability in extreme close-ups. These are qualitative paper comparisons; they do not imply guaranteed accuracy for every input.

The supplementary table “Comparison of tracking efficiency on a 436-frame video” reports:

| Tracker | Total time | Throughput | Relative speed |
| --- | ---: | ---: | ---: |
| GUAVA Tracker | 424 s | 1.03 FPS | 1× |
| EMOSH Tracker | 223 s | 1.96 FPS | approximately 1.9× |

These values are quoted from the paper's evaluation. They are not a fresh benchmark of this packaged release, a universal speedup, or a real-time performance claim.

## Changes in this source release

- Configurable input/canvas width and height, with aspect-preserving fitting and explicit source/canvas camera transforms. Odd dimensions are preserved in videos and PNGs.
- One image/video entry point, plus a portrait/upper-body objective for cropped scenes. Initialization networks retain their own fixed-size crops.
- Source frame-rate and frame-index retention. Detection failures stop explicitly instead of silently dropping frames.
- An inference-only dependency set: matting, training/dataset preparation, LMDB output, and historical tracker variants are omitted.
- Reproducible model-file checks, installation instructions, licensed examples, and geometry/rendering/end-to-end validation.

Resolution support is an engineering extension in this release. The paper's square-resolution generation evaluation should not be read as an arbitrary-resolution tracker benchmark. See [validation.md](validation.md) for the tested release behavior and remaining limitations.
