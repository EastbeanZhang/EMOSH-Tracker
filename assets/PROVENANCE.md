# Asset provenance

## Example inputs and generated previews

| Files | Author and original source | Terms |
| --- | --- | --- |
| `examples/full_body.jpg`, `examples/full_body_motion.mp4`, `examples/full_body_motion_2.mp4` | Xiangliangzai, [WikiNgigel — Ragam Gerak Humbak Moloh (Tari Bedana)](https://commons.wikimedia.org/wiki/File:WikiNgigel_%E2%80%93_Ragam_Gerak_Humbak_Moloh_(Tari_Bedana)_%E2%80%93_Bandar_Lampung_(2026).webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `examples/upper_body.jpg`, `examples/upper_body_motion.mp4` | E5SUON, [Hello1.ogv: “Hello” in American Sign Language](https://commons.wikimedia.org/wiki/File:Hello1.ogv) | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
| `examples/head_motion.mp4` | NASA’s Goddard Space Flight Center, [Dr. Kelsey Young / Artemis II Lunar Science Lead](https://svs.gsfc.nasa.gov/14934) | [NASA media guidelines / public domain](https://www.nasa.gov/nasa-brand-center/images-and-media/) |
| `examples/head.jpg` | Robert Markowitz / NASA, [Commander Eileen Collins](https://commons.wikimedia.org/wiki/File:Commander_Eileen_Collins_-_GPN-2000-001177.jpg) | [Public domain — US government work](https://commons.wikimedia.org/wiki/Template:PD-USGov-NASA) |

The dance examples crop the two performers separately from the original recording. The full-body image is 736×1024; its video is uniformly resized to 460×640. The upper-body examples retain 320×240. The portrait is cropped and uniformly resized to 545×660. Images use source frame 12 (zero-based) where applicable. The dance video uses source frames 210–304 with stride 2, resampling 60 FPS to 30 FPS; the sign-language video uses the first 48 frames at its original 25 FPS. Both have 48 frames and omit audio. The second full-body video uses the same source frames and 30 FPS, with the other performer cropped at [1000,60,1600,1060] and uniformly resized to 384×640. The portrait video uses frames 480–527 of NASA’s Kelsey Young interview, cropped to [580,40,1380,1000] and uniformly resized to 400×480; its 48 frames retain the reported 23.98 FPS and omit audio. Exact crop rectangles, changes, original download links, and source/derived SHA-256 hashes are in [examples/manifest.json](examples/manifest.json).

Files under `previews/` are tracker-generated input/mesh/overlay illustrations of these examples, resized or frame-sampled for display. The full-body previews retain CC BY-SA 4.0; upper-body previews retain CC BY-SA 3.0; the NASA input remains public domain. The `gallery_*.gif` tiles center input and mesh horizontally and vertically within equal-sized panels, with labels and aspect-preserving black padding; their links open the full three-panel animations. These media licenses are separate from the root code license. Attribution does not imply endorsement by the creators, performers, or NASA.

These are small demonstration cases, not a benchmark dataset. They replace samples from the research snapshot whose redistribution terms were undocumented.

## Paper figures

`docs/motion_tracker_pipeline.pdf` and `docs/show_tracking.pdf` are unchanged files extracted from `figs/` in the [EMOSH arXiv v1 source archive](https://arxiv.org/src/2606.28026v1). Matching PNGs are rendered previews. The pipeline is Fig. 3(a); the result montage is Fig. 4. The montage shows paper results, distinct from the new release previews.

## Model support data

`support_manifest.json` pins URLs and upstream Git blob hashes for ten EHM joint, correspondence, and landmark files. Downloaded files and licensed model weights are excluded from Git. Third-party GVHMR helper tensors remain in their source package under its research/non-commercial terms.
