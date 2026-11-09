# Frozen detector integration record — 9 November 2026

Both official source repositories and their published checkpoints have been
obtained locally. The adapter in `run_frozen_detector_pilot.py` ran each frozen
detector on the same 20 **development-only pristine working PNGs**. This is a
pipeline smoke test, not a benchmark of local AI edits or detector accuracy.
The 1600-pixel upright PNG conversion is logged in
`pilot_working_png/working_image_manifest.csv`. Do not interpret the values in
`data/pilot_pristine_scores.csv` as test results.

## CNNDetection (Wang et al.; often called CNNSpot)

- Official repository: <https://github.com/PeterWang512/CNNDetection>
- Local source commit: `ea0b5622365e3a9cd31d1b54b6b5971131a839ab`.
- Official `blur_jpg_prob0.5.pth` checkpoint from the repository's
  `weights/download_weights.sh`; SHA-256
  `3533f3e58a3b82215f21197f2faf4e99b911ec42a4a6e2d7823e04d5f3fd245a`.
- Adapter reproduces default `demo.py` input: RGB conversion, **no detector
  crop or resize**, `ToTensor`, ImageNet mean/std normalisation. The common
  working PNG was previously resized to a maximum side of 1600 pixels; that
  conversion is separate and must be recorded in any final benchmark.
- Output is `sigmoid(model(image))`; higher means more likely synthetic.

## UniversalFakeDetect (Ojha et al.; frozen CLIP feature detector)

- Official repository: <https://github.com/WisconsinAIVision/UniversalFakeDetect>
- Local source commit: `030495aea3300a8b54c0ec37ec7fe1dd7e63c619`.
- Official CLIP `ViT-L/14` backbone from the URL embedded in the repository;
  SHA-256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.
- Repository `pretrained_weights/fc_weights.pth` classifier head; SHA-256
  `477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847`.
- Adapter reproduces `validate.py` input: RGB conversion,
  `CenterCrop(224)`, `ToTensor`, CLIP mean/std normalisation. Its validation
  transform does **not** resize before cropping. The model-loader's separate
  CLIP transform is not the transform used by `validate.py`.
- The entire supplied backbone and head are frozen; output is sigmoid of the
  classifier logit, with higher meaning more likely synthetic.

## Verification and remaining gates

- `data/pilot_pristine_scores.csv` has 40 records: 20 per detector, exactly
  matched by root ID. Both official checkpoints loaded strictly. Scores are
  finite and within [0,1]. Run metadata and hashes are in
  `data/pilot_cnn_run.json` and `data/pilot_clip_run.json`.
- The adapter changes device routing from the repositories' hard-coded CUDA
  to Apple MPS. It does not change model weights or the chosen official input
  transforms. Run `USE_TF=0 USE_FLAX=0 .venv/bin/python
  run_frozen_detector_pilot.py --detector cnn --device mps` and similarly for
  `--detector clip` to reproduce the development-only smoke test.
- Final analysis still requires accepted AI-edited images and operation masks,
  independently defined foreground masks, confirmed scene groups, and frozen
  root-disjoint splits. Thresholds must be derived from pristine validation
  controls; final test images must not select preprocessing or thresholds.
