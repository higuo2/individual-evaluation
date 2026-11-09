# Verified status — 9 November 2026

## Completed locally

- Read the two supplied ZIP archives without modifying them. There are 100
  readable photographs in `14pro_100张.zip` and 200 in `16pro_200张.zip`.
  The apparent extra ZIP entries are macOS `__MACOSX/._*` resource files.
- EXIF model values match iPhone 14 Pro for all 100 and iPhone 16 Pro for all
  200. No image-decoding errors occurred. The inventory did not export GPS.
- No identical SHA-256 photo files and no perceptual-hash pairs at Hamming
  distance <= 6 were found. This does **not** establish scene independence.
- A contact sheet for all 300 photos was produced. Visual review flagged 25
  related-scene pairs, conservatively merged into 15 multi-photo candidate
  groups. The owner has not yet confirmed those relationships.
- A provisional 180/60/60 development/validation/test partition puts each
  candidate group in one split. Its audit currently finds no cross-split
  candidate group or identical hash. The partition is **not frozen**.
- Twenty development photos were copied byte-for-byte into `pilot_originals/`
  for a workflow pilot. They were selected for convenience from thumbnails;
  foreground suitability has not yet been established.
- The same 20 were converted to upright, 1600-pixel-long-side PNG working
  images for development-only mask authoring. The conversion and dimensions
  are logged in `pilot_working_png/working_image_manifest.csv`.
- A local browser mask painter and a mask-first inpainting pilot protocol are
  available. No foreground mask produced by the painter has been reviewed.
- Area and location measurement, sample geometry validation, split audit,
  validation-threshold analysis, area AUROC, root bootstrap, and paired
  location summary code were written. Eight tests using synthetic fixtures
  pass. These tests verify code behaviour, not detector performance.
- Diffusers 0.35.2 and a pinned Stable Diffusion v1.5 inpainting checkpoint
  were installed locally. Apple MPS runs in an approved local process. Five
  development-only generation calls were attempted on two photos. The first
  returned wrong geometry and a black image. The other four had correct
  geometry but did not produce the requested pineapple, traffic cone, or
  pavement arrow. The two uncropped outputs changed many pixels outside the
  mask. A documented local overlay or the crop option preserved unmasked
  pixels, but did not make the semantic edits convincing. **All five attempts
  are rejected as benchmark samples.** The trial record and available
  inputs, outputs, and metadata are in `pilot_trial/`.
- Official CNNDetection and UniversalFakeDetect repositories, pretrained
  weights, and SHA-256 identities are recorded in `DETECTORS.md`. Both frozen
  detectors successfully produced scores on the same 20 development-only
  pristine working PNGs. The 40 inference records are in
  `data/pilot_pristine_scores.csv`. This confirms that inference runs; it is
  **not** a performance estimate or an AI-edit result.

## Not yet available or verified

- No accepted foreground masks have been annotated for these photos. A quick
  rectangle-based segmentation experiment on two originals spilled into
  background regions, so its draft masks were rejected and removed.
- No **accepted** locally AI-edited derivative exists. Trial operation masks
  are present but are not part of a research evaluation dataset.
- No detector has been evaluated on an **accepted** AI-edited image or a
  held-out test set. No local-edit performance claim can be made.
- The final scene grouping, frozen root-disjoint partition, editor/export
  procedure, mask alignment, and full sample count remain open.

## Immediate next gate

Qualify an editor that produces a **visible, plausible, mask-confined** local
AI edit, with the input operation mask preserved. Do not scale the failed
SD v1.5 configuration. Resolve the visual scene candidates before treating
validation or test as held out. The frozen-detector inference gate is complete;
the next gate is an accepted, mask-aligned, visually plausible AI edit.
