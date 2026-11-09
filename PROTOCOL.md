# Protocol for the 300 phone photographs

**Status on 9 November 2026:** 300 source photographs inventoried; 20 source
copies selected for a development-only pilot. No local AI edits, operation
masks, or foreground annotations have been accepted. Two official frozen
detectors have been smoke-tested on 20 development-only pristine copies;
those 40 scores are not local-edit results. The 180/60/60 split is provisional.

## 1. Source and split review

1. Review all 12 contact sheets and `visual_scene_candidates.csv`. Mark photos
   of the same capture burst, subject, or near-identical viewpoint as a common
   grouping unit. Perceptual hashes miss many same-scene views; no automatic
   duplicate finding should be treated as a complete scene audit.
2. Resolve each candidate pair as `same_group` or `different_group`, with a
   short reason. Rebuild the split using confirmed groups; preserve the 20
   pilot roots in development or exclude them from the final held-out test.
3. Save the final manifest with a SHA-256 checksum and do not reassign test
   roots after viewing test detector outputs. Keep all derivatives of one
   original root in the same partition. If the team dataset contains any of
   these same photos, link its entries to the same root group.
4. Keep the original ZIPs unchanged. They may contain EXIF metadata, including
   data that should not be distributed. This pipeline records orientation and
   camera model but does not export GPS metadata.

## 2. Development pilot using the 20 copied originals

The 20 pilot candidates are convenience-selected **only** to test the
workflow. They are not a representative sample or a final result. Before
editing each one:

1. View the source photo upright, using EXIF orientation. Decide whether a
   clear foreground can be defined from the original photo alone. Record
   `valid` or `uncertain` and an annotation note.
2. Produce one 8-bit, single-channel foreground PNG with white foreground
   (255) and black background (0). Human annotation or automatic image-wide
   candidates plus human review are allowed. Do not inspect a planned or
   completed edit mask when deciding the foreground. Never prompt SAM with
   the edit-mask box or points.
3. Store the mask path in `root_manifest.csv`. An uncertain foreground should
   remain uncertain; do not force a binary position label.
4. Select an edit region from the foreground or background as planned. Save
   the *operation mask supplied to the AI editor* as a single-channel PNG.
   Save the AI-edited image, prompt, editor/version, settings, export format,
   and any geometric transform. Do not substitute a pixel-difference map for
   the operation mask.
5. Export a pristine control through the same final decode/encode path where
   feasible. This helps detect whether format changes, rather than AI editing,
   explain detector scores. Keep an untouched original as an audit reference.
6. Review edit quality and image-mask alignment **before** seeing detector
   scores. Log failures and exclusions. An image that is resized or cropped
   without a recorded transform must not enter the primary mask analysis.

## 3. Full experimental design after pilot verification

- Fixed area bands: (0,2%], (2,5%], (5,10%], (10,20%], (20,40%], (40,100%].
  An empty or full-image mask is excluded from local-edit analysis.
- Where feasible, make a foreground and a background edit of comparable area
  from the **same root**, using the same editor and task type. Assign target
  conditions before looking at detector scores. Realised mask area, rather
  than planned target, determines the recorded band. Balance assigned area
  bands and locations across the two phone models and broad scene types where
  feasible; record any imbalance rather than concealing it.
- The primary location rule uses `w = |M ∩ F| / |M|`: foreground if `w >= 0.6`,
  background if `w <= 0.2`, mixed otherwise. Uncertain foreground or geometry
  is excluded from the primary binary location comparison and counted.
- Keep detector checkpoints frozen. Each official detector adapter must use its
  own prescribed colour conversion, resize/crop, normalisation, and score
  orientation. Record code revision and checkpoint hash.
- Derive the 5% FPR threshold from pristine **validation** controls and freeze
  it before final test analysis. Report AUROC, recall at the fixed threshold,
  scores, sample counts, unique-root counts, and uncertainty by root. Compare
  foreground versus background within area bands, noting residual differences
  in edit type, content, and editor.

The 300 photos are all phone photographs from two devices. The final report
must describe this source domain and limit claims accordingly. It must not
claim a detector failure, an area trend, or a location effect until real AI
edits, masks, frozen-checkpoint scores, and audited splits exist.
