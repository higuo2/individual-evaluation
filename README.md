# Individual local-edit detector evaluation

This directory contains the student's **individual evaluation pipeline**. The
two ZIP archives in the parent directory are treated as immutable source
photographs. The inventory reads them without extracting or modifying them.
No accepted locally AI-edited image has been produced yet. The two frozen
detectors have been smoke-tested on 20 development-only pristine controls;
their scores are not a detection result for local edits.

See `PROTOCOL.md` for the concrete annotation, editing, and test procedure.
See `STATUS.md` for what has actually been verified and `DETECTORS.md` for
official checkpoint integration requirements.
See `EDITING.md` for the mask-first inpainting pilot and qualification workflow.
See `PROGRESS_AND_NEXT_STEPS.md` for the completed work, unresolved gates, and
the exact pause point after the development trials.

## Current inputs and outputs

- `data/root_manifest.csv`: one row per readable source photograph. `split` is
  intentionally empty until a root-disjoint assignment is reviewed and frozen.
- `data/duplicate_candidates.csv`: exact or perceptual-hash-close photo pairs.
  Perceptual proximity is only a *candidate* for human review, not proof that
  images share a scene or root.
- `data/contact_sheets/`: local thumbnails to review repeated scenes and
  foreground suitability. These may contain private images; keep them local.
- `data/visual_scene_candidates.csv`: visually spotted related-scene pairs.
  These are conservative candidates, not confirmed duplicates.
- `data/visual_candidate_root_manifest.csv`: candidate groups merged before
  a provisional split.
- `data/inventory_summary.json`: verified counts and decoding errors.
- `data/provisional_root_manifest.csv`: optional review-only 60/20/20 split.
- `data/edited_samples_template.csv`: blank schema for future edited samples.
- `data/predictions_template.csv`: expected frozen-detector score schema.
- `data/pilot_candidates.csv`: 20 convenience-selected development originals,
  based on contact sheets only. Foreground suitability is not yet confirmed.
- `pilot_originals/`: byte-identical local copies of those 20 source photos,
  ready for development-only annotation or local AI editing.
- `pilot_working_png/`: upright 1600-pixel-long-side development copies for
  mask authoring. They are derived images, not AI edits or final benchmark data.
- `data/pilot_pristine_scores.csv`: paired CNNDetection/CLIP inference checks
  on 20 development-only pristine copies; not benchmark results.
- `DETECTORS.md`: official repository commits, checkpoint hashes, exact
  preprocessing, and inference-gate status.

## Commands

Run from this directory:

```bash
python3 fyp_eval.py inventory '../14pro_100张.zip' '../16pro_200张.zip' --output data
python3 fyp_eval.py apply-visual-candidates data/root_manifest.csv data/visual_scene_candidates.csv --output data/visual_candidate_root_manifest.csv
python3 fyp_eval.py provisional-split data/visual_candidate_root_manifest.csv --output data/provisional_root_manifest.csv
python3 fyp_eval.py audit data/provisional_root_manifest.csv
python3 fyp_eval.py extract-pilot data/provisional_root_manifest.csv data/pilot_candidates.csv --archives .. --output pilot_originals
python3 fyp_eval.py prepare-pilot-images pilot_originals --output pilot_working_png --root-manifest data/provisional_root_manifest.csv
python3 fyp_eval.py measure-samples data/provisional_root_manifest.csv data/edited_samples.csv --output data/measured_samples.csv
python3 -m unittest discover -s tests -v
# See run_inpaint_trial.py --help for required root, mask and prompt arguments.
USE_TF=0 USE_FLAX=0 .venv/bin/python run_frozen_detector_pilot.py --detector cnn --device mps
USE_TF=0 USE_FLAX=0 .venv/bin/python run_frozen_detector_pilot.py --detector clip --device mps
# Only after real model scores exist:
python3 analyze.py data/predictions.csv --output data/test_analysis.json
```

## Protocol before detector testing

1. Review `duplicate_candidates.csv` and **all** contact sheets for the same
   scene, burst, crop, or near duplicate. Group related photographs before a
   final split. The provisional split is not frozen and must not be used to
   support a held-out test claim.
2. On **pristine root images only**, define a foreground rule using development
   roots. Freeze it before looking at edit masks. SAM automatic proposals may
   assist review, but do not prompt SAM with edit-mask coordinates. Mark
   ambiguous images uncertain.
3. Save each local AI-edit operation mask, edited image, tool/version, prompt,
   output settings, and root ID. Review image/mask alignment and visual quality
   without inspecting detector scores. Operation masks are not pixel-difference
   masks.
   Copy `data/edited_samples_template.csv` to `data/edited_samples.csv` only
   when real edits exist; paths are relative to that CSV's directory. The
   measurement command rejects missing files and mismatched dimensions.
   For the 1600-pixel development pilot, use
   `pilot_working_png/working_root_manifest.csv` instead of the full-size
   `data/provisional_root_manifest.csv` when validating geometry. Open
   `mask_editor.html` in a browser to draw masks on those upright PNGs.
4. Use `mask_measurement()` with fixed threshold 127 for binary/soft 8-bit
   masks; record the threshold in the final run configuration. Area bins are
   (0,2%], (2,5%], (5,10%], (10,20%], (20,40%], (40,100%]. Location is
   foreground for overlap >= 0.6, background for overlap <= 0.2, and mixed
   otherwise. Uncertain foreground maps are excluded from the primary binary
   location contrast and counted.
5. Run published frozen detector checkpoints with their own preprocessing.
   Calibrate the 5% FPR threshold using pristine validation roots only. Freeze
   code and rules before viewing final test metrics. Report counts, AUROC,
   thresholded recall, and root-clustered uncertainty, including exclusions.

The phone-photo collection represents a particular source domain and does not
by itself establish generalisation to other cameras, editors, or content.
