# Local AI-editing pilot and qualification workflow

The user has not selected a final editor and supplied no accepted edited
images or operation masks. Five local development-only attempts are recorded
in `pilot_trial/QUALIFICATION.md`; all failed quality checks. The pilot uses a mask-conditioned inpainting
pipeline because the *exact input mask* can be saved. Hugging Face Diffusers
documents `AutoPipelineForInpainting` with `image` and `mask_image`; white mask
pixels are edited and black pixels are retained:
<https://huggingface.co/docs/diffusers/using-diffusers/inpaint>.

The computer has an Apple M2 and 16 GB memory. Diffusers documents Apple MPS
support: <https://huggingface.co/docs/diffusers/optimization/mps>. Diffusers
0.35.2 and a pinned SD v1.5 inpainting checkpoint were installed in the
project, and MPS worked in an approved local process. The first five
development-only trials are documented in `STATUS.md`: none passed visual
semantic quality review, so this configuration is **not approved for scale**.
The crop option confined pixel changes to the input mask, while the raw
uncropped outputs changed many pixels outside it. The official Diffusers
documentation also describes the `apply_overlay` operation for preserving
unmasked pixels: <https://huggingface.co/docs/diffusers/using-diffusers/inpaint>.

## Gate A — masks before generation

1. Open `mask_editor.html` locally. Load a file from `pilot_working_png/`.
2. Draw and save a foreground mask while viewing **only the pristine working
   image**. The mask is an annotation proposal, not an approved foreground
   label. Review it for omitted objects, spill into the background, and
   ambiguity. If the scene has no clear primary object, mark it uncertain.
3. Only after freezing that foreground mask, draw a distinct operation mask
   intended for foreground editing and one for background editing. Save both
   masks as PNGs. Aim for comparable realised `r` values; use the measurement
   function to check rather than assuming the drawn size is correct.
4. Save the input mask even if an editor also offers a visual preview. The
   *submitted operation mask*, not a pixel-difference image, is the primary
   area definition.

## Gate B — first two-image editor trial

1. Select one development root with a clear object and background. Use only
   its upright `pilot_working_png` image and saved masks. Keep that root in
   development even if the final split is revised.
2. Use one inpainting model/checkpoint and fixed export settings. Record model
   identifier/revision, Diffusers and PyTorch versions, prompt, negative
   prompt if any, random seed, inference steps, guidance, strength, and any
   crop/padding setting. Save the raw edited outputs separately.
3. Verify that the returned image dimensions match the working image and mask.
   Inspect whether pixels outside the nominal mask changed, whether the edit
   looks plausible, and whether mask edges are visibly artificial. Record the
   observations before any detector score is viewed.
4. Run `fyp_eval.py measure-samples` on the resulting edited-image/mask pairs.
   A mismatch blocks the primary analysis until the transform is understood.
5. Create a matching unedited export control if the editor changes format or
   compression. Otherwise a detector might respond to the export path rather
   than the AI edit.

## Gate C — scale only after the trial works

Use the same documented editor pipeline across experimental conditions. Record
any failed generation and the exclusion reason. To study area, cover the six
pre-specified realised `r` bands. To study location, use same-root foreground
and background edits of similar area where feasible. Do not select or revise
samples based on detector outputs. A one-editor collection supports conclusions
about that editor and these phone-photo sources; it does not establish
generality across all AI editors or camera domains.
