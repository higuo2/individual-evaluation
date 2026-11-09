"""Run exactly one local, development-only mask-conditioned inpainting trial.

Requires the pinned model files downloaded under models/. No photo is uploaded.
This output is a technical check, not a benchmark sample or detector result.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from fyp_eval import read_grayscale_mask


BASE = Path(__file__).resolve().parent
MODEL = BASE / "models" / "stable-diffusion-inpainting"
TRIAL = BASE / "pilot_trial"
DEFAULT_PROMPT = ("A natural photograph of watermelon slices on a white plate, "
                  "with one central slice replaced by a piece of yellow pineapple.")
SEED = 20261109
STRENGTH = 1.0

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="16pro_199")
parser.add_argument("--mask", default="16pro_199_operation_mask_512.png")
parser.add_argument("--prompt", default=DEFAULT_PROMPT)
parser.add_argument("--tag", default="trial")
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--guidance", type=float, default=7.5)
parser.add_argument("--padding-mask-crop", type=int)
args = parser.parse_args()
ROOT = args.root
PROMPT = args.prompt
STEPS = args.steps
GUIDANCE = args.guidance
if not args.tag.replace("_", "").isalnum():
    raise ValueError("tag must be alphanumeric with optional underscores")

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS unavailable in this process; run in an MPS-enabled local environment")
if not (MODEL / "MODEL_REVISION.txt").exists():
    raise FileNotFoundError("Pinned local inpainting weights have not finished downloading")
input_path = TRIAL / f"{ROOT}_pristine_512.png"
mask_path = TRIAL / args.mask
with Image.open(input_path) as opened:
    original = opened.convert("RGB")
mask_data = read_grayscale_mask(mask_path)
if mask_data.shape != (original.height, original.width):
    raise ValueError("trial mask and image dimensions differ")
mask = Image.fromarray(mask_data, "L")

start = time.monotonic()
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    str(MODEL), variant="fp16", torch_dtype=torch.float32,
    use_safetensors=True, local_files_only=True,
)
pipeline.to("mps")
generator = torch.Generator(device="mps").manual_seed(SEED)
with torch.inference_mode():
    result = pipeline(prompt=PROMPT, image=original, mask_image=mask,
                      generator=generator, num_inference_steps=STEPS,
                      guidance_scale=GUIDANCE, strength=STRENGTH,
                      width=original.width, height=original.height,
                      padding_mask_crop=args.padding_mask_crop)
    safety_flags = result.nsfw_content_detected
    result = result.images[0]
elapsed = time.monotonic() - start
if result.size != original.size:
    raise ValueError(f"output geometry differs: {result.size} versus {original.size}")
output_path = TRIAL / f"{ROOT}_ai_edit_{args.tag}.png"
result.save(output_path)
before = np.asarray(original, dtype=np.int16)
after = np.asarray(result, dtype=np.int16)
outside = mask_data <= 127
absolute_difference = np.abs(after - before).mean(axis=2)
metadata = {
    "status": "development-only technical inpainting trial; no detector result",
    "root_id": ROOT,
    "source_file": input_path.name,
    "operation_mask_file": mask_path.name,
    "edited_file": output_path.name,
    "model_repository": "stable-diffusion-v1-5/stable-diffusion-inpainting",
    "model_revision": (MODEL / "MODEL_REVISION.txt").read_text().splitlines()[1],
    "model_variant": "fp16",
    "compute_dtype": "float32 (weights loaded from fp16 files)",
    "diffusers_version": __import__("diffusers").__version__,
    "torch_version": torch.__version__,
    "device": "mps",
    "prompt": PROMPT,
    "seed": SEED,
    "inference_steps": STEPS,
    "guidance_scale": GUIDANCE,
    "strength": STRENGTH,
    "padding_mask_crop": args.padding_mask_crop,
    "safety_flags": safety_flags,
    "elapsed_seconds": elapsed,
    "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    "fraction_outside_mask_with_rgb_difference_gt_2": float(
        (absolute_difference[outside] > 2).mean()),
    "mean_absolute_rgb_difference_outside_mask": float(
        absolute_difference[outside].mean()),
}
(TRIAL / f"inpaint_{args.tag}_result.json").write_text(
    json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(output_path), "elapsed_seconds": elapsed,
                  "outside_mask_changed_fraction": metadata[
                      "fraction_outside_mask_with_rgb_difference_gt_2"]}, indent=2))
