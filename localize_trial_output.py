"""Create a development-only local composite with Diffusers' documented overlay.

This does not make the prompt follow more closely. It only tests whether the
mask can strictly confine changed pixels while preserving the raw model output.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from fyp_eval import read_grayscale_mask

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", default="16pro_199")
parser.add_argument("--tag", default="trial")
args = parser.parse_args()
base = Path(__file__).resolve().parent / "pilot_trial"
root = args.root
with Image.open(base / f"{root}_pristine_512.png") as opened:
    original = opened.convert("RGB")
with Image.open(base / f"{root}_ai_edit_{args.tag}.png") as opened:
    generated = opened.convert("RGB")
mask_array = read_grayscale_mask(base / f"{root}_operation_mask_512.png")
mask = Image.fromarray(mask_array, "L")
if generated.size != original.size or mask.size != original.size:
    raise ValueError("trial geometry mismatch")
composite = VaeImageProcessor().apply_overlay(mask, original, generated)
output = base / f"{root}_local_composite_{args.tag}.png"
composite.save(output)
outside = mask_array <= 127
inside = ~outside
before = np.asarray(original, dtype=np.int16)
after = np.asarray(composite, dtype=np.int16)
difference = np.abs(after - before).mean(axis=2)
report = {
    "status": "development-only postprocessing trial; not accepted as benchmark sample",
    "raw_generated_image": f"{root}_ai_edit_{args.tag}.png",
    "local_composite": output.name,
    "method": "diffusers VaeImageProcessor.apply_overlay",
    "outside_mask_changed_fraction_gt_2": float((difference[outside] > 2).mean()),
    "inside_mask_changed_fraction_gt_2": float((difference[inside] > 2).mean()),
    "mean_absolute_rgb_difference_inside": float(difference[inside].mean()),
    "quality_note": "Manual semantic and boundary-quality review required before any use",
}
(base / f"local_composite_{args.tag}_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
