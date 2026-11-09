"""Make a second development-only mask on pristine scooter photo 16pro_187."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fyp_eval import mask_measurement

base = Path(__file__).resolve().parent
root = "16pro_187"
out = base / "pilot_trial"
out.mkdir(exist_ok=True)
with Image.open(base / "pilot_working_png" / f"{root}.png") as opened:
    original = opened.convert("RGB")
    original.thumbnail((512, 512), Image.Resampling.LANCZOS)
original.save(out / f"{root}_pristine_512.png")
mask = Image.new("L", original.size, 0)
draw = ImageDraw.Draw(mask)
# Paved empty area to the left of the central scooter, selected without using
# any generated edit or foreground mask. The mask is a technical trial only.
draw.ellipse((52, 103, 139, 198), fill=255)
mask.save(out / f"{root}_operation_mask_512.png")
measure = mask_measurement(np.asarray(mask), original.size)
(out / "cone_trial_input_metadata.json").write_text(json.dumps({
    "root_id": root,
    "status": "development-only technical input; no foreground label",
    "pristine_path": f"{root}_pristine_512.png",
    "mask_path": f"{root}_operation_mask_512.png",
    "r": measure["r"],
    "area_band": measure["area_band"],
    "prompt_proposal": "A realistic photo of a black scooter on paved ground with a small orange traffic cone to its left.",
}, indent=2) + "\n")
print(json.dumps({"size": original.size, "r": measure["r"],
                  "area_band": measure["area_band"]}))
