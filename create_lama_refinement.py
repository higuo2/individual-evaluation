"""Prepare revised development-only removal masks after visual QA of v1.

v1 masks and outputs remain intact for an auditable rejection record.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from fyp_eval import mask_measurement, read_grayscale_mask


BASE = Path(__file__).resolve().parent
OUT = BASE / "lama_pilot"
SOURCE = BASE / "pilot_working_png" / "16pro_187.png"
FOREGROUND = BASE / "pilot_working_png" / "16pro_187_foreground.png"
with Image.open(SOURCE) as opened:
    size = opened.size
if size != (1600, 1200):
    raise ValueError("Unexpected source geometry")
foreground = read_grayscale_mask(FOREGROUND)

# v2 foreground mask includes all of the left mirror, stalk and attachment;
# v1 left part of the stalk visible and the model recreated the mirror.
pixels = np.zeros((size[1], size[0]), dtype=np.uint8)
pixels[80:340, 560:810] = foreground[80:340, 560:810]
fg = Image.fromarray(pixels, "L").filter(ImageFilter.MaxFilter(21))

# v1 background output left small yellow cone tips outside the polygon.
# Enclose both cones and a small surrounding margin before LaMa synthesis.
bg = Image.new("L", size, 0)
ImageDraw.Draw(bg).rectangle((948, 104, 1050, 228), fill=255)

rows = []
for label, mask, wanted in [
    ("foreground_mirror_v2", fg, "foreground"),
    ("background_cones_v2", bg, "background"),
]:
    path = OUT / f"16pro_187_{label}_operation_mask.png"
    mask.save(path)
    measurement = mask_measurement(read_grayscale_mask(path), size, foreground)
    if measurement["location"] != wanted:
        raise ValueError(f"{label} measured as {measurement['location']}")
    rows.append({"label": label, "mask": path.name, **measurement})
(OUT / "operation_mask_plan_v2.json").write_text(json.dumps({
    "root_id": "16pro_187", "purpose": "revised development-only LaMa qualification",
    "reason": "v1 left a mirror and yellow cone tips visible in masked output",
    "regions": rows,
}, indent=2) + "\n")
print(json.dumps(rows, indent=2))
