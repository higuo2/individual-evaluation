"""Create explicitly specified development-only object-removal operation masks.

These are *operation* masks for a LaMa qualification test, not foreground
annotations. The foreground mask was supplied by the owner beforehand.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from fyp_eval import mask_measurement, read_grayscale_mask


BASE = Path(__file__).resolve().parent
SOURCE = BASE / "pilot_working_png" / "16pro_187.png"
FOREGROUND = BASE / "pilot_working_png" / "16pro_187_foreground.png"
OUT = BASE / "lama_pilot"
OUT.mkdir(exist_ok=True)

with Image.open(SOURCE) as opened:
    size = opened.size
if size != (1600, 1200):
    raise ValueError(f"Unexpected source size {size}; inspect polygons")
foreground = read_grayscale_mask(FOREGROUND)

# Coordinates are deliberately recorded in this script for reproducibility.
# Mirror and stalk on scooter; two traffic cones on the rear driveway.
regions = {
    "background_cones": [
        [(961, 138), (982, 134), (997, 155), (1000, 192), (1005, 210),
         (958, 212), (963, 185)],
        [(1003, 122), (1022, 121), (1037, 143), (1043, 188),
         (1041, 207), (1004, 208), (1009, 173)],
    ],
}

results = []
for label in ("foreground_mirror", "background_cones"):
    if label == "foreground_mirror":
        # Select the previously annotated scooter mirror/stalk, before looking
        # at any edited output. The ROI limits the owner's foreground map to
        # one small physical component; it does not define the foreground map.
        pixels = np.zeros((size[1], size[0]), dtype=np.uint8)
        pixels[80:242, 580:740] = foreground[80:242, 580:740]
        mask = Image.fromarray(pixels, "L").filter(ImageFilter.MaxFilter(7))
    else:
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        for polygon in regions[label]:
            draw.polygon(polygon, fill=255)
        mask = mask.filter(ImageFilter.MaxFilter(11))
    path = OUT / f"16pro_187_{label}_operation_mask.png"
    mask.save(path)
    measurement = mask_measurement(read_grayscale_mask(path), size, foreground)
    results.append({"label": label, "mask": path.name,
                    "target": "remove visible object(s) with pretrained LaMa",
                    **measurement})
    if label == "foreground_mirror" and measurement["location"] != "foreground":
        raise ValueError("Foreground mirror mask is not foreground by frozen rule")
    if label == "background_cones" and measurement["location"] != "background":
        raise ValueError("Background cone mask is not background by frozen rule")

(OUT / "operation_mask_plan.json").write_text(json.dumps({
    "root_id": "16pro_187", "source": str(SOURCE.relative_to(BASE)),
    "foreground_annotation": str(FOREGROUND.relative_to(BASE)),
    "purpose": "development-only editor qualification; no benchmark acceptance yet",
    "regions": results,
}, indent=2) + "\n")
print(json.dumps(results, indent=2))
