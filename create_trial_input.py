"""Create one clearly labelled, development-only inpainting smoke-test input.

The polygon is an operation-mask proposal on the pristine photo. It is not a
foreground annotation and this script does not create an AI-edited image.
"""

from pathlib import Path
import json

from PIL import Image, ImageDraw

from fyp_eval import mask_measurement
import numpy as np


ROOT = "16pro_199"
BASE = Path(__file__).resolve().parent
SOURCE = BASE / "pilot_working_png" / f"{ROOT}.png"
OUT = BASE / "pilot_trial"
OUT.mkdir(exist_ok=True)

with Image.open(SOURCE) as opened:
    pristine = opened.convert("RGB")
    pristine.thumbnail((512, 512), Image.Resampling.LANCZOS)
pristine.save(OUT / f"{ROOT}_pristine_512.png")

mask = Image.new("L", pristine.size, 0)
draw = ImageDraw.Draw(mask)
# Covers one central watermelon piece in the source; this is deliberately
# separate from any foreground-definition process.
draw.polygon([(231, 207), (264, 173), (326, 182), (357, 229),
              (333, 266), (273, 264), (228, 241)], fill=255)
mask.save(OUT / f"{ROOT}_operation_mask_512.png")
measurement = mask_measurement(np.asarray(mask), pristine.size)
(OUT / "trial_input_metadata.json").write_text(json.dumps({
    "root_id": ROOT,
    "status": "development-only technical input; no AI edit generated yet",
    "pristine_path": f"{ROOT}_pristine_512.png",
    "operation_mask_path": f"{ROOT}_operation_mask_512.png",
    "image_width": pristine.width,
    "image_height": pristine.height,
    "mask_r": measurement["r"],
    "mask_area_band": measurement["area_band"],
    "foreground_label": "not_assigned",
    "prompt_proposal": "A natural photograph of watermelon slices on a white plate, with one central slice replaced by a piece of yellow pineapple.",
}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"size": pristine.size, "r": measurement["r"],
                  "area_band": measurement["area_band"]}))
