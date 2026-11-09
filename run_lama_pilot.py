"""Run pretrained LaMa on the two predeclared development-only masks.

This is an editor qualification test. Output remains a trial until image
quality is reviewed independently of detector scores.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fyp_eval import read_grayscale_mask


BASE = Path(__file__).resolve().parent
MODEL = BASE / "model_cache" / "lama" / "big-lama.pt"
SOURCE = BASE / "pilot_working_png" / "16pro_187.png"
OUT = BASE / "lama_pilot"
DEFAULT_LABELS = ["foreground_mirror", "background_cones"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--record", default="lama_trial_results.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = torch.jit.load(str(MODEL), map_location="cpu").eval()
    with Image.open(SOURCE) as opened:
        source = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    height, width, _ = source.shape
    if height % 8 or width % 8:
        raise ValueError("LaMa pilot expects dimensions divisible by 8")
    image_tensor = torch.from_numpy(source.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    reports = []
    with torch.no_grad():
        for label in args.labels:
            mask_path = OUT / f"16pro_187_{label}_operation_mask.png"
            mask = read_grayscale_mask(mask_path) > 127
            if mask.shape != (height, width):
                raise ValueError(f"Mask geometry differs for {label}")
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            result = model(image_tensor, mask_tensor)
            raw = np.clip(result[0].permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
            if raw.shape != source.shape:
                raise ValueError(f"Output geometry differs for {label}")
            raw_path = OUT / f"16pro_187_{label}_lama_raw.png"
            Image.fromarray(raw, "RGB").save(raw_path)
            # Same deterministic postprocessing is applied to both conditions.
            # It makes the operation boundary exact; the raw output is retained.
            composite = source.copy()
            composite[mask] = raw[mask]
            output_path = OUT / f"16pro_187_{label}_lama_local.png"
            Image.fromarray(composite, "RGB").save(output_path)
            raw_difference = np.abs(raw.astype(np.int16) - source.astype(np.int16)).mean(axis=2)
            report = {
                "root_id": "16pro_187", "condition": label,
                "status": "development-only editor qualification; visual decision pending",
                "source_file": str(SOURCE.relative_to(BASE)),
                "operation_mask_file": mask_path.name,
                "raw_file": raw_path.name,
                "localized_file": output_path.name,
                "source_sha256": sha256(SOURCE),
                "mask_sha256": sha256(mask_path),
                "checkpoint_sha256": sha256(MODEL),
                "raw_sha256": sha256(raw_path),
                "localized_sha256": sha256(output_path),
                "model": "IOPaint published big-lama.pt TorchScript (pretrained LaMa)",
                "device": "cpu", "torch_version": torch.__version__,
                "geometry": [width, height],
                "operation_fraction": float(mask.mean()),
                "raw_outside_mask_changed_fraction_rgb_mean_gt_2": float((raw_difference[~mask] > 2).mean()),
                "localized_outside_mask_changed_fraction_rgb_mean_gt_2": float((np.abs(composite.astype(np.int16)-source.astype(np.int16)).mean(axis=2)[~mask] > 2).mean()),
            }
            reports.append(report)
            print(json.dumps({"condition": label, "output": str(output_path),
                              "operation_fraction": report["operation_fraction"]}), flush=True)
    (OUT / args.record).write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
