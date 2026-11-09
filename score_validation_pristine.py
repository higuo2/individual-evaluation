"""Score only frozen validation pristine controls, never test images.

Every root is decoded from the immutable supplied ZIP, EXIF oriented, and
converted to the same 1600-pixel-long-side RGB PNG path as the development
working images. The two detectors then apply their own official transforms.
"""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import torch
from PIL import Image, ImageOps

from run_frozen_detector_pilot import HERE, load_detector


MANIFEST = HERE / "data" / "frozen_root_manifest_v1.csv"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["cnn", "clip"], required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process")
    with MANIFEST.open(newline="") as stream:
        selected = [row for row in csv.DictReader(stream) if row["split"] == "validation"]
    if len(selected) != 60 or len({row["root_id"] for row in selected}) != 60:
        raise ValueError("Frozen manifest does not have 60 unique validation roots")
    selected.sort(key=lambda row: row["root_id"])
    model, transform, preprocessing, checkpoints = load_detector(args.detector, args.device)
    archives = {}
    rows = []
    try:
        with torch.no_grad():
            for entry in selected:
                archive_name = entry["archive"]
                if archive_name not in archives:
                    archives[archive_name] = zipfile.ZipFile(HERE.parent / archive_name)
                source_bytes = archives[archive_name].read(entry["archive_member"])
                if hashlib.sha256(source_bytes).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Source hash mismatch: {entry['root_id']}")
                with Image.open(io.BytesIO(source_bytes)) as opened:
                    working = ImageOps.exif_transpose(opened).convert("RGB")
                    if max(working.size) > 1600:
                        working.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                tensor = transform(working).unsqueeze(0).to(args.device)
                score = float(torch.sigmoid(model(tensor)).flatten()[0].item())
                if not 0 <= score <= 1:
                    raise ValueError(f"Invalid score: {entry['root_id']}")
                rows.append({"detector": args.detector, "root_id": entry["root_id"],
                             "sample_id": entry["root_id"] + "_pristine_1600_png",
                             "split": "validation", "kind": "pristine",
                             "score": f"{score:.10f}", "area_band": "", "location": ""})
                print(f"{args.detector} {entry['root_id']} {score:.6f}", flush=True)
    finally:
        for archive in archives.values():
            archive.close()
    output = HERE / "data" / f"validation_pristine_scores_{args.detector}.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "purpose": "frozen validation pristine control scoring only",
        "detector": args.detector, "device": args.device,
        "manifest": MANIFEST.name, "manifest_sha256": file_hash(MANIFEST),
        "canonical_conversion": "EXIF transpose; RGB; thumbnail max side 1600 using PIL Lanczos; PNG-equivalent pixels",
        "detector_preprocessing": preprocessing,
        "checkpoints": checkpoints,
        "score_direction": "sigmoid(logit); larger means more likely synthetic",
        "validation_roots": len(rows),
        "score_file": output.name, "score_file_sha256": file_hash(output),
    }
    (HERE / "data" / f"validation_pristine_run_{args.detector}.json").write_text(
        json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
