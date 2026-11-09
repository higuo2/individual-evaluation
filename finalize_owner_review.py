"""Record owner-confirmed scene pairs and ingest working-size foreground masks.

The split is frozen only after verifying that confirmed scene grouping agrees
with the provisional split and that all 20 annotated pilot roots are in dev.
"""

import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

from PIL import Image

from fyp_eval import ROOT_FIELDS, apply_visual_candidates, audit, read_csv, read_grayscale_mask, write_csv


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
WORK = BASE / "pilot_working_png"
ROOTS = DATA / "provisional_root_manifest.csv"
FROZEN = DATA / "frozen_root_manifest_v1.csv"


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_rows(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    candidates = read_csv(DATA / "visual_scene_candidates.csv")
    if len(candidates) != 25:
        raise ValueError("Expected 25 owner-reviewed scene pairs; inspect changed candidate list")
    for pair in candidates:
        if pair["review_status"] != "pending_owner_review":
            raise ValueError("Candidate list has already been changed; inspect before finalizing")
        pair["review_status"] = "owner_confirmed_similar_scene"
    reviewed_path = DATA / "visual_scene_reviewed.csv"
    write_rows(reviewed_path, candidates, list(candidates[0]))

    grouped_path = DATA / "owner_confirmed_grouped_roots.csv"
    apply_visual_candidates(DATA / "root_manifest.csv", reviewed_path, grouped_path)
    grouped = {r["root_id"]: r for r in read_csv(grouped_path)}
    roots = read_csv(ROOTS)
    if len(roots) != 300:
        raise ValueError("Expected 300 source roots")
    for root in roots:
        if root["candidate_cluster"] != grouped[root["root_id"]]["candidate_cluster"]:
            raise ValueError("Confirmed grouping differs from provisional grouping; rebuild split")
    if not audit(ROOTS)["pass"]:
        raise ValueError("Provisional split fails audit")

    review_path = WORK / "foreground_review.csv"
    reviews = read_csv(review_path)
    working_path = WORK / "working_root_manifest.csv"
    working_roots = read_csv(working_path)
    working_by_id = {r["root_id"]: r for r in working_roots}
    if len(reviews) != 20 or len(working_by_id) != 20:
        raise ValueError("Expected 20 pilot roots")
    annotations = []
    for review in reviews:
        root_id = review["root_id"]
        root = working_by_id[root_id]
        if root["split"] != "development":
            raise ValueError(f"Pilot root {root_id} is not in development")
        image_path = WORK / review["working_image"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        foreground_path = WORK / f"{root_id}_foreground.png"
        uncertain_path = WORK / f"{root_id}_uncertain.png"
        if foreground_path.exists() == uncertain_path.exists():
            raise ValueError(f"Expected exactly one annotation type for {root_id}")
        with Image.open(image_path) as opened:
            image_size = opened.size
        if foreground_path.exists():
            mask = read_grayscale_mask(foreground_path)
            if mask.shape != (image_size[1], image_size[0]):
                raise ValueError(f"Foreground geometry mismatch: {root_id}")
            fraction = float((mask > 127).mean())
            if fraction <= 0 or fraction >= 1:
                raise ValueError(f"Invalid foreground fraction: {root_id}")
            review.update({"foreground_mask": foreground_path.name,
                           "status": "valid_owner_annotation",
                           "reviewer": "photo owner",
                           "review_note": "Owner-supplied mask; format and dimensions verified; semantic boundary not independently audited"})
            root.update({"foreground_status": "valid",
                         "foreground_mask_path": foreground_path.name,
                         "review_note": "Working-size owner foreground mask; see foreground_review.csv"})
            annotations.append({"root_id": root_id, "status": "valid_owner_annotation",
                                "mask_file": foreground_path.name,
                                "mask_sha256": digest(foreground_path),
                                "foreground_fraction": f"{fraction:.8f}"})
        else:
            mask = read_grayscale_mask(uncertain_path)
            if mask.shape != (image_size[1], image_size[0]):
                raise ValueError(f"Uncertain marker geometry mismatch: {root_id}")
            review.update({"foreground_mask": "", "status": "uncertain",
                           "reviewer": "photo owner",
                           "review_note": "Owner marked uncertain via filename; no reason supplied"})
            root.update({"foreground_status": "uncertain",
                         "foreground_mask_path": "",
                         "review_note": "Working-size owner uncertain label; see foreground_review.csv"})
            annotations.append({"root_id": root_id, "status": "uncertain",
                                "mask_file": uncertain_path.name,
                                "mask_sha256": digest(uncertain_path),
                                "foreground_fraction": ""})

    write_rows(review_path, reviews, list(reviews[0]))
    write_csv(working_path, working_roots, ROOT_FIELDS)
    annotation_path = WORK / "foreground_annotation_audit.csv"
    write_rows(annotation_path, annotations, list(annotations[0]))
    write_csv(FROZEN, roots, ROOT_FIELDS)
    result = audit(FROZEN)
    if not result["pass"]:
        raise ValueError(result)
    record = {
        "date": str(date.today()),
        "review_basis": "User stated all 25 selected similar-scene pairs were manually reviewed and confirmed similar",
        "confirmed_pairs": len(candidates),
        "source_roots": len(roots),
        "split_counts": result["split_counts"],
        "pilot_foreground_status_counts": dict(Counter(r["status"] for r in reviews)),
        "frozen_manifest": FROZEN.name,
        "frozen_manifest_sha256": digest(FROZEN),
        "reviewed_pairs_sha256": digest(reviewed_path),
        "annotation_audit_sha256": digest(annotation_path),
        "audit": result,
        "caveat": "Foreground masks are 1600-pixel working derivatives, not full-resolution root masks; use working_root_manifest.csv for development measurements",
    }
    (DATA / "freeze_record_v1.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
