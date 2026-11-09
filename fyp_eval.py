"""Local, read-only photo inventory and evaluation checks for the individual FYP.

The source ZIP archives are never extracted or modified. No detector result is
produced by this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn


ROOT_FIELDS = [
    "root_id", "device_group", "archive", "archive_member", "filename",
    "sha256", "width", "height", "exif_orientation", "exif_model",
    "phash", "candidate_cluster", "split", "foreground_status",
    "foreground_mask_path", "review_note",
]
SAMPLE_FIELDS = [
    "sample_id", "root_id", "split", "edited_image_path", "operation_mask_path",
    "edit_tool", "edit_tool_version", "edit_instruction", "edit_settings",
    "target_area_band", "actual_r", "actual_area_band", "foreground_overlap",
    "location_label", "quality_status", "exclusion_reason",
]
AREA_BANDS = ((0.0, .02), (.02, .05), (.05, .10), (.10, .20), (.20, .40), (.40, 1.0))
SPLIT_NAMES = ("development", "validation", "test")


def actual_area_band(r: float) -> str:
    if not (0 < r <= 1):
        raise ValueError("area ratio must be in (0, 1]")
    for lo, hi in AREA_BANDS:
        if lo < r <= hi:
            return f"({lo:g},{hi:g}]"
    raise AssertionError("unreachable")


def mask_measurement(operation_mask: np.ndarray, image_size: tuple[int, int],
                     foreground_mask: np.ndarray | None = None,
                     threshold: int = 127) -> dict:
    """Measure masks in the *full edited image* coordinate system.

    image_size is (width, height). Soft masks use > threshold; the threshold
    must be frozen in the evaluation config before reading test scores.
    """
    width, height = image_size
    if operation_mask.ndim != 2 or operation_mask.shape != (height, width):
        raise ValueError("operation mask dimensions do not match edited image")
    if not 0 <= threshold <= 254:
        raise ValueError("threshold must be between 0 and 254")
    m = operation_mask > threshold
    count = int(m.sum())
    if count == 0:
        return {"r": 0.0, "area_band": None, "overlap": None,
                "location": "excluded_empty_mask"}
    r = count / (height * width)
    if count == height * width:
        return {"r": 1.0, "area_band": None, "overlap": None,
                "location": "excluded_full_image_mask"}
    result = {"r": r, "area_band": actual_area_band(r),
              "overlap": None, "location": "unlabelled"}
    if foreground_mask is not None:
        if foreground_mask.ndim != 2 or foreground_mask.shape != m.shape:
            raise ValueError("foreground mask dimensions do not match edited image")
        f = foreground_mask > threshold
        overlap = int(np.logical_and(m, f).sum()) / count
        result["overlap"] = overlap
        result["location"] = ("foreground" if overlap >= .6 else
                              "background" if overlap <= .2 else "mixed")
    return result


def perceptual_hash(image: Image.Image) -> int:
    grey = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coefficients = dctn(np.asarray(grey, dtype=np.float64), norm="ortho")[:8, :8]
    values = coefficients.flatten()[1:]
    bits = coefficients.flatten() > np.median(values)
    bits[0] = False
    return sum(int(bit) << i for i, bit in enumerate(bits))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        a, b = self.find(x), self.find(y)
        self.parent[b] = a


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_grayscale_mask(path: Path) -> np.ndarray:
    """Read a true grayscale mask or a black/white opaque browser PNG."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode in {"1", "L"}:
            return np.asarray(image.convert("L"))
        if image.mode not in {"RGB", "RGBA"}:
            raise ValueError("mask must be grayscale or opaque monochrome RGB/RGBA")
        pixels = np.asarray(image)
        if image.mode == "RGBA" and not np.all(pixels[:, :, 3] == 255):
            raise ValueError("mask alpha channel must be fully opaque")
        rgb = pixels[:, :, :3]
        if not (np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and
                np.array_equal(rgb[:, :, 0], rgb[:, :, 2])):
            raise ValueError("mask RGB channels must be identical grayscale")
        return rgb[:, :, 0].copy()


def contact_sheets(images: list[tuple[str, Image.Image]], output: Path,
                   columns: int = 5, rows_per_page: int = 5) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h = 224, 194
    per_page = columns * rows_per_page
    for page_index in range(math.ceil(len(images) / per_page)):
        page = Image.new("RGB", (columns * cell_w, rows_per_page * cell_h), "white")
        draw = ImageDraw.Draw(page)
        batch = images[page_index * per_page:(page_index + 1) * per_page]
        for i, (label, thumbnail) in enumerate(batch):
            x, y = (i % columns) * cell_w, (i // columns) * cell_h
            thumb = thumbnail.copy()
            thumb.thumbnail((cell_w - 12, cell_h - 32), Image.Resampling.LANCZOS)
            page.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 4))
            draw.text((x + 6, y + cell_h - 25), label, fill="black")
        page.save(output / f"contact_{page_index + 1:02d}.jpg", quality=88)


def inventory(archive_paths: list[Path], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    thumbs: list[tuple[str, Image.Image]] = []
    decode_errors: list[dict] = []
    archive_counts: dict[str, int] = {}
    for archive in archive_paths:
        device_group = "14pro" if "14pro" in archive.name.lower() else (
            "16pro" if "16pro" in archive.name.lower() else archive.stem)
        with zipfile.ZipFile(archive) as bundle:
            members = sorted(
                (member for member in bundle.infolist()
                 if not member.is_dir() and
                 not member.filename.startswith("__MACOSX/") and
                 not Path(member.filename).name.startswith("._") and
                 Path(member.filename).suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"}),
                key=lambda member: member.filename.lower(),
            )
            archive_counts[archive.name] = len(members)
            for number, member in enumerate(members, 1):
                root_id = f"{device_group}_{number:03d}"
                try:
                    raw = bundle.read(member)
                    sha = hashlib.sha256(raw).hexdigest()
                    with Image.open(io.BytesIO(raw)) as opened:
                        exif = opened.getexif()
                        orientation = exif.get(274, 1)
                        model = str(exif.get(272, ""))
                        image = ImageOps.exif_transpose(opened)
                        image.load()
                        width, height = image.size
                        photo_hash = perceptual_hash(image)
                        thumbnail = image.convert("RGB")
                        thumbnail.thumbnail((212, 162), Image.Resampling.LANCZOS)
                    records.append({
                        "root_id": root_id, "device_group": device_group,
                        "archive": archive.name, "archive_member": member.filename,
                        "filename": Path(member.filename).name, "sha256": sha,
                        "width": width, "height": height,
                        "exif_orientation": orientation, "exif_model": model,
                        "phash": f"{photo_hash:016x}", "candidate_cluster": "",
                        "split": "", "foreground_status": "not_annotated",
                        "foreground_mask_path": "", "review_note": "",
                    })
                    thumbs.append((root_id + " " + Path(member.filename).name, thumbnail))
                except Exception as error:
                    decode_errors.append({"archive": archive.name,
                                          "member": member.filename,
                                          "error": f"{type(error).__name__}: {error}"})
    uf = UnionFind(len(records))
    candidate_pairs: list[dict] = []
    for i, first in enumerate(records):
        for j in range(i + 1, len(records)):
            second = records[j]
            exact = first["sha256"] == second["sha256"]
            distance = hamming(int(first["phash"], 16), int(second["phash"], 16))
            if exact or distance <= 6:
                uf.union(i, j)
                candidate_pairs.append({"root_a": first["root_id"],
                                        "root_b": second["root_id"],
                                        "exact_sha256": exact,
                                        "phash_distance": distance,
                                        "review_status": "pending_manual_review"})
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        groups[uf.find(i)].append(i)
    for indices in groups.values():
        cluster_id = min(records[i]["root_id"] for i in indices)
        for i in indices:
            records[i]["candidate_cluster"] = cluster_id
    write_csv(output / "root_manifest.csv", records, ROOT_FIELDS)
    write_csv(output / "duplicate_candidates.csv", candidate_pairs,
              ["root_a", "root_b", "exact_sha256", "phash_distance", "review_status"])
    contact_sheets(thumbs, output / "contact_sheets")
    summary = {
        "archive_photo_counts": archive_counts,
        "readable_photos": len(records),
        "decode_errors": decode_errors,
        "exact_duplicate_pairs": sum(x["exact_sha256"] for x in candidate_pairs),
        "near_duplicate_candidate_pairs": sum(not x["exact_sha256"] for x in candidate_pairs),
        "candidate_clusters_with_multiple_photos": sum(len(v) > 1 for v in groups.values()),
        "status": "inventory_complete; duplicate candidates require human review; splits not frozen",
    }
    (output / "inventory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def provisional_split(manifest_path: Path, output_path: Path, seed: str) -> dict:
    """Create a review-only split; it is NOT a frozen test assignment."""
    records = read_csv(manifest_path)
    clusters: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        clusters[record["candidate_cluster"] or record["root_id"]].append(record)
    ordered = sorted(clusters.items(), key=lambda item:
                     hashlib.sha256((seed + ":" + item[0]).encode()).hexdigest())
    total = len(records)
    targets = {"development": .6 * total, "validation": .2 * total,
               "test": .2 * total}
    counts = Counter()
    for cluster_id, cluster_records in ordered:
        # Greedy assignment approximates 60/20/20 while keeping candidate
        # near-duplicate clusters together. Manual scene review remains vital.
        available = sorted(SPLIT_NAMES,
                           key=lambda name: (counts[name] / targets[name],
                                             SPLIT_NAMES.index(name)))
        chosen = available[0]
        counts[chosen] += len(cluster_records)
        for record in cluster_records:
            record["split"] = chosen
    write_csv(output_path, records, ROOT_FIELDS)
    return {"counts": dict(counts), "seed": seed,
            "status": "PROVISIONAL: review duplicate candidates and all contact sheets before freezing"}


def apply_visual_candidates(manifest_path: Path, candidates_path: Path,
                            output_path: Path) -> dict:
    """Conservatively group visually related roots for provisional splitting.

    Candidate pairs are not a claim that the files are identical. They require
    owner review before a final held-out test partition is frozen.
    """
    records = read_csv(manifest_path)
    index = {r["root_id"]: i for i, r in enumerate(records)}
    if len(index) != len(records):
        raise ValueError("duplicate root_id in manifest")
    uf = UnionFind(len(records))
    clusters = defaultdict(list)
    for i, record in enumerate(records):
        clusters[record["candidate_cluster"] or record["root_id"]].append(i)
    for group in clusters.values():
        for i in group[1:]:
            uf.union(group[0], i)
    candidates = read_csv(candidates_path)
    for pair in candidates:
        a, b = pair["root_a"], pair["root_b"]
        if a not in index or b not in index:
            raise ValueError(f"candidate pair references unknown root: {a}, {b}")
        uf.union(index[a], index[b])
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[uf.find(i)].append(i)
    for group in groups.values():
        cluster_id = min(records[i]["root_id"] for i in group)
        for i in group:
            records[i]["candidate_cluster"] = cluster_id
    write_csv(output_path, records, ROOT_FIELDS)
    return {"visual_candidate_pairs": len(candidates),
            "candidate_groups_with_multiple_photos": sum(len(g) > 1 for g in groups.values()),
            "status": "visual candidates conservatively grouped; manual review still required"}


def audit(manifest_path: Path, samples_path: Path | None = None) -> dict:
    records = read_csv(manifest_path)
    errors: list[str] = []
    ids = [r["root_id"] for r in records]
    if len(set(ids)) != len(ids):
        errors.append("duplicate root_id in root manifest")
    missing_splits = [r["root_id"] for r in records if r["split"] not in SPLIT_NAMES]
    split_by_cluster: dict[str, set[str]] = defaultdict(set)
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r["split"] in SPLIT_NAMES:
            split_by_cluster[r["candidate_cluster"] or r["root_id"]].add(r["split"])
            hash_to_splits[r["sha256"]].add(r["split"])
    for cluster, splits in split_by_cluster.items():
        if len(splits) > 1:
            errors.append(f"candidate cluster {cluster} crosses splits")
    for sha, splits in hash_to_splits.items():
        if len(splits) > 1:
            errors.append(f"identical SHA-256 {sha[:12]} crosses splits")
    sample_count = 0
    if samples_path and samples_path.exists():
        samples = read_csv(samples_path)
        sample_count = len(samples)
        root_splits = {r["root_id"]: r["split"] for r in records}
        for sample in samples:
            root_id = sample["root_id"]
            if root_id not in root_splits:
                errors.append(f"sample {sample['sample_id']} has unknown root_id")
            elif sample["split"] != root_splits[root_id]:
                errors.append(f"sample {sample['sample_id']} split differs from root")
            for field in ("edited_image_path", "operation_mask_path"):
                path_text = sample.get(field, "")
                if not path_text or not (samples_path.parent / path_text).exists():
                    errors.append(f"sample {sample['sample_id']} missing {field}")
    return {"roots": len(records), "samples": sample_count,
            "split_counts": dict(Counter(r["split"] for r in records)),
            "missing_split_root_ids": missing_splits,
            "errors": errors,
            "pass": not errors and not missing_splits}


def measure_samples(root_manifest: Path, samples_manifest: Path,
                    output_path: Path, threshold: int = 127) -> dict:
    """Validate real edited-image/mask pairs before any detector inference.

    Paths in CSV files are relative to the CSV's directory. A root foreground
    map may be omitted; in that case location remains unlabelled.
    """
    roots = {r["root_id"]: r for r in read_csv(root_manifest)}
    samples = read_csv(samples_manifest)
    results = []
    for sample in samples:
        result = dict(sample)
        result["measurement_status"] = "valid"
        result["measurement_error"] = ""
        try:
            root = roots[sample["root_id"]]
            if sample["split"] != root["split"]:
                raise ValueError("sample split does not match root split")
            edited_path = (samples_manifest.parent / sample["edited_image_path"]).resolve()
            mask_path = (samples_manifest.parent / sample["operation_mask_path"]).resolve()
            with Image.open(edited_path) as opened:
                edited = ImageOps.exif_transpose(opened)
                edited_size = edited.size
            expected = (int(root["width"]), int(root["height"]))
            if edited_size != expected:
                raise ValueError("edited image geometry differs from upright root; record an alignment transform")
            mask = read_grayscale_mask(mask_path)
            foreground = None
            if root["foreground_status"] == "valid":
                fg_path = (root_manifest.parent / root["foreground_mask_path"]).resolve()
                foreground = read_grayscale_mask(fg_path)
            result.update(mask_measurement(mask, edited_size, foreground, threshold))
            if result["location"] in {"excluded_empty_mask", "excluded_full_image_mask"}:
                result["measurement_status"] = "excluded"
                result["measurement_error"] = result["location"]
            elif foreground is None:
                result["location"] = "uncertain" if root["foreground_status"] == "uncertain" else "unlabelled"
        except (KeyError, OSError, ValueError) as error:
            result["measurement_status"] = "excluded"
            result["measurement_error"] = f"{type(error).__name__}: {error}"
            result["r"] = ""
            result["area_band"] = ""
            result["overlap"] = ""
            result["location"] = ""
        results.append(result)
    output_fields = SAMPLE_FIELDS + ["measurement_status", "measurement_error",
                                     "r", "area_band", "overlap", "location"]
    write_csv(output_path, results, output_fields)
    return {"samples": len(results),
            "valid": sum(r["measurement_status"] == "valid" for r in results),
            "excluded": sum(r["measurement_status"] == "excluded" for r in results),
            "output": str(output_path)}


def extract_pilot(root_manifest: Path, pilot_candidates: Path,
                  archive_directory: Path, output_dir: Path) -> dict:
    """Copy original bytes of development photos for a software-only pilot."""
    roots = {r["root_id"]: r for r in read_csv(root_manifest)}
    candidates = read_csv(pilot_candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for candidate in candidates:
        root_id = candidate["root_id"]
        root = roots[root_id]
        if root["split"] != "development":
            raise ValueError(f"pilot root {root_id} is not in development")
        archive = archive_directory / root["archive"]
        with zipfile.ZipFile(archive) as bundle:
            raw = bundle.read(root["archive_member"])
        if hashlib.sha256(raw).hexdigest() != root["sha256"]:
            raise ValueError(f"source hash changed for {root_id}")
        destination = output_dir / f"{root_id}{Path(root['filename']).suffix.lower()}"
        destination.write_bytes(raw)
        copied.append({"root_id": root_id, "path": str(destination),
                       "sha256": root["sha256"]})
    (output_dir / "source_checksums.json").write_text(
        json.dumps(copied, indent=2) + "\n", encoding="utf-8")
    return {"copied_originals": len(copied), "output_dir": str(output_dir),
            "status": "development-only source copies; no AI edits or masks created"}


def prepare_pilot_images(pilot_originals: Path, output_dir: Path,
                         max_side: int = 1600,
                         root_manifest: Path | None = None) -> dict:
    """Make upright, bounded-size PNG working images for annotation only."""
    if max_side < 512:
        raise ValueError("max_side must be at least 512")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in sorted(pilot_originals.glob("*.jpg")):
        with Image.open(source) as opened:
            upright = ImageOps.exif_transpose(opened).convert("RGB")
            original_size = upright.size
            if max(upright.size) > max_side:
                upright.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            destination = output_dir / f"{source.stem}.png"
            upright.save(destination, format="PNG")
        rows.append({"root_id": source.stem, "source_file": source.name,
                     "source_upright_width": original_size[0],
                     "source_upright_height": original_size[1],
                     "working_width": upright.width,
                     "working_height": upright.height,
                     "working_file": destination.name,
                     "working_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    write_csv(output_dir / "working_image_manifest.csv", rows,
              ["root_id", "source_file", "source_upright_width",
               "source_upright_height", "working_width", "working_height",
               "working_file", "working_sha256"])
    if root_manifest is not None:
        roots = {record["root_id"]: record for record in read_csv(root_manifest)}
        working_roots = []
        for row in rows:
            original = dict(roots[row["root_id"]])
            if original["split"] != "development":
                raise ValueError(f"pilot root {row['root_id']} must remain in development")
            original["width"] = row["working_width"]
            original["height"] = row["working_height"]
            original["foreground_status"] = "not_annotated"
            original["foreground_mask_path"] = ""
            original["review_note"] = "development-only 1600px working derivative; see working_image_manifest.csv"
            working_roots.append(original)
        write_csv(output_dir / "working_root_manifest.csv", working_roots, ROOT_FIELDS)
    return {"working_images": len(rows), "max_side": max_side,
            "status": "development-only derived PNGs; not AI edits"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("archives", nargs="+", type=Path)
    inv.add_argument("--output", type=Path, required=True)
    spl = sub.add_parser("provisional-split")
    spl.add_argument("manifest", type=Path)
    spl.add_argument("--output", type=Path, required=True)
    spl.add_argument("--seed", default="fyp-individual-v1")
    vis = sub.add_parser("apply-visual-candidates")
    vis.add_argument("manifest", type=Path)
    vis.add_argument("candidates", type=Path)
    vis.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("audit")
    check.add_argument("manifest", type=Path)
    check.add_argument("--samples", type=Path)
    meas = sub.add_parser("measure-samples")
    meas.add_argument("roots", type=Path)
    meas.add_argument("samples", type=Path)
    meas.add_argument("--output", type=Path, required=True)
    meas.add_argument("--threshold", type=int, default=127)
    pilot = sub.add_parser("extract-pilot")
    pilot.add_argument("roots", type=Path)
    pilot.add_argument("candidates", type=Path)
    pilot.add_argument("--archives", type=Path, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("prepare-pilot-images")
    prep.add_argument("originals", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--max-side", type=int, default=1600)
    prep.add_argument("--root-manifest", type=Path)
    args = parser.parse_args()
    if args.command == "inventory":
        result = inventory(args.archives, args.output)
    elif args.command == "provisional-split":
        result = provisional_split(args.manifest, args.output, args.seed)
    elif args.command == "apply-visual-candidates":
        result = apply_visual_candidates(args.manifest, args.candidates, args.output)
    elif args.command == "measure-samples":
        result = measure_samples(args.roots, args.samples, args.output, args.threshold)
    elif args.command == "extract-pilot":
        result = extract_pilot(args.roots, args.candidates, args.archives, args.output)
    elif args.command == "prepare-pilot-images":
        result = prepare_pilot_images(args.originals, args.output, args.max_side,
                                      args.root_manifest)
    else:
        result = audit(args.manifest, args.samples)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
