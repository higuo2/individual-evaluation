"""Pre-specified score analysis. Requires real, frozen-checkpoint predictions.

Input CSV columns: detector, root_id, sample_id, split, kind, score,
area_band, location. `kind` is `pristine` or `edited`; higher scores must
always mean stronger AI-edit evidence. One pristine score per root and detector
is required. This program never changes detector scores or thresholds on test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


AREA_ORDER = ["(0,0.02]", "(0.02,0.05]", "(0.05,0.1]", "(0.1,0.2]",
              "(0.2,0.4]", "(0.4,1]"]


def threshold_at_fpr(real_scores: list[float], target_fpr: float = .05) -> tuple[float, float]:
    if not real_scores:
        raise ValueError("validation pristine controls are required")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must lie between 0 and 1")
    sorted_scores = sorted(real_scores, reverse=True)
    allowed = math.floor(target_fpr * len(sorted_scores))
    threshold = sorted_scores[allowed] if allowed < len(sorted_scores) else -math.inf
    actual_fpr = sum(score > threshold for score in real_scores) / len(real_scores)
    return threshold, actual_fpr


def auroc_with_root_ci(positives_by_root: dict[str, list[float]],
                       controls_by_root: dict[str, float],
                       bootstrap_repeats: int = 1000,
                       seed: int = 2026) -> dict:
    roots = sorted(set(positives_by_root) & set(controls_by_root))
    if len(roots) < 2:
        return {"auroc": None, "ci95": None, "unique_roots": len(roots)}

    def estimate(chosen: list[str]) -> float:
        positive = [score for root in chosen for score in positives_by_root[root]]
        negative = [controls_by_root[root] for root in chosen]
        labels = [1] * len(positive) + [0] * len(negative)
        return float(roc_auc_score(labels, positive + negative))

    estimate_all = estimate(roots)
    rng = np.random.default_rng(seed)
    draws = [estimate([roots[i] for i in rng.integers(0, len(roots), len(roots))])
             for _ in range(bootstrap_repeats)]
    return {"auroc": estimate_all,
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "unique_roots": len(roots)}


def analyze(rows: list[dict], bootstrap_repeats: int = 1000) -> dict:
    results = {}
    detectors = sorted(set(row["detector"] for row in rows))
    for detector in detectors:
        subset = [row for row in rows if row["detector"] == detector]
        for row in subset:
            row["score"] = float(row["score"])
            if not math.isfinite(row["score"]):
                raise ValueError(f"non-finite score for {row['sample_id']}")
        validation_real = [row for row in subset if row["split"] == "validation"
                           and row["kind"] == "pristine"]
        if len({row["root_id"] for row in validation_real}) != len(validation_real):
            raise ValueError("duplicate validation pristine root")
        threshold, actual_val_fpr = threshold_at_fpr(
            [row["score"] for row in validation_real])
        test = [row for row in subset if row["split"] == "test"]
        controls = {row["root_id"]: row["score"] for row in test
                    if row["kind"] == "pristine"}
        if len(controls) != sum(row["kind"] == "pristine" for row in test):
            raise ValueError("duplicate test pristine root")
        area_results = {}
        paired_location = {}
        for band in AREA_ORDER:
            edited = [row for row in test if row["kind"] == "edited"
                      and row["area_band"] == band]
            positives = defaultdict(list)
            for row in edited:
                if row["root_id"] not in controls:
                    raise ValueError(f"missing test pristine control for {row['root_id']}")
                positives[row["root_id"]].append(row["score"])
            area_results[band] = {
                "edited_images": len(edited),
                "unique_roots": len(positives),
                "recall_at_validation_5pct_fpr": (
                    sum(row["score"] > threshold for row in edited) / len(edited)
                    if edited else None),
                **auroc_with_root_ci(positives, controls, bootstrap_repeats),
            }
            per_root = defaultdict(lambda: defaultdict(list))
            for row in edited:
                if row["location"] in {"foreground", "background"}:
                    per_root[row["root_id"]][row["location"]].append(
                        float(row["score"] > threshold))
            paired = [value for value in per_root.values()
                      if value["foreground"] and value["background"]]
            differences = [np.mean(x["foreground"]) - np.mean(x["background"])
                           for x in paired]
            paired_location[band] = {
                "paired_roots": len(paired),
                "foreground_minus_background_detection_rate": (
                    float(np.mean(differences)) if differences else None),
            }
        results[detector] = {
            "validation_pristine_roots": len(validation_real),
            "validation_threshold": threshold,
            "achieved_validation_fpr": actual_val_fpr,
            "test_pristine_roots": len(controls),
            "test_edited_images": sum(row["kind"] == "edited" for row in test),
            "test_edited_location_counts": dict(Counter(row["location"] for row in test
                                                        if row["kind"] == "edited")),
            "area": area_results,
            "paired_location": paired_location,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    with args.predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = analyze(rows, args.bootstrap_repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"detectors": list(result), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
