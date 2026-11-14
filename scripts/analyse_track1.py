#!/usr/bin/env python3
"""Create Track 1 grounding-robustness outcomes from normalized predictions."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from benchmark_common import (
    DEFAULT_CONFIG,
    box_center,
    center_hits_any,
    clustered_bootstrap_interval,
    format_float,
    load_config,
    parse_boxes,
    proportion,
    read_csv,
    resolve_path,
    supports_box_grounding,
    truthy,
    validated_prediction_paths,
    write_csv,
)


SPATIAL_TYPES = {"laterality", "vertical"}
OUTCOMES = [
    "source_evidence_adherence",
    "corrupted_text_following",
    "abstention",
    "other_mixed",
    "invalid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def target_zone_hit(box: list[float], corruption_type: str, changed_to: str) -> bool:
    x, y = box_center(box)
    target = changed_to.strip().lower()
    if corruption_type == "laterality":
        # Standard frontal display: image right corresponds to patient left.
        if target == "left":
            return x > 0.5
        if target == "right":
            return x < 0.5
        return False
    if corruption_type == "vertical":
        if any(word in target for word in ("upper", "apical", "apex")):
            return y < 0.5
        if any(word in target for word in ("lower", "basal", "base")):
            return y > 0.5
    return False


def stable_seed(base: int, *values: str) -> int:
    return base + sum((index + 1) * ord(char) for index, char in enumerate("|".join(values)))


def representative_center(boxes: list[list[float]]) -> tuple[float, float] | None:
    if not boxes:
        return None
    centers = [box_center(box) for box in boxes]
    return mean(x for x, _ in centers), mean(y for _, y in centers)


def box_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def maximum_pairwise_iou(source: list[list[float]], corrupted: list[list[float]]) -> float | None:
    if not source or not corrupted:
        return None
    return max(box_iou(first, second) for first in source for second in corrupted)


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    manifest_rows = read_csv(resolve_path(config, "manifest"))
    manifest_groups: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in manifest_rows:
        if row["corruption_type"] in SPATIAL_TYPES:
            manifest_groups[row["pair_id"]][row["variant"]] = row

    results_root = resolve_path(config, "results")
    prediction_paths = validated_prediction_paths(config, manifest_rows)
    if not prediction_paths:
        raise SystemExit(f"No normalized_predictions.csv files found under {results_root}")

    sample_rows: list[dict[str, Any]] = []
    for prediction_path in prediction_paths:
        if not supports_box_grounding(prediction_path):
            print(f"Skipping {prediction_path.parent.name}: no native box output; use score/heatmap analysis")
            continue
        predictions = read_csv(prediction_path)
        prediction_groups: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        default_model = prediction_path.parent.name
        for row in predictions:
            prediction_groups[row.get("pair_id", "")][row.get("variant", "")] = row

        for pair_id, pair in manifest_groups.items():
            source_manifest = pair.get("source")
            corrupted_manifest = pair.get("corrupted")
            if not source_manifest or not corrupted_manifest:
                continue
            source_prediction = prediction_groups.get(pair_id, {}).get("source")
            corrupted_prediction = prediction_groups.get(pair_id, {}).get("corrupted")
            model = (
                (source_prediction or {}).get("model")
                or (corrupted_prediction or {}).get("model")
                or default_model
            )
            gt_boxes = parse_boxes(source_manifest["source_gt_boxes"])

            source_valid = bool(source_prediction) and truthy(source_prediction.get("valid_output"))
            source_boxes: list[list[float]] = []
            if source_valid:
                try:
                    source_boxes = parse_boxes(source_prediction.get("boxes_xyxy_normalized"))
                except Exception:
                    source_valid = False
            source_success = source_valid and any(center_hits_any(box, gt_boxes) for box in source_boxes)

            corrupted_valid = bool(corrupted_prediction) and truthy(corrupted_prediction.get("valid_output"))
            corrupted_boxes: list[list[float]] = []
            if corrupted_valid:
                try:
                    corrupted_boxes = parse_boxes(corrupted_prediction.get("boxes_xyxy_normalized"))
                except Exception:
                    corrupted_valid = False

            source_hit = False
            corrupted_zone_hit = False
            if not corrupted_valid:
                outcome = "invalid"
            elif not corrupted_boxes:
                outcome = "abstention"
            else:
                source_hit = any(center_hits_any(box, gt_boxes) for box in corrupted_boxes)
                corrupted_zone_hit = any(
                    target_zone_hit(box, source_manifest["corruption_type"], source_manifest["changed_to"])
                    for box in corrupted_boxes
                )
                if source_hit and not corrupted_zone_hit:
                    outcome = "source_evidence_adherence"
                elif corrupted_zone_hit and not source_hit:
                    outcome = "corrupted_text_following"
                else:
                    outcome = "other_mixed"

            source_center = representative_center(source_boxes)
            corrupted_center = representative_center(corrupted_boxes)
            displacement = None
            if source_center is not None and corrupted_center is not None:
                displacement = math.dist(source_center, corrupted_center)
            prediction_iou = maximum_pairwise_iou(source_boxes, corrupted_boxes)

            sample_rows.append(
                {
                    "model": model,
                    "pair_id": pair_id,
                    "patient_id": source_manifest["patient_id"],
                    "study_id": source_manifest["study_id"],
                    "corruption_type": source_manifest["corruption_type"],
                    "corruption_direction": source_manifest["corruption_direction"],
                    "finding_label": source_manifest["finding_label"],
                    "source_finding_train_status": source_manifest["source_finding_train_status"],
                    "corrupted_finding_train_status": source_manifest["corrupted_finding_train_status"],
                    "source_valid_output": source_valid,
                    "source_box_count": len(source_boxes),
                    "source_success": source_success,
                    "corrupted_valid_output": corrupted_valid,
                    "corrupted_box_count": len(corrupted_boxes),
                    "source_region_hit": source_hit,
                    "corrupted_zone_hit": corrupted_zone_hit,
                    "behavioural_outcome": outcome,
                    "paired_boxes": bool(source_boxes and corrupted_boxes),
                    "center_displacement": "" if displacement is None else displacement,
                    "prediction_box_iou": "" if prediction_iou is None else prediction_iou,
                    "prediction_file": str(prediction_path),
                }
            )

    combined = results_root / "combined"
    write_csv(combined / "track1_sample_results.csv", sample_rows)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        model, kind, direction = str(row["model"]), str(row["corruption_type"]), str(row["corruption_direction"])
        grouped[(model, kind, "overall")].append(row)
        grouped[(model, kind, direction)].append(row)

    replicates = int(config["bootstrap_replicates"])
    base_seed = int(config["bootstrap_seed"])
    summary_rows: list[dict[str, Any]] = []
    for (model, corruption_type, direction), all_rows in sorted(grouped.items()):
        populations = {
            "all_eligible": all_rows,
            "model_original_success": [row for row in all_rows if row["source_success"]],
        }
        for population, rows in populations.items():
            if not rows:
                continue
            summary: dict[str, Any] = {
                "model": model,
                "corruption_type": corruption_type,
                "corruption_direction": direction,
                "population": population,
                "n": len(rows),
                "patients": len({row["patient_id"] for row in rows}),
                "studies": len({row["study_id"] for row in rows}),
                "source_success_rate": format_float(proportion(rows, lambda row: bool(row["source_success"]))),
                "valid_corrupted_output_rate": format_float(
                    proportion(rows, lambda row: bool(row["corrupted_valid_output"]))
                ),
            }
            for outcome in OUTCOMES:
                statistic = lambda sample, target=outcome: proportion(
                    sample, lambda row: row["behavioural_outcome"] == target
                )
                value = statistic(rows)
                low, high = clustered_bootstrap_interval(
                    rows,
                    "patient_id",
                    statistic,
                    replicates,
                    stable_seed(base_seed, model, corruption_type, direction, population, outcome),
                )
                summary[f"{outcome}_rate"] = format_float(value)
                summary[f"{outcome}_ci_low"] = format_float(low)
                summary[f"{outcome}_ci_high"] = format_float(high)
            paired = [row for row in rows if row["paired_boxes"]]
            summary["paired_box_n"] = len(paired)
            if paired:
                for field in ("center_displacement", "prediction_box_iou"):
                    mean_stat = lambda sample, key=field: mean(float(row[key]) for row in sample)
                    low, high = clustered_bootstrap_interval(
                        paired,
                        "patient_id",
                        mean_stat,
                        replicates,
                        stable_seed(base_seed, model, corruption_type, direction, population, field),
                    )
                    summary[f"mean_{field}"] = format_float(mean_stat(paired))
                    summary[f"mean_{field}_ci_low"] = format_float(low)
                    summary[f"mean_{field}_ci_high"] = format_float(high)
                    summary[f"median_{field}"] = format_float(median(float(row[field]) for row in paired))
            summary_rows.append(summary)

    write_csv(combined / "track1_summary.csv", summary_rows)
    print(f"Wrote {len(sample_rows)} sample rows and {len(summary_rows)} summary rows to {combined}")


if __name__ == "__main__":
    main()

