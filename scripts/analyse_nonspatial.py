#!/usr/bin/env python3
"""Summarize polarity and finding-substitution spatial-output behaviour."""

from __future__ import annotations

import argparse
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


NONSPATIAL_TARGET_TYPES = {"polarity", "finding_substitution_proxy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


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


def stable_seed(base: int, *values: str) -> int:
    return base + sum((index + 1) * ord(char) for index, char in enumerate("|".join(values)))


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    manifest_groups: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    manifest_rows = read_csv(resolve_path(config, "manifest"))
    for row in manifest_rows:
        if row["corruption_type"] in NONSPATIAL_TARGET_TYPES:
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
        prediction_groups: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        for row in read_csv(prediction_path):
            prediction_groups[row.get("pair_id", "")][row.get("variant", "")] = row
        default_model = prediction_path.parent.name

        for pair_id, pair in manifest_groups.items():
            source_manifest = pair.get("source")
            if not source_manifest:
                continue
            predictions = prediction_groups.get(pair_id, {})
            source_prediction = predictions.get("source")
            corrupted_prediction = predictions.get("corrupted")
            model = (
                (source_prediction or {}).get("model")
                or (corrupted_prediction or {}).get("model")
                or default_model
            )

            source_valid = bool(source_prediction) and truthy(source_prediction.get("valid_output"))
            corrupted_valid = bool(corrupted_prediction) and truthy(corrupted_prediction.get("valid_output"))
            source_boxes: list[list[float]] = []
            corrupted_boxes: list[list[float]] = []
            if source_valid:
                try:
                    source_boxes = parse_boxes(source_prediction.get("boxes_xyxy_normalized"))
                except Exception:
                    source_valid = False
            if corrupted_valid:
                try:
                    corrupted_boxes = parse_boxes(corrupted_prediction.get("boxes_xyxy_normalized"))
                except Exception:
                    corrupted_valid = False

            source_center = representative_center(source_boxes)
            corrupted_center = representative_center(corrupted_boxes)
            displacement = None
            if source_center is not None and corrupted_center is not None:
                displacement = math.dist(source_center, corrupted_center)
            prediction_iou = maximum_pairwise_iou(source_boxes, corrupted_boxes)
            gt_boxes = parse_boxes(source_manifest["source_gt_boxes"])
            source_region_hit = bool(gt_boxes) and any(center_hits_any(box, gt_boxes) for box in source_boxes)
            corrupted_source_region_hit = bool(gt_boxes) and any(
                center_hits_any(box, gt_boxes) for box in corrupted_boxes
            )

            sample_rows.append(
                {
                    "model": model,
                    "pair_id": pair_id,
                    "patient_id": source_manifest["patient_id"],
                    "study_id": source_manifest["study_id"],
                    "corruption_type": source_manifest["corruption_type"],
                    "corruption_direction": source_manifest["corruption_direction"],
                    "benchmark_group": source_manifest["benchmark_group"],
                    "source_finding_train_status": source_manifest["source_finding_train_status"],
                    "corrupted_finding_train_status": source_manifest["corrupted_finding_train_status"],
                    "source_valid_output": source_valid,
                    "corrupted_valid_output": corrupted_valid,
                    "source_box_count": len(source_boxes),
                    "corrupted_box_count": len(corrupted_boxes),
                    "paired_boxes": bool(source_boxes and corrupted_boxes),
                    "source_region_available": bool(gt_boxes),
                    "source_region_hit": source_region_hit,
                    "corrupted_source_region_hit": corrupted_source_region_hit,
                    "center_displacement": "" if displacement is None else displacement,
                    "prediction_box_iou": "" if prediction_iou is None else prediction_iou,
                    "prediction_file": str(prediction_path),
                }
            )

    combined = results_root / "combined"
    write_csv(combined / "nonspatial_sample_results.csv", sample_rows)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        model, kind, direction = str(row["model"]), str(row["corruption_type"]), str(row["corruption_direction"])
        grouped[(model, kind, "overall")].append(row)
        if kind == "polarity":
            grouped[(model, kind, direction)].append(row)

    replicates = int(config["bootstrap_replicates"])
    base_seed = int(config["bootstrap_seed"])
    summaries: list[dict[str, Any]] = []
    for (model, corruption_type, direction), rows in sorted(grouped.items()):
        paired = [row for row in rows if row["paired_boxes"]]
        source_success = [row for row in rows if row["source_region_hit"]]
        summary: dict[str, Any] = {
            "model": model,
            "corruption_type": corruption_type,
            "corruption_direction": direction,
            "benchmark_group": rows[0]["benchmark_group"],
            "n": len(rows),
            "patients": len({row["patient_id"] for row in rows}),
            "studies": len({row["study_id"] for row in rows}),
            "paired_box_n": len(paired),
            "source_success_n": len(source_success),
        }

        proportion_metrics = {
            "source_valid_output_rate": lambda row: bool(row["source_valid_output"]),
            "corrupted_valid_output_rate": lambda row: bool(row["corrupted_valid_output"]),
            "source_box_rate": lambda row: int(row["source_box_count"]) > 0,
            "corrupted_box_rate": lambda row: int(row["corrupted_box_count"]) > 0,
        }
        if any(row["source_region_available"] for row in rows):
            proportion_metrics.update(
                {
                    "source_region_hit_rate": lambda row: bool(row["source_region_hit"]),
                    "corrupted_source_region_hit_rate": lambda row: bool(row["corrupted_source_region_hit"]),
                }
            )
        for name, predicate in proportion_metrics.items():
            statistic = lambda sample, test=predicate: proportion(sample, test)
            value = statistic(rows)
            low, high = clustered_bootstrap_interval(
                rows,
                "patient_id",
                statistic,
                replicates,
                stable_seed(base_seed, model, corruption_type, direction, name),
            )
            summary[name] = format_float(value)
            summary[f"{name}_ci_low"] = format_float(low)
            summary[f"{name}_ci_high"] = format_float(high)

        if source_success:
            retention_stat = lambda sample: proportion(
                sample, lambda row: bool(row["corrupted_source_region_hit"])
            )
            low, high = clustered_bootstrap_interval(
                source_success,
                "patient_id",
                retention_stat,
                replicates,
                stable_seed(base_seed, model, corruption_type, direction, "conditional_retention"),
            )
            summary["conditional_source_region_retention_rate"] = format_float(
                retention_stat(source_success)
            )
            summary["conditional_source_region_retention_rate_ci_low"] = format_float(low)
            summary["conditional_source_region_retention_rate_ci_high"] = format_float(high)

        if paired:
            for field in ("center_displacement", "prediction_box_iou"):
                mean_stat = lambda sample, key=field: mean(float(row[key]) for row in sample)
                low, high = clustered_bootstrap_interval(
                    paired,
                    "patient_id",
                    mean_stat,
                    replicates,
                    stable_seed(base_seed, model, corruption_type, direction, field),
                )
                summary[f"mean_{field}"] = format_float(mean_stat(paired))
                summary[f"mean_{field}_ci_low"] = format_float(low)
                summary[f"mean_{field}_ci_high"] = format_float(high)
                summary[f"median_{field}"] = format_float(median(float(row[field]) for row in paired))
        summaries.append(summary)

    write_csv(combined / "nonspatial_summary.csv", summaries)
    print(f"Wrote {len(sample_rows)} sample rows and {len(summaries)} summary rows to {combined}")


if __name__ == "__main__":
    main()
