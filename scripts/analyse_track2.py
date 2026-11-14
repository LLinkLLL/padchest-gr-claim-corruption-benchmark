#!/usr/bin/env python3
"""Create Track 2 paired verification-sensitivity summaries."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from benchmark_common import (
    DEFAULT_CONFIG,
    clustered_bootstrap_interval,
    format_float,
    load_config,
    proportion,
    read_csv,
    resolve_path,
    truthy,
    validated_prediction_paths,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def numeric_score(row: dict[str, str] | None) -> float | None:
    if not row or not truthy(row.get("valid_output")) or row.get("score", "").strip() == "":
        return None
    try:
        value = float(row["score"])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def auc(rows: list[dict[str, Any]]) -> float:
    samples: list[tuple[float, int]] = []
    for row in rows:
        samples.append((float(row["source_score"]), 1))
        samples.append((float(row["corrupted_score"]), 0))
    positives = sum(label for _, label in samples)
    negatives = len(samples) - positives
    if not positives or not negatives:
        return float("nan")
    ordered = sorted(enumerate(samples), key=lambda item: item[1][0])
    ranks = [0.0] * len(samples)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1][0] == ordered[cursor][1][0]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for index in range(cursor, end):
            ranks[ordered[index][0]] = average_rank
        cursor = end
    positive_rank_sum = sum(rank for rank, (_, label) in zip(ranks, samples) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def stable_seed(base: int, *values: str) -> int:
    return base + sum((index + 1) * ord(char) for index, char in enumerate("|".join(values)))


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    manifest_rows = read_csv(resolve_path(config, "manifest"))
    manifest_groups: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in manifest_rows:
        manifest_groups[row["pair_id"]][row["variant"]] = row

    results_root = resolve_path(config, "results")
    prediction_paths = validated_prediction_paths(config, manifest_rows)
    if not prediction_paths:
        raise SystemExit(f"No normalized_predictions.csv files found under {results_root}")

    paired_rows: list[dict[str, Any]] = []
    for prediction_path in prediction_paths:
        predictions = read_csv(prediction_path)
        if not any(numeric_score(row) is not None for row in predictions):
            print(f"Skipping {prediction_path.parent.name}: no meaningful numeric score")
            continue
        prediction_groups: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        default_model = prediction_path.parent.name
        for row in predictions:
            prediction_groups[row.get("pair_id", "")][row.get("variant", "")] = row

        for pair_id, pair in manifest_groups.items():
            source_manifest = pair.get("source")
            if not source_manifest:
                continue
            predictions_for_pair = prediction_groups.get(pair_id, {})
            source_prediction = predictions_for_pair.get("source")
            corrupted_prediction = predictions_for_pair.get("corrupted")
            source_score = numeric_score(source_prediction)
            corrupted_score = numeric_score(corrupted_prediction)
            model = (
                (source_prediction or {}).get("model")
                or (corrupted_prediction or {}).get("model")
                or default_model
            )
            score_types = {
                row.get("score_type", "").strip()
                for row in (source_prediction, corrupted_prediction)
                if row and row.get("score_type", "").strip()
            }
            score_type = next(iter(score_types)) if len(score_types) == 1 else "mixed_or_unspecified"
            valid_pair = source_score is not None and corrupted_score is not None
            delta = source_score - corrupted_score if valid_pair else None
            paired_rows.append(
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
                    "score_type": score_type,
                    "valid_score_pair": valid_pair,
                    "source_score": "" if source_score is None else source_score,
                    "corrupted_score": "" if corrupted_score is None else corrupted_score,
                    "source_minus_corrupted": "" if delta is None else delta,
                    "source_higher": bool(valid_pair and delta > 0),
                    "tie": bool(valid_pair and delta == 0),
                    "corrupted_higher": bool(valid_pair and delta < 0),
                    "prediction_file": str(prediction_path),
                }
            )

    combined = results_root / "combined"
    write_csv(combined / "track2_sample_results.csv", paired_rows)

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        model = str(row["model"]); kind = str(row["corruption_type"])
        score_type = str(row["score_type"]); direction = str(row["corruption_direction"])
        grouped[(model, kind, score_type, "overall")].append(row)
        grouped[(model, kind, score_type, direction)].append(row)

    replicates = int(config["bootstrap_replicates"])
    base_seed = int(config["bootstrap_seed"])
    summary_rows: list[dict[str, Any]] = []
    for (model, corruption_type, score_type, direction), rows in sorted(grouped.items()):
        valid = [row for row in rows if row["valid_score_pair"]]
        valid_rate_stat = lambda sample: proportion(sample, lambda row: bool(row["valid_score_pair"]))
        valid_low, valid_high = clustered_bootstrap_interval(
            rows,
            "patient_id",
            valid_rate_stat,
            replicates,
            stable_seed(base_seed, model, corruption_type, score_type, direction, "valid"),
        )
        summary: dict[str, Any] = {
            "model": model,
            "corruption_type": corruption_type,
            "score_type": score_type,
            "corruption_direction": direction,
            "n_total": len(rows),
            "n_valid_pairs": len(valid),
            "valid_pair_rate": format_float(valid_rate_stat(rows)),
            "valid_pair_ci_low": format_float(valid_low),
            "valid_pair_ci_high": format_float(valid_high),
        }
        if valid:
            pairwise_stat = lambda sample: proportion(sample, lambda row: bool(row["source_higher"]))
            delta_stat = lambda sample: mean(float(row["source_minus_corrupted"]) for row in sample)
            auc_stat = lambda sample: auc(sample)
            seed_prefix = (model, corruption_type, score_type, direction)
            pair_low, pair_high = clustered_bootstrap_interval(
                valid,
                "patient_id",
                pairwise_stat,
                replicates,
                stable_seed(base_seed, *seed_prefix, "pairwise"),
            )
            delta_low, delta_high = clustered_bootstrap_interval(
                valid,
                "patient_id",
                delta_stat,
                replicates,
                stable_seed(base_seed, *seed_prefix, "delta"),
            )
            auc_low, auc_high = clustered_bootstrap_interval(
                valid,
                "patient_id",
                auc_stat,
                replicates,
                stable_seed(base_seed, *seed_prefix, "auc"),
            )
            deltas = [float(row["source_minus_corrupted"]) for row in valid]
            summary.update(
                {
                    "pairwise_accuracy": format_float(pairwise_stat(valid)),
                    "pairwise_accuracy_ci_low": format_float(pair_low),
                    "pairwise_accuracy_ci_high": format_float(pair_high),
                    "mean_score_delta": format_float(mean(deltas)),
                    "mean_score_delta_ci_low": format_float(delta_low),
                    "mean_score_delta_ci_high": format_float(delta_high),
                    "median_score_delta": format_float(median(deltas)),
                    "tie_rate": format_float(proportion(valid, lambda row: bool(row["tie"]))),
                    "corrupted_higher_rate": format_float(
                        proportion(valid, lambda row: bool(row["corrupted_higher"]))
                    ),
                    "auroc": format_float(auc(valid)),
                    "auroc_ci_low": format_float(auc_low),
                    "auroc_ci_high": format_float(auc_high),
                }
            )
        summary_rows.append(summary)

    write_csv(combined / "track2_summary.csv", summary_rows)
    print(f"Wrote {len(paired_rows)} paired rows and {len(summary_rows)} summary rows to {combined}")


if __name__ == "__main__":
    main()

