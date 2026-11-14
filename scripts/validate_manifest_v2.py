#!/usr/bin/env python3
"""Semantic and integrity validation for PadChest-GR Benchmark v2."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_common import DEFAULT_CONFIG, load_config, parse_boxes, read_csv, resolve_path, sha256_file, write_json

VERSION = "v2"
REQUIRED = {
    "benchmark_version", "pair_id", "variant", "benchmark_group", "corruption_type",
    "corruption_direction", "patient_id", "study_id", "image_id", "image_relpath",
    "source_claim", "corrupted_claim", "tested_claim", "finding_label", "finding_labels",
    "source_finding_train_status", "corrupted_finding_train_status", "structured_locations",
    "source_gt_boxes", "changed_from", "changed_to", "claim_status",
    "phrase_groundability", "construction_rule",
}
ALLOWED = {"laterality", "vertical", "polarity", "finding_substitution_proxy"}
GROUPS = {
    "laterality": "main_benchmark", "vertical": "main_benchmark",
    "polarity": "main_benchmark", "finding_substitution_proxy": "proxy_lower_confidence",
}
POLARITY_PHRASES = {
    "infiltrates": "pulmonary infiltrates", "consolidation": "pulmonary consolidation",
    "pleural effusion": "pleural effusion", "nodule": "pulmonary nodule",
    "pneumothorax": "pneumothorax",
}
NONCANONICAL = {"apical pleural thickening", "atelectasis basal"}


def occurrence(text: str, phrase: str) -> int:
    return len(re.findall(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.I))


def validate_predictions(config: dict, manifest_rows: list[dict[str, str]], manifest_hash: str, errors: list[str], warnings: list[str]) -> list[str]:
    results_root = resolve_path(config, "results")
    expected_keys = {(row["pair_id"], row["variant"]) for row in manifest_rows}
    checked: list[str] = []
    if not results_root.exists():
        return checked
    for model_dir in sorted(path for path in results_root.iterdir() if path.is_dir() and path.name not in {"archive", "combined"}):
        prediction_path = model_dir / "normalized_predictions.csv"
        if not prediction_path.exists():
            continue
        checked.append(model_dir.name)
        metadata_path = model_dir / "run_metadata.json"
        if not metadata_path.exists():
            errors.append(f"predictions {model_dir.name}: missing run_metadata.json")
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"predictions {model_dir.name}: invalid run_metadata.json: {exc}")
            continue
        inference_hash = str(metadata.get("inference_manifest_sha256") or "")
        if inference_hash != manifest_hash:
            errors.append(
                f"predictions {model_dir.name}: inference_manifest_sha256 {inference_hash or '<missing>'} "
                f"does not equal current manifest {manifest_hash}"
            )
        rows = read_csv(prediction_path)
        keys = [(row.get("pair_id", ""), row.get("variant", "")) for row in rows]
        if len(keys) != len(set(keys)):
            errors.append(f"predictions {model_dir.name}: duplicate pair_id/variant rows")
        key_set = set(keys)
        missing, extra = expected_keys - key_set, key_set - expected_keys
        if len(rows) != len(manifest_rows) or missing or extra:
            errors.append(
                f"predictions {model_dir.name}: incomplete current-manifest coverage "
                f"(rows={len(rows)}, expected={len(manifest_rows)}, missing={len(missing)}, extra={len(extra)})"
            )
    if not checked:
        warnings.append("no active normalized predictions to validate; all models must run inference on v2")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    manifest_path, image_root = resolve_path(config, "manifest"), resolve_path(config, "image_root")
    rows = read_csv(manifest_path)
    errors: list[str] = []; warnings: list[str] = []
    if not rows:
        errors.append("manifest is empty")
    fields = set(rows[0]) if rows else set()
    if REQUIRED - fields:
        errors.append(f"missing columns: {sorted(REQUIRED - fields)}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set(); missing_images: set[str] = set()
    for line, row in enumerate(rows, 2):
        pair_id, variant, kind = row.get("pair_id", ""), row.get("variant", ""), row.get("corruption_type", "")
        key = (pair_id, variant)
        if key in seen:
            errors.append(f"line {line}: duplicate pair/variant {key}")
        seen.add(key); grouped[pair_id].append(row)
        if row.get("benchmark_version") != VERSION or not pair_id.startswith(f"{VERSION}_"):
            errors.append(f"line {line}: invalid v2 version/pair ID")
        if variant not in {"source", "corrupted"}:
            errors.append(f"line {line}: invalid variant {variant!r}")
        if kind not in ALLOWED:
            errors.append(f"line {line}: invalid corruption_type {kind!r}")
        if row.get("benchmark_group") != GROUPS.get(kind):
            errors.append(f"line {line}: wrong benchmark_group for {kind}")
        expected_claim = row.get("source_claim") if variant == "source" else row.get("corrupted_claim")
        if row.get("tested_claim") != expected_claim:
            errors.append(f"line {line}: tested_claim does not match variant")
        if row.get("source_claim") == row.get("corrupted_claim"):
            errors.append(f"line {line}: identical source/corrupted claims")
        if row.get("finding_label") in NONCANONICAL:
            errors.append(f"line {line}: noncanonical location-bearing finding_label")
        if row.get("source_finding_train_status") not in {"train_seen", "train_unseen"} or row.get("corrupted_finding_train_status") not in {"train_seen", "train_unseen"}:
            errors.append(f"line {line}: invalid train-seen status")
        try:
            boxes = parse_boxes(row.get("source_gt_boxes"))
            if any(not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1) for box in boxes):
                errors.append(f"line {line}: invalid normalized box")
        except Exception as exc:
            errors.append(f"line {line}: {exc}")
        for field_name in ("structured_locations", "finding_labels"):
            try:
                parsed = json.loads(row.get(field_name) or "[]")
                if not isinstance(parsed, list):
                    raise ValueError("not a list")
            except Exception as exc:
                errors.append(f"line {line}: invalid {field_name}: {exc}")
        image_path = image_root / row.get("image_relpath", "")
        if not image_path.is_file():
            missing_images.add(str(image_path))

    invariant = [
        "benchmark_version", "benchmark_group", "corruption_type", "corruption_direction",
        "patient_id", "study_id", "image_id", "source_claim", "corrupted_claim",
        "finding_label", "finding_labels", "source_gt_boxes", "changed_from", "changed_to",
        "construction_rule",
    ]
    pair_counts: Counter[str] = Counter(); direction_counts: Counter[tuple[str, str]] = Counter(); group_counts: Counter[str] = Counter()
    polarity_category_counts: Counter[tuple[str, str]] = Counter()
    for pair_id, members in grouped.items():
        if len(members) != 2 or {row.get("variant") for row in members} != {"source", "corrupted"}:
            errors.append(f"pair {pair_id}: expected exactly source+corrupted rows")
            continue
        for field in invariant:
            if len({row.get(field) for row in members}) != 1:
                errors.append(f"pair {pair_id}: inconsistent {field}")
        row = members[0]; kind = row["corruption_type"]; direction = row["corruption_direction"]
        pair_counts[kind] += 1; direction_counts[(kind, direction)] += 1; group_counts[row["benchmark_group"]] += 1
        source, corrupt = row["source_claim"], row["corrupted_claim"]
        boxes = parse_boxes(row["source_gt_boxes"])
        if kind in {"laterality", "vertical"}:
            old, new = row["changed_from"], row["changed_to"]
            if occurrence(source, old) != 1 or occurrence(source, new) != 0:
                errors.append(f"pair {pair_id}: source spatial terms are not atomic/nonconflicting")
            if occurrence(corrupt, old) != 0 or occurrence(corrupt, new) != 1:
                errors.append(f"pair {pair_id}: corrupted spatial terms are not atomic/nonconflicting")
        elif kind == "polarity":
            label = row["finding_label"]; phrase = POLARITY_PHRASES.get(label)
            polarity_category_counts[(direction, label)] += 1
            labels = json.loads(row["finding_labels"])
            if phrase is None or labels != [label]:
                errors.append(f"pair {pair_id}: polarity must contain one canonical finding")
                continue
            positive, negative = f"Evidence of {phrase}.", f"No evidence of {phrase}."
            if direction == "negative_to_positive":
                if (source, corrupt) != (negative, positive) or boxes:
                    errors.append(f"pair {pair_id}: invalid negative_to_positive template/bbox")
            elif direction == "positive_to_negative":
                if (source, corrupt) != (positive, negative) or not boxes:
                    errors.append(f"pair {pair_id}: invalid positive_to_negative template/bbox")
            else:
                errors.append(f"pair {pair_id}: invalid polarity direction")
            if not negative.startswith("No ") or negative[3:].casefold() != positive.casefold():
                errors.append(f"pair {pair_id}: polarity claims differ by more than negation")
        else:
            if direction != "finding_to_finding" or not source.startswith("Evidence of ") or not corrupt.startswith("Evidence of "):
                errors.append(f"pair {pair_id}: invalid substitution template/direction")

    expected_pairs = {key: int(value) for key, value in config["expected_pair_counts"].items()}
    if dict(pair_counts) != expected_pairs:
        errors.append(f"pair counts {dict(pair_counts)} != {expected_pairs}")
    for kind, expected in config["expected_direction_counts"].items():
        for direction, count in expected.items():
            if direction_counts[(kind, direction)] != int(count):
                errors.append(f"direction count {kind}/{direction}={direction_counts[(kind, direction)]}, expected {count}")
    expected_groups = {key: int(value) for key, value in config["expected_group_pair_counts"].items()}
    if dict(group_counts) != expected_groups:
        errors.append(f"benchmark group counts {dict(group_counts)} != {expected_groups}")
    for direction in ("negative_to_positive", "positive_to_negative"):
        for label, count in config["polarity_category_quota_per_direction"].items():
            if polarity_category_counts[(direction, label)] != int(count):
                errors.append(
                    f"polarity category count {direction}/{label}="
                    f"{polarity_category_counts[(direction, label)]}, expected {count}"
                )
    if len(rows) != 1020 or len(grouped) != 510:
        errors.append(f"fixed size violation: rows={len(rows)}, pairs={len(grouped)}")
    if missing_images:
        errors.append(f"{len(missing_images)} referenced images are missing")
    actual_hash = sha256_file(manifest_path)
    hash_path = resolve_path(config, "manifest_hash")
    frozen_hash = hash_path.read_text(encoding="utf-8").strip() if hash_path.exists() else ""
    if actual_hash != frozen_hash:
        errors.append("manifest SHA-256 does not match manifest_sha256.txt")
    checked_predictions = validate_predictions(config, rows, actual_hash, errors, warnings)
    multi_box_counts = Counter()
    for row in rows:
        if row["variant"] == "source" and len(parse_boxes(row["source_gt_boxes"])) > 1:
            multi_box_counts[row["corruption_type"]] += 1
    summary = {
        "valid": not errors, "benchmark_version": VERSION, "manifest": str(manifest_path),
        "manifest_sha256": actual_hash, "rows": len(rows), "pairs": len(grouped),
        "patients": len({row["patient_id"] for row in rows}), "studies": len({row["study_id"] for row in rows}),
        "pair_counts": dict(sorted(pair_counts.items())),
        "direction_pair_counts": {f"{kind}:{direction}": count for (kind, direction), count in sorted(direction_counts.items())},
        "polarity_category_pair_counts": {f"{direction}:{label}": count for (direction, label), count in sorted(polarity_category_counts.items())},
        "benchmark_group_pair_counts": dict(sorted(group_counts.items())),
        "multi_box_pair_counts": dict(sorted(multi_box_counts.items())),
        "checked_prediction_models": checked_predictions, "errors": errors, "warnings": warnings,
    }
    write_json(resolve_path(config, "validation_summary"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
