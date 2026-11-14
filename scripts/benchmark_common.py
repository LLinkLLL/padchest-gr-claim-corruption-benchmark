"""Shared standard-library utilities for the claim corruption benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BENCHMARK_ROOT / "configs" / "benchmark.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(path.resolve())
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    path = Path(config["paths"][key])
    return path.resolve() if path.is_absolute() else (BENCHMARK_ROOT / path).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validated_prediction_paths(
    config: dict[str, Any], manifest_rows: list[dict[str, str]]
) -> list[Path]:
    """Return only complete predictions produced against the current manifest.

    The inference hash is immutable provenance. A post-hoc synchronization may
    not replace it, and every active prediction file must cover all manifest
    rows exactly once.
    """
    manifest_path = resolve_path(config, "manifest")
    current_hash = sha256_file(manifest_path)
    expected = {(row["pair_id"], row["variant"]) for row in manifest_rows}
    results_root = resolve_path(config, "results")
    paths = sorted(
        path
        for path in results_root.glob("*/normalized_predictions.csv")
        if path.parent.name not in {"combined", "archive"}
    )
    for path in paths:
        metadata_path = path.parent / "run_metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"{path.parent.name}: missing run_metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        inference_hash = str(metadata.get("inference_manifest_sha256") or "")
        if inference_hash != current_hash:
            raise ValueError(
                f"{path.parent.name}: inference manifest hash mismatch "
                f"({inference_hash or '<missing>'} != {current_hash}); rerun inference"
            )
        rows = read_csv(path)
        keys = [(row.get("pair_id", ""), row.get("variant", "")) for row in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected or len(rows) != len(manifest_rows):
            raise ValueError(
                f"{path.parent.name}: predictions do not exactly cover the current "
                f"manifest ({len(rows)} rows, expected {len(manifest_rows)})"
            )
    return paths


def supports_box_grounding(prediction_path: Path) -> bool:
    """Heatmap/score-only outputs must not be counted as box abstention.

    Existing native-box runs remain eligible when the capability is omitted.
    New non-box adapters explicitly set supports_box_grounding=false.
    """
    metadata_path = prediction_path.parent / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("supports_box_grounding", True) is not False


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_boxes(value: Any) -> list[list[float]]:
    if value in (None, ""):
        return []
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise ValueError("boxes must be a JSON list")
    boxes: list[list[float]] = []
    for box in parsed:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"invalid box: {box!r}")
        numeric = [float(item) for item in box]
        if not all(math.isfinite(item) for item in numeric):
            raise ValueError(f"non-finite box: {box!r}")
        boxes.append(numeric)
    return boxes


def box_center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def center_hits_any(box: list[float], targets: list[list[float]]) -> bool:
    x, y = box_center(box)
    return any(target[0] <= x <= target[2] and target[1] <= y <= target[3] for target in targets)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def clustered_bootstrap_interval(
    rows: list[dict[str, Any]],
    cluster_key: str,
    statistic: Callable[[list[dict[str, Any]]], float],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if not rows:
        return float("nan"), float("nan")
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row[cluster_key])].append(row)
    keys = sorted(clusters)
    if not keys:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sample: list[dict[str, Any]] = []
        for key in rng.choices(keys, k=len(keys)):
            sample.extend(clusters[key])
        estimate = statistic(sample)
        if math.isfinite(estimate):
            estimates.append(estimate)
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def proportion(rows: Iterable[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float:
    values = list(rows)
    return sum(bool(predicate(row)) for row in values) / len(values) if values else float("nan")


def average(rows: Iterable[dict[str, Any]], getter: Callable[[dict[str, Any]], float]) -> float:
    values = [float(getter(row)) for row in rows]
    return mean(values) if values else float("nan")


def format_float(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.6f}"

