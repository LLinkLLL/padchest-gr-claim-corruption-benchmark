"""Frozen, inference-free point and area-weighted heatmap metrics for BioViL-T."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

import numpy as np

MODEL = "biovilt_official"
CNR_EPS = 1e-8
TOP_FRACTION = 0.05
NATIVE_METRICS = ("signed_cnr", "mu_gt", "mu_background", "zone_contrast", "top5_mean", "map_mean")
OUTCOMES = ("source_evidence_adherence", "corrupted_text_following", "other_mixed", "invalid")


def intersection(a, b):
    r = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return r if r[2] > r[0] and r[3] > r[1] else None


def union_area(rects):
    """Exact rectangle-union area by x sweep; never double counts overlapping GT."""
    rects = [r for r in rects if r is not None and r[2] > r[0] and r[3] > r[1]]
    xs = sorted({r[i] for r in rects for i in (0, 2)})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((r[1], r[3]) for r in rects if r[0] < right and r[2] > left)
        if not intervals:
            continue
        low, high = intervals[0]
        length = 0.0
        for a, b in intervals[1:]:
            if a > high:
                length += high - low
                low, high = a, b
            else:
                high = max(high, b)
        area += (right - left) * (length + high - low)
    return area


def zone_rect(kind: str, location: str):
    t = location.strip().lower()
    if kind == "laterality":
        if t == "left":
            return (0.5, 0.0, 1.0, 1.0)
        if t == "right":
            return (0.0, 0.0, 0.5, 1.0)
    elif kind == "vertical":
        if any(w in t for w in ("upper", "apical", "apex")):
            return (0.0, 0.0, 1.0, 0.5)
        if any(w in t for w in ("lower", "basal", "base")):
            return (0.0, 0.5, 1.0, 1.0)
    raise ValueError(f"Unsupported spatial target: {kind}/{location}")


def point_in_zone(point, kind: str, location: str):
    if point is None:
        return False
    x, y = point
    rect = zone_rect(kind, location)
    if kind == "laterality":
        return x > 0.5 if rect[0] == 0.5 else x < 0.5
    return y > 0.5 if rect[1] == 0.5 else y < 0.5


def point_hits(point, boxes):
    return point is not None and any(a <= point[0] <= c and b <= point[1] <= d for a, b, c, d in boxes)


def geometry(width: int, height: int, shape, boxes, kind: str, before: str, after: str):
    if width <= 0 or height <= 0 or len(shape) != 2 or min(shape) <= 0:
        raise ValueError("Invalid image/grid geometry")
    h, w = shape
    size = math.floor(448 * min(width, height) / 512)
    left, top = math.floor((width - size) / 2), math.floor((height - size) / 2)
    xe = (left + np.arange(w + 1, dtype=np.float64) * size / w) / width
    ye = (top + np.arange(h + 1, dtype=np.float64) * size / h) / height
    crop = (float(xe[0]), float(ye[0]), float(xe[-1]), float(ye[-1]))
    total = np.outer(np.diff(ye), np.diff(xe))
    foreground = np.zeros(shape, dtype=np.float64)
    target = np.zeros(shape, dtype=np.float64)
    original = np.zeros(shape, dtype=np.float64)
    target_rect, original_rect = zone_rect(kind, after), zone_rect(kind, before)
    for r in range(h):
        for c in range(w):
            tile = (xe[c], ye[r], xe[c + 1], ye[r + 1])
            foreground[r, c] = union_area([intersection(tile, b) for b in boxes])
            target[r, c] = union_area([intersection(tile, target_rect)])
            original[r, c] = union_area([intersection(tile, original_rect)])
    foreground = np.minimum(total, np.maximum(0.0, foreground))
    gt_area, visible = union_area(boxes), float(foreground.sum())
    fraction = visible / gt_area if gt_area > 0 else float("nan")
    fully_inside = all(b[0] >= crop[0] and b[1] >= crop[1] and b[2] <= crop[2] and b[3] <= crop[3] for b in boxes)
    crop_status = "fully_outside" if visible == 0 else "fully_visible" if fully_inside else "partially_visible"
    return dict(x_edges=xe, y_edges=ye, crop=crop, crop_size=size, crop_left=left, crop_top=top,
                total=total, foreground=foreground, background=np.maximum(0.0, total - foreground),
                target=target, original=original, gt_visible_fraction=fraction, crop_status=crop_status,
                gt_union_area=gt_area, gt_visible_area=visible)


def peak_info(array, geom):
    h = np.asarray(array, dtype=np.float64)
    constant = bool(np.all(h == h.flat[0]))
    maximum = float(h.max())
    indexes = np.flatnonzero(h.ravel() == maximum)
    result = dict(constant_map=constant, peak_tie_count=len(indexes), peak_value=maximum,
                  map_min=float(h.min()), map_max=maximum, map_range=float(np.ptp(h)), point=None,
                  peak_row=None, peak_col=None)
    if not constant:
        r, c = np.unravel_index(int(indexes[0]), h.shape)
        x = float((geom["x_edges"][c] + geom["x_edges"][c + 1]) / 2)
        y = float((geom["y_edges"][r] + geom["y_edges"][r + 1]) / 2)
        result.update(point=(x, y), peak_row=int(r), peak_col=int(c))
    return result


def weighted_stats(h, weights):
    mass = float(np.sum(weights))
    if mass <= 0:
        return float("nan"), float("nan")
    mu = float(np.sum(weights * h, dtype=np.float64) / mass)
    var = float(np.sum(weights * (h - mu) ** 2, dtype=np.float64) / mass)
    return mu, var


def native_metrics(array, geom):
    h = np.asarray(array, dtype=np.float64)
    mu_f, var_f = weighted_stats(h, geom["foreground"])
    mu_b, var_b = weighted_stats(h, geom["background"])
    mu_t, _ = weighted_stats(h, geom["target"])
    mu_o, _ = weighted_stats(h, geom["original"])
    k = max(1, math.ceil(TOP_FRACTION * h.size))
    reason = ""
    denom = math.sqrt(var_f + var_b) if math.isfinite(var_f + var_b) else float("nan")
    if not math.isfinite(mu_f):
        reason = "gt_fully_outside_crop"
    elif not math.isfinite(mu_b):
        reason = "no_visible_background"
    elif denom <= CNR_EPS:
        reason = "zero_or_negligible_contrast_denominator"
    cnr = (mu_f - mu_b) / denom if not reason else float("nan")
    result = dict(signed_cnr=cnr, mu_gt=mu_f, mu_background=mu_b, var_gt=var_f, var_background=var_b,
                  cnr_denominator=denom, gt_background_difference=mu_f - mu_b,
                  mu_target=mu_t, mu_original_zone=mu_o, zone_contrast=mu_t - mu_o,
                  top5_mean=float(np.partition(h.ravel(), h.size - k)[-k:].mean()),
                  map_mean=float(h.mean()), top_k=k)
    reasons = {key: "" for key in NATIVE_METRICS}
    reasons["signed_cnr"] = reason
    if not math.isfinite(mu_f):
        reasons["mu_gt"] = "gt_fully_outside_crop"
    if not math.isfinite(mu_b):
        reasons["mu_background"] = "no_visible_background"
    if not math.isfinite(mu_t - mu_o):
        reasons["zone_contrast"] = "no_visible_target_or_original_zone"
    result["reasons"] = reasons
    return result


def classify(corrupted_valid, point, boxes, kind, after):
    if not corrupted_valid:
        return "invalid", False, False
    if point is None:
        return "other_mixed", False, False
    a, c = point_hits(point, boxes), point_in_zone(point, kind, after)
    if a and not c:
        return "source_evidence_adherence", a, c
    if c and not a:
        return "corrupted_text_following", a, c
    return "other_mixed", a, c


def stable_seed(base: int, *values: str):
    return base + sum((index + 1) * ord(char) for index, char in enumerate("|".join(values)))


def cluster_mean_ci(rows: list[dict[str, Any]], getter, seed: int, replicates: int = 1000):
    """Patient-cluster percentile CI, identical draws to shared random.choices.

    Aggregate sums/counts only for speed. Missing metric values remain missing;
    all patients in the analysis population are retained in the sampling frame.
    """
    if not rows:
        return dict(mean=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
                    n_valid_pairs=0, n_valid_patients=0, valid_replicates=0, median=float("nan"))
    clusters = defaultdict(list)
    values = []
    for row in rows:
        v = float(getter(row))
        clusters[str(row["patient_id"])].append(v)
        if math.isfinite(v):
            values.append(v)
    keys = sorted(clusters)
    valid_clusters = [[v for v in clusters[k] if math.isfinite(v)] for k in keys]
    sums = np.array([sum(vs) for vs in valid_clusters], dtype=np.float64)
    counts = np.array([len(vs) for vs in valid_clusters], dtype=np.int64)
    rng = random.Random(seed)
    draws = np.array([rng.choices(range(len(keys)), k=len(keys)) for _ in range(replicates)], dtype=np.int64)
    denominators = counts[draws].sum(axis=1)
    numerators = sums[draws].sum(axis=1)
    estimates = numerators[denominators > 0] / denominators[denominators > 0]
    ci = np.quantile(estimates, [0.025, 0.975]) if len(estimates) else [float("nan")] * 2
    return dict(mean=float(np.mean(values)) if values else float("nan"), ci_low=float(ci[0]), ci_high=float(ci[1]),
                n_valid_pairs=len(values), n_valid_patients=sum(bool(vs) for vs in valid_clusters),
                valid_replicates=len(estimates), median=float(np.median(values)) if values else float("nan"))
