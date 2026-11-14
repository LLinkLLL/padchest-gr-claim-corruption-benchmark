#!/usr/bin/env python3
"""Implementation of the frozen PadChest-GR Benchmark v2 manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark_common import DEFAULT_CONFIG, load_config, resolve_path, sha256_file, write_csv, write_json

VERSION = "v2"
UNCERTAIN = re.compile(r"\b(possible|possibly|probable|probably|likely|suggestive|suspected|cannot exclude|could be)\b", re.I)
COMPARISON = re.compile(r"\b(improve(?:d|ment)?|worsen(?:ed|ing)?|unchanged|increase|decrease|persist(?:s|ent|ing)?|previous(?:ly)?|prior|compared|evolution|old|residual)\b", re.I)
COMPOUND = re.compile(r"\b(and|with|associated with|as well as)\b|;", re.I)
BILATERAL = re.compile(r"\b(bilateral|both|biapical|bibasal)\b", re.I)
DIFFUSE = re.compile(r"\b(diffuse|diffusely|extensive|generalized)\b", re.I)
NEGATION = re.compile(r"\b(no|without|absence of|free of|not seen|not observed|not appreciated|no evidence of|no signs? of)\b", re.I)

LABEL_NORMALIZATION = {
    "apical pleural thickening": "pleural thickening",
    "atelectasis basal": "atelectasis",
}
UPPER_LOCATIONS = {"apical", "right upper lobe", "left upper lobe", "upper lobe", "upper lung field"}
LOWER_LOCATIONS = {"basal", "right lower lobe", "left lower lobe", "lower lobe", "lower lung field"}
VERTICAL_PATTERNS = [
    (re.compile(r"\bupper lung field\b", re.I), "lower lung field", "upper", "lower"),
    (re.compile(r"\blower lung field\b", re.I), "upper lung field", "lower", "upper"),
    (re.compile(r"\bupper lobe\b", re.I), "lower lobe", "upper", "lower"),
    (re.compile(r"\blower lobe\b", re.I), "upper lobe", "lower", "upper"),
    (re.compile(r"\bapical\b", re.I), "basal", "upper", "lower"),
    (re.compile(r"\bbasal\b", re.I), "apical", "lower", "upper"),
    (re.compile(r"\bapex\b", re.I), "base", "upper", "lower"),
    (re.compile(r"\bbase\b", re.I), "apex", "lower", "upper"),
]
POLARITY_QUOTAS = {"infiltrates": 24, "consolidation": 4, "pleural effusion": 11, "nodule": 7, "pneumothorax": 3}
POLARITY_PHRASES = {
    "infiltrates": "pulmonary infiltrates",
    "consolidation": "pulmonary consolidation",
    "pleural effusion": "pleural effusion",
    "nodule": "pulmonary nodule",
    "pneumothorax": "pneumothorax",
}
NEGATED_FINDINGS = [
    ("pleural effusion", re.compile(r"\bpleural effusions?\b", re.I)),
    ("pneumothorax", re.compile(r"\bpneumothorax\b", re.I)),
    ("consolidation", re.compile(r"\bconsolidations?\b", re.I)),
    ("infiltrates", re.compile(r"\binfiltrates?\b", re.I)),
    ("nodule", re.compile(r"\b(?:pulmonary\s+)?nodules?\b", re.I)),
]
REPLACEMENTS = [
    "cardiomegaly", "aortic elongation", "scoliosis", "pleural effusion",
    "costophrenic angle blunting", "nodule", "pneumothorax", "infiltrates",
    "interstitial pattern", "pacemaker", "atelectasis", "calcified granuloma",
]
FIELDS = [
    "benchmark_version", "pair_id", "variant", "benchmark_group", "corruption_type",
    "corruption_direction", "patient_id", "study_id", "image_id", "image_relpath",
    "source_claim", "corrupted_claim", "tested_claim", "finding_label", "finding_labels",
    "source_finding_train_status", "corrupted_finding_train_status", "structured_locations",
    "source_gt_boxes", "changed_from", "changed_to", "claim_status",
    "phrase_groundability", "construction_rule",
]


def values(finding: dict[str, Any], key: str) -> list[Any]:
    value = finding.get(key)
    return value if isinstance(value, list) else []


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def canonical(value: str) -> str:
    value = normalize(value)
    return LABEL_NORMALIZATION.get(value, value)


def status(label: str, train_vocab: set[str]) -> str:
    return "train_seen" if canonical(label) in train_vocab else "train_unseen"


def stable_id(kind: str, study: str, source: str, corrupt: str, direction: str) -> str:
    payload = "\0".join((VERSION, kind, study, source, corrupt, direction))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{VERSION}_{kind}_{study}_{digest}"


def replace_once(text: str, pattern: re.Pattern[str], replacement: str) -> str:
    def repl(match: re.Match[str]) -> str:
        if match.group(0).isupper():
            return replacement.upper()
        if match.group(0)[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement
    return pattern.sub(repl, text, count=1)


def clean_positive(finding: dict[str, Any]) -> bool:
    sentence = str(finding.get("sentence_en") or "").strip()
    return bool(
        finding.get("abnormal") and sentence and len(values(finding, "labels")) == 1
        and values(finding, "boxes") and not UNCERTAIN.search(sentence)
        and not COMPARISON.search(sentence) and not COMPOUND.search(sentence)
        and not DIFFUSE.search(sentence) and not NEGATION.search(sentence)
    )


def polarity_positive(finding: dict[str, Any], label: str) -> bool:
    sentence = str(finding.get("sentence_en") or "").strip()
    return bool(
        label in POLARITY_QUOTAS and finding.get("abnormal") and sentence
        and len(values(finding, "labels")) == 1 and values(finding, "boxes")
        and not UNCERTAIN.search(sentence) and not NEGATION.search(sentence)
    )


def explicitly_negated(sentence: str, pattern: re.Pattern[str]) -> bool:
    finding = pattern.search(sentence)
    if finding is None:
        return False
    cues = list(NEGATION.finditer(sentence[: finding.start()]))
    if not cues:
        return False
    between = sentence[cues[-1].end() : finding.start()]
    return not re.search(r"[.;:]", between) and len(re.findall(r"\b\w+\b", between)) <= 6


def structured_side(locations: list[str]) -> str | None:
    text = " ".join(locations).lower()
    left, right = bool(re.search(r"\bleft\b", text)), bool(re.search(r"\bright\b", text))
    return None if left == right else ("left" if left else "right")


def structured_vertical(locations: list[str]) -> str | None:
    locations_set = {normalize(item) for item in locations}
    upper, lower = bool(locations_set & UPPER_LOCATIONS), bool(locations_set & LOWER_LOCATIONS)
    return None if upper == lower else ("upper" if upper else "lower")


def boxes_match_side(boxes: list[list[float]], side: str, boundary: float) -> bool:
    return bool(boxes) and (all(float(b[0]) > boundary for b in boxes) if side == "left" else all(float(b[2]) < boundary for b in boxes))


def boxes_match_vertical(boxes: list[list[float]], zone: str, boundary: float) -> bool:
    return bool(boxes) and (all(float(b[3]) < boundary for b in boxes) if zone == "upper" else all(float(b[1]) > boundary for b in boxes))


def context_has(findings: list[dict[str, Any]], source: dict[str, Any], label: str, attribute: str, kind: str) -> bool:
    for other in findings:
        if other is source:
            continue
        labels = {canonical(str(item)) for item in values(other, "labels")}
        if label not in labels or not values(other, "boxes"):
            continue
        locations = [str(item) for item in values(other, "locations")]
        sentence = str(other.get("sentence_en") or "")
        if kind == "side" and (structured_side(locations) == attribute or re.search(rf"\b{attribute}\b", sentence, re.I)):
            return True
        if kind == "vertical" and structured_vertical(locations) == attribute:
            return True
    return False


def add_pair(records: list[dict[str, Any]], **item: Any) -> None:
    common = {
        "benchmark_version": VERSION,
        "pair_id": item["pair_id"], "benchmark_group": item["group"],
        "corruption_type": item["kind"], "corruption_direction": item["direction"],
        "patient_id": item["patient"], "study_id": item["study"],
        "image_id": item["image"], "image_relpath": item["image"],
        "source_claim": item["source"], "corrupted_claim": item["corrupt"],
        "finding_label": item["label"],
        "finding_labels": json.dumps(item["labels"], ensure_ascii=False),
        "source_finding_train_status": item["source_train"],
        "corrupted_finding_train_status": item["corrupt_train"],
        "structured_locations": json.dumps(item["locations"], ensure_ascii=False),
        "source_gt_boxes": json.dumps(item["boxes"], ensure_ascii=False),
        "changed_from": item["changed_from"], "changed_to": item["changed_to"],
        "construction_rule": item["rule"],
    }
    records.append(common | {
        "variant": "source", "tested_claim": item["source"], "claim_status": "supported",
        "phrase_groundability": item["source_groundability"],
    })
    records.append(common | {
        "variant": "corrupted", "tested_claim": item["corrupt"],
        "claim_status": item["corrupt_status"],
        "phrase_groundability": item["corrupt_groundability"],
    })


def select_quota(candidates: dict[str, list[dict[str, Any]]], seed: int, direction: str) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for category, required in POLARITY_QUOTAS.items():
        unique: dict[str, dict[str, Any]] = {}
        for item in sorted(candidates.get(category, []), key=lambda x: (x["study"], x["image"], x["sentence"])):
            unique.setdefault(item["study"], item)
        pool = list(unique.values())
        category_seed = int(hashlib.sha256(f"{seed}:{direction}:{category}".encode()).hexdigest()[:16], 16)
        random.Random(category_seed).shuffle(pool)
        if len(pool) < required:
            raise ValueError(f"Insufficient {direction} candidates for {category}: {len(pool)} < {required}")
        selected[category] = pool[:required]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overrides", type=Path, required=True,
                        help="Private study-specific benchmark overrides; not distributed")
    args = parser.parse_args()
    overrides = json.loads(args.overrides.read_text(encoding="utf-8"))
    spatial_exclusions = overrides["spatial_exclusions"]
    laterality_rewrites = overrides["laterality_rewrites"]
    vertical_rewrites = overrides["vertical_rewrites"]
    config = load_config(args.config.resolve())
    reports_path, metadata_path = resolve_path(config, "reports"), resolve_path(config, "metadata")
    image_root = resolve_path(config, "image_root")
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    metadata: dict[str, dict[str, str]] = {}
    for row in metadata_rows:
        study = str(row.get("StudyID") or "")
        if not study:
            continue
        if study in metadata and (metadata[study].get("split"), metadata[study].get("PatientID")) != (row.get("split"), row.get("PatientID")):
            raise ValueError(f"Inconsistent metadata for StudyID {study}")
        metadata[study] = row
    with reports_path.open("r", encoding="utf-8-sig") as handle:
        reports = json.load(handle)

    train_vocab = {
        canonical(str(label))
        for study in reports
        if metadata.get(str(study.get("StudyID") or ""), {}).get("split") == "train"
        for finding in study.get("findings") or []
        for label in values(finding, "labels")
    }
    records: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    directions: Counter[str] = Counter()
    substitutions: list[dict[str, Any]] = []
    negative: dict[str, list[dict[str, Any]]] = defaultdict(list)
    positive: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for study in reports:
        study_id = str(study.get("StudyID") or "")
        meta = metadata.get(study_id)
        if not meta or meta.get("split") != str(config["split"]):
            continue
        patient, image = str(meta.get("PatientID") or ""), str(study.get("ImageID") or meta.get("ImageID") or "")
        if not image:
            exclusions["missing_image_id"] += 1
            continue
        findings = study.get("findings") or []
        positive_labels = {canonical(str(label)) for finding in findings if finding.get("abnormal") for label in values(finding, "labels")}
        positive_text = " ".join(str(f.get("sentence_en") or "").lower() for f in findings if f.get("abnormal"))

        for finding in findings:
            raw = str(finding.get("sentence_en") or "").strip()
            labels = [canonical(str(item)) for item in values(finding, "labels")]
            locations = [normalize(str(item)) for item in values(finding, "locations")]
            boxes = values(finding, "boxes")
            if len(labels) == 1 and polarity_positive(finding, labels[0]):
                positive[labels[0]].append({"patient": patient, "study": study_id, "image": image, "sentence": raw, "locations": locations, "boxes": boxes})

            if clean_positive(finding):
                label, sentence = labels[0], laterality_rewrites.get(study_id, raw)
                label_status = status(label, train_vocab)
                has_left, has_right = bool(re.search(r"\bleft\b", sentence, re.I)), bool(re.search(r"\bright\b", sentence, re.I))
                if has_left ^ has_right and not BILATERAL.search(sentence):
                    source_side = "left" if has_left else "right"
                    target_side = "right" if source_side == "left" else "left"
                    pattern = re.compile(rf"\b{source_side}\b", re.I)
                    if study_id in spatial_exclusions["laterality"]:
                        exclusions["laterality_explicit_v2_exclusion"] += 1
                    elif len(pattern.findall(sentence)) != 1:
                        exclusions["laterality_non_atomic_location"] += 1
                    elif structured_side(locations) != source_side:
                        exclusions["laterality_structured_location_mismatch"] += 1
                    elif not boxes_match_side(boxes, source_side, float(config["side_boundary"])):
                        exclusions["laterality_bbox_mismatch"] += 1
                    elif context_has(findings, finding, label, target_side, "side"):
                        exclusions["laterality_opposite_context"] += 1
                    else:
                        corrupt, direction = replace_once(sentence, pattern, target_side), f"{source_side}_to_{target_side}"
                        add_pair(records, pair_id=stable_id("laterality", study_id, sentence, corrupt, direction), group="main_benchmark", kind="laterality", direction=direction, patient=patient, study=study_id, image=image, source=sentence, corrupt=corrupt, label=label, labels=labels, source_train=label_status, corrupt_train=label_status, locations=locations, boxes=boxes, changed_from=source_side, changed_to=target_side, source_groundability="localizable_positive", corrupt_groundability="localizable_positive", corrupt_status="unsupported", rule="single finding; textual, structured, and bbox laterality agree; opposite side lacks same finding")
                        counts["laterality"] += 1; directions[f"laterality:{direction}"] += 1

                sentence = vertical_rewrites.get(study_id, raw)
                matches = [item for item in VERTICAL_PATTERNS if item[0].search(sentence)]
                if len(matches) == 1 and not BILATERAL.search(sentence):
                    pattern, replacement, source_zone, target_zone = matches[0]
                    if study_id in spatial_exclusions["vertical"]:
                        exclusions["vertical_explicit_v2_exclusion"] += 1
                    elif len(pattern.findall(sentence)) != 1:
                        exclusions["vertical_non_atomic_location"] += 1
                    elif structured_vertical(locations) != source_zone:
                        exclusions["vertical_structured_location_mismatch"] += 1
                    elif not boxes_match_vertical(boxes, source_zone, float(config["vertical_boundary"])):
                        exclusions["vertical_bbox_mismatch"] += 1
                    elif context_has(findings, finding, label, target_zone, "vertical"):
                        exclusions["vertical_opposite_context"] += 1
                    else:
                        corrupt, changed_from = replace_once(sentence, pattern, replacement), pattern.search(sentence).group(0).lower()  # type: ignore[union-attr]
                        direction = f"{source_zone}_to_{target_zone}"
                        add_pair(records, pair_id=stable_id("vertical", study_id, sentence, corrupt, direction), group="main_benchmark", kind="vertical", direction=direction, patient=patient, study=study_id, image=image, source=sentence, corrupt=corrupt, label=label, labels=labels, source_train=label_status, corrupt_train=label_status, locations=locations, boxes=boxes, changed_from=changed_from, changed_to=replacement, source_groundability="localizable_positive", corrupt_groundability="localizable_positive", corrupt_status="unsupported", rule="single finding; textual, structured, and bbox vertical zone agree; opposite zone lacks same finding")
                        counts["vertical"] += 1; directions[f"vertical:{direction}"] += 1

                substitutions.append({"patient": patient, "study": study_id, "image": image, "sentence": raw, "label": label, "labels": labels, "locations": locations, "boxes": boxes, "positive_labels": positive_labels, "positive_text": positive_text})

            if not boxes and NEGATION.search(raw) and not UNCERTAIN.search(raw):
                lexical = [(name, pattern) for name, pattern in NEGATED_FINDINGS if pattern.search(raw)]
                scoped = [(name, pattern) for name, pattern in lexical if explicitly_negated(raw, pattern)]
                if lexical and not scoped:
                    exclusions["polarity_negation_scope_mismatch"] += 1
                if len(scoped) == 1:
                    name, _ = scoped[0]
                    if name in positive_labels:
                        exclusions["polarity_positive_context"] += 1
                    else:
                        negative[name].append({"patient": patient, "study": study_id, "image": image, "sentence": raw, "locations": [], "boxes": []})

    seed = int(config["seed"])
    for direction, selected in (("negative_to_positive", select_quota(negative, seed, "negative_to_positive")), ("positive_to_negative", select_quota(positive, seed, "positive_to_negative"))):
        for category in POLARITY_QUOTAS:
            positive_claim = f"Evidence of {POLARITY_PHRASES[category]}."
            negative_claim = f"No evidence of {POLARITY_PHRASES[category]}."
            for item in selected[category]:
                source = negative_claim if direction == "negative_to_positive" else positive_claim
                corrupt = positive_claim if direction == "negative_to_positive" else negative_claim
                source_boxes = [] if direction == "negative_to_positive" else item["boxes"]
                locations = [] if direction == "negative_to_positive" else item["locations"]
                label_status = status(category, train_vocab)
                add_pair(records, pair_id=stable_id("polarity", item["study"], source, corrupt, direction), group="main_benchmark", kind="polarity", direction=direction, patient=item["patient"], study=item["study"], image=item["image"], source=source, corrupt=corrupt, label=category, labels=[category], source_train=label_status, corrupt_train=label_status, locations=locations, boxes=source_boxes, changed_from="negative" if direction == "negative_to_positive" else "positive", changed_to="positive" if direction == "negative_to_positive" else "negative", source_groundability="nonlocalizable_negative" if direction == "negative_to_positive" else "localizable_positive", corrupt_groundability="localizable_positive" if direction == "negative_to_positive" else "nonlocalizable_negative", corrupt_status="unsupported", rule="atomic matched-template polarity reversal; source evidence direction determined by annotations")
                counts["polarity"] += 1; directions[f"polarity:{direction}"] += 1

    random.Random(seed).shuffle(substitutions)
    replacement_counts: Counter[str] = Counter(); used_studies: set[str] = set()
    for item in substitutions:
        if len(used_studies) >= int(config["max_finding_substitution_pairs"]):
            break
        if item["study"] in used_studies:
            continue
        choices = []
        for replacement in REPLACEMENTS:
            replacement = canonical(replacement)
            if replacement == item["label"] or replacement in item["positive_labels"]:
                continue
            if re.search(rf"\b{re.escape(replacement)}s?\b", item["positive_text"], re.I):
                continue
            choices.append(replacement)
        if not choices:
            exclusions["substitution_no_candidate"] += 1
            continue
        choices.sort(key=lambda choice: (replacement_counts[choice], choice))
        replacement = choices[0]; replacement_counts[replacement] += 1; used_studies.add(item["study"])
        source, corrupt = f"Evidence of {item['label']}.", f"Evidence of {replacement}."
        add_pair(records, pair_id=stable_id("finding_substitution_proxy", item["study"], source, corrupt, "finding_to_finding"), group="proxy_lower_confidence", kind="finding_substitution_proxy", direction="finding_to_finding", patient=item["patient"], study=item["study"], image=item["image"], source=source, corrupt=corrupt, label=item["label"], labels=item["labels"], source_train=status(item["label"], train_vocab), corrupt_train=status(replacement, train_vocab), locations=item["locations"], boxes=item["boxes"], changed_from=item["label"], changed_to=replacement, source_groundability="localizable_positive", corrupt_groundability="localizable_positive", corrupt_status="proxy_unsupported", rule="replacement absent from study positive labels/text; annotation-negative proxy assumption")
        counts["finding_substitution_proxy"] += 1; directions["finding_substitution_proxy:finding_to_finding"] += 1

    expected = {"laterality": 245, "vertical": 67, "polarity": 98, "finding_substitution_proxy": 100}
    if dict(counts) != expected:
        raise ValueError(
            f"Unexpected v2 pair counts: {dict(counts)} != {expected}; "
            f"exclusions={dict(exclusions)}; directions={dict(directions)}"
        )
    if directions["polarity:negative_to_positive"] != 49 or directions["polarity:positive_to_negative"] != 49:
        raise ValueError(f"Unexpected polarity direction counts: {dict(directions)}")
    records.sort(key=lambda row: (row["corruption_type"], row["pair_id"], 0 if row["variant"] == "source" else 1))
    manifest = resolve_path(config, "manifest"); write_csv(manifest, records, FIELDS)
    manifest_hash = sha256_file(manifest); hash_path = resolve_path(config, "manifest_hash")
    hash_path.parent.mkdir(parents=True, exist_ok=True); hash_path.write_text(manifest_hash + "\n", encoding="utf-8")
    missing_images = sorted({row["image_id"] for row in records if not (image_root / row["image_relpath"]).is_file()})
    multi_box_pairs = len({row["pair_id"] for row in records if row["variant"] == "source" and len(json.loads(row["source_gt_boxes"])) > 1})
    summary = {
        "benchmark_name": config["benchmark_name"], "benchmark_version": VERSION,
        "split": config["split"], "seed": seed, "source_reports": str(reports_path),
        "source_reports_sha256": sha256_file(reports_path), "source_metadata": str(metadata_path),
        "source_metadata_sha256": sha256_file(metadata_path), "manifest": str(manifest),
        "manifest_sha256": manifest_hash, "pair_counts": dict(sorted(counts.items())),
        "direction_pair_counts": dict(sorted(directions.items())),
        "benchmark_group_pair_counts": {key: value // 2 for key, value in sorted(Counter(row["benchmark_group"] for row in records).items())},
        "pair_total": sum(counts.values()), "row_total": len(records),
        "unique_patients": len({row["patient_id"] for row in records}),
        "unique_studies": len({row["study_id"] for row in records}),
        "unique_images": len({row["image_id"] for row in records}),
        "multi_box_pairs": multi_box_pairs, "training_vocabulary_size": len(train_vocab),
        "missing_images": missing_images, "exclusion_counts": dict(sorted(exclusions.items())),
        "finding_substitution_replacement_counts": dict(sorted(replacement_counts.items())),
        "label_normalization": LABEL_NORMALIZATION,
        "note": "Laterality, vertical, and bidirectional polarity form the 410-pair main benchmark. Finding substitution is a 100-pair core proxy/lower-confidence subset and is not pooled into one overall score.",
    }
    write_json(resolve_path(config, "construction_summary"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
