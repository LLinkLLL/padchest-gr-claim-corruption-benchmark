"""Paired score, text-only reference and grounding-score joint statistics."""
from __future__ import annotations

import math
from collections import defaultdict, Counter
from statistics import mean, median

from analyse_track2 import auc, stable_seed
from benchmark_common import clustered_bootstrap_interval, format_float, proportion, truthy
from biovilt_track1_metrics import cluster_mean_ci

OUTCOMES = ("source_evidence_adherence", "corrupted_text_following", "abstention", "other_mixed", "invalid")
RESPONSES = ("source_higher", "corrupted_higher", "tie", "invalid_score")
DIRECTIONS = ("negative_to_positive", "positive_to_negative")


def response(source, corrupted, valid=True):
    if not valid or source is None or corrupted is None or not math.isfinite(source) or not math.isfinite(corrupted):
        return "invalid_score"
    return "source_higher" if source > corrupted else "corrupted_higher" if source < corrupted else "tie"


def primary_summary(samples, replicates=1000, base_seed=2026):
    """Recompute the unmodified shared Track 2 protocol, including its seeds."""
    grouped = defaultdict(list)
    for r in samples:
        key = r["model"], r["corruption_type"], r["score_type"]
        grouped[(*key, "overall")].append(r)
        grouped[(*key, r["corruption_direction"])].append(r)
    output = []
    for (model, kind, score_type, direction), rows in sorted(grouped.items()):
        valid = [r for r in rows if r["valid_score_pair"]]
        valid_stat = lambda rs: proportion(rs, lambda r: r["valid_score_pair"])
        low, high = clustered_bootstrap_interval(rows, "patient_id", valid_stat, replicates,
                                                stable_seed(base_seed, model, kind, score_type, direction, "valid"))
        result = dict(model=model, corruption_type=kind, score_type=score_type, corruption_direction=direction,
                      n_total=len(rows), n_valid_pairs=len(valid), valid_pair_rate=format_float(valid_stat(rows)),
                      valid_pair_ci_low=format_float(low), valid_pair_ci_high=format_float(high))
        if valid:
            statistics = {
                "pairwise": lambda rs: proportion(rs, lambda r: r["source_higher"]),
                "delta": lambda rs: mean(float(r["source_minus_corrupted"]) for r in rs),
                "auc": auc,
            }
            cis = {name: clustered_bootstrap_interval(valid, "patient_id", stat, replicates,
                   stable_seed(base_seed, model, kind, score_type, direction, name)) for name, stat in statistics.items()}
            result.update(pairwise_accuracy=format_float(statistics["pairwise"](valid)),
                          pairwise_accuracy_ci_low=format_float(cis["pairwise"][0]), pairwise_accuracy_ci_high=format_float(cis["pairwise"][1]),
                          mean_score_delta=format_float(statistics["delta"](valid)),
                          mean_score_delta_ci_low=format_float(cis["delta"][0]), mean_score_delta_ci_high=format_float(cis["delta"][1]),
                          median_score_delta=format_float(median(r["source_minus_corrupted"] for r in valid)),
                          tie_rate=format_float(proportion(valid, lambda r: r["tie"])),
                          corrupted_higher_rate=format_float(proportion(valid, lambda r: r["corrupted_higher"])),
                          auroc=format_float(auc(valid)), auroc_ci_low=format_float(cis["auc"][0]), auroc_ci_high=format_float(cis["auc"][1]))
        output.append(result)
    return output


def polarity_baselines(manifest_pairs):
    samples = []
    categories = {d: Counter() for d in DIRECTIONS}
    for pair in manifest_pairs.values():
        source, corrupted = pair["source"], pair["corrupted"]
        if source["corruption_type"] != "polarity":
            continue
        direction = source["corruption_direction"]
        categories[direction][source["finding_label"]] += 1
        flags = []
        findings = []
        for m in (source, corrupted):
            claim = m["tested_claim"].lower()
            neg = claim.startswith("no evidence of ")
            assert neg or claim.startswith("evidence of "), "Non-atomic polarity template"
            flags.append(neg)
            findings.append(claim[len("no evidence of ") if neg else len("evidence of "):])
        assert flags[0] != flags[1] and findings[0] == findings[1]
        assert flags[0] == (direction == "negative_to_positive")
        for rule in ("affirmative_preference", "negative_preference"):
            ss = int(not flags[0]) if rule == "affirmative_preference" else int(flags[0])
            cs = 1 - ss
            samples.append(dict(model=rule, pair_id=source["pair_id"], patient_id=source["patient_id"],
                                study_id=source["study_id"], corruption_type="polarity", corruption_direction=direction,
                                finding_label=source["finding_label"], source_score=ss, corrupted_score=cs,
                                valid_score_pair=True, source_higher=ss > cs, corrupted_higher=cs > ss, tie=False,
                                score_type="text_only_rule_reference", source_minus_corrupted=ss - cs))
    assert sum(categories[DIRECTIONS[0]].values()) == sum(categories[DIRECTIONS[1]].values()) == 49
    assert categories[DIRECTIONS[0]] == categories[DIRECTIONS[1]], "Polarity categories no longer matched"
    return samples


def macro_stat(rows):
    rates = []
    for direction in DIRECTIONS:
        valid = [r for r in rows if r["corruption_direction"] == direction and r["valid_score_pair"]]
        if not valid:
            return float("nan")
        rates.append(mean(float(r["source_higher"]) for r in valid))
    return mean(rates)


def polarity_macro_summary(samples, replicates=1000, base_seed=2026):
    groups = defaultdict(list)
    for r in samples:
        if r["corruption_type"] == "polarity":
            groups[r["model"]].append(r)
    output = []
    for model, rows in sorted(groups.items()):
        seed = stable_seed(base_seed, model, "polarity", "two_direction_macro")
        lo, hi = clustered_bootstrap_interval(rows, "patient_id", macro_stat, replicates, seed)
        record = dict(model=model, score_type=rows[0]["score_type"], n_total=len(rows),
                      macro_source_higher=macro_stat(rows), macro_ci_low=lo, macro_ci_high=hi,
                      ci_definition="mean of two directional rates recalculated within each patient bootstrap replicate")
        for d in DIRECTIONS:
            group = [r for r in rows if r["corruption_direction"] == d]
            stat = cluster_mean_ci(group, lambda r: float(r["source_higher"]) if r["valid_score_pair"] else float("nan"),
                                   stable_seed(base_seed, model, "polarity", rows[0]["score_type"], d, "pairwise"), replicates)
            record[d + "_n"] = len(group)
            record[d + "_rate"] = stat["mean"]
            record[d + "_ci_low"], record[d + "_ci_high"] = stat["ci_low"], stat["ci_high"]
        output.append(record)
    return output


def ratio_ci(rows, numerator, denominator, seed, replicates=1000):
    def getter(r):
        return float(bool(numerator(r))) if denominator(r) else float("nan")
    # A constant statistic over the entire sampling frame has a degenerate
    # empirical CI. Avoid random draws only when every row is in the denominator.
    if rows and all(denominator(r) for r in rows):
        values = [getter(r) for r in rows]
        if min(values) == max(values):
            return dict(mean=values[0], ci_low=values[0], ci_high=values[0], n_valid_pairs=len(rows),
                        n_valid_patients=len({r["patient_id"] for r in rows}), valid_replicates=replicates,
                        median=values[0])
    if not any(denominator(r) for r in rows):
        return cluster_mean_ci([], getter, seed, replicates)
    return cluster_mean_ci(rows, getter, seed, replicates)


def joint_summary(samples, replicates=1000, base_seed=2026):
    groups = defaultdict(list)
    for r in samples:
        prefix = r["model"], r["corruption_type"]
        groups[(*prefix, "overall")].append(r)
        groups[(*prefix, r["corruption_direction"])].append(r)
    output = []
    for (model, kind, direction), all_rows in sorted(groups.items()):
        for population, rows in (("all_eligible", all_rows), ("model_original_success", [r for r in all_rows if r["source_success"]])):
            for outcome in OUTCOMES:
                in_outcome = lambda r, o=outcome: r["behavioural_outcome"] == o
                count_outcome = sum(in_outcome(r) for r in rows)
                valid_in_outcome = sum(in_outcome(r) and r["score_response"] != "invalid_score" for r in rows)
                for resp in RESPONSES:
                    hit = lambda r, t=resp: in_outcome(r) and r["score_response"] == t
                    seed_parts = [model, kind, direction, population, outcome, resp]
                    joint = ratio_ci(rows, hit, lambda r: True, stable_seed(base_seed, *seed_parts, "joint"), replicates)
                    conditional = ratio_ci(rows, hit, in_outcome, stable_seed(base_seed, *seed_parts, "within_outcome"), replicates)
                    record = dict(model=model, corruption_type=kind, corruption_direction=direction, population=population,
                                  score_type=all_rows[0]["score_type"], n_population=len(rows),
                                  population_patients=len({r["patient_id"] for r in rows}), behavioural_outcome=outcome,
                                  n_outcome=count_outcome, n_valid_score_pairs_in_outcome=valid_in_outcome,
                                  score_response=resp, count=sum(hit(r) for r in rows),
                                  joint_population_rate=joint["mean"], joint_ci_low=joint["ci_low"], joint_ci_high=joint["ci_high"],
                                  within_outcome_rate=conditional["mean"], within_ci_low=conditional["ci_low"], within_ci_high=conditional["ci_high"],
                                  within_valid_replicates=conditional["valid_replicates"])
                    output.append(record)
    return output
