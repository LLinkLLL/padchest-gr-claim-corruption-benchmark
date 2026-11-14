"""Verify public benchmark point estimates from coordinate-free evidence."""
from collections import Counter, defaultdict
from hashlib import sha256
from math import isclose
from pathlib import Path
from statistics import mean

from benchmark_common import read_csv, truthy
from track2_final_metrics import macro_stat, primary_summary, response

ROOT = Path(__file__).resolve().parents[1]
GROUP = ("model", "corruption_type", "corruption_direction")
FOLLOW = "corrupted_text_following"
PG = "pgfc_inspired_padchest"
BIO = "biovilt_official"


def rows(folder, name):
    return read_csv(ROOT / folder / f"{name}.csv")


def key(row, fields=GROUP):
    return tuple(row[f] for f in fields)


def same(observed, published, label):
    if not isclose(float(observed), float(published), rel_tol=0, abs_tol=1.1e-6):
        raise ValueError(f"{label}: calculated {observed}, published {published}")


def choose(items, target, fields=GROUP):
    return [r for r in items if all(
        r[f] == target[f] or (f == "corruption_direction" and target[f] == "overall")
        for f in fields
    )]


def populate(items, population, field="source_success"):
    return items if population == "all_eligible" else [r for r in items if truthy(r[field])]


def verify_hashes():
    records = (ROOT / "provenance/public_files.sha256").read_text().splitlines()
    for line in records:
        digest, name = line.split("  ", 1)
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT) or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Public evidence checksum mismatch: {name}")
    return len(records)


def main():
    files = verify_hashes()
    spatial = rows("evidence", "spatial_outcomes")
    paired = rows("evidence", "paired_scores")
    native = rows("evidence", "nonspatial_outcomes")
    bio_n = rows("evidence", "biovilt_nonspatial_changes")
    bio_s = rows("evidence", "biovilt_spatial_changes")
    zero = rows("evidence", "pgfc_zero_box_outcomes")
    assert len(spatial) == 1560 and len(zero) == 1020
    assert len({r["pair_key"] for r in zero}) == 510
    assert len({r["cluster_key"] for r in zero}) == 298
    assert all(len({r["cluster_key"] for r in zero if r["pair_key"] == p}) == 1
               for p in {r["pair_key"] for r in zero})
    assert len(paired) == 1726 and len(bio_s) == 312 and len(bio_n) == 198

    for target in rows("results", "track1_spatial"):
        group = populate(choose(spatial, target), target["population"])
        same(len(group), target["n"], "Spatial denominator")
        same(mean(truthy(r["source_success"]) for r in group), target["source_success_rate"], "Source hit")
        for outcome in ("source_evidence_adherence", FOLLOW, "other_mixed", "invalid", "abstention"):
            value = target[outcome + "_rate"]
            if value != "":
                same(mean(r["behavioural_outcome"] == outcome for r in group), value, outcome)

    historical = rows("results", "track2_scores")
    standard, rules = [], []
    for row in paired:
        src, bad = float(row["source_score"]), float(row["corrupted_score"])
        valid = truthy(row["valid_score_pair"])
        outcome = response(src, bad, valid)
        row.update(patient_id=row["cluster_key"], source_score=src, corrupted_score=bad,
                   source_minus_corrupted=src - bad, valid_score_pair=valid,
                   source_higher=(outcome == "source_higher"), tie=(outcome == "tie"),
                   corrupted_higher=(outcome == "corrupted_higher"), score_response=outcome)
        (rules if row["score_type"] == "text_only_rule_reference" else standard).append(row)
    keys = (*GROUP, "score_type")
    recomputed = {key(r, keys): r for r in primary_summary(standard, replicates=1)}
    assert len(recomputed) == len(historical)
    for target in historical:
        actual = recomputed[key(target, keys)]
        for f in ("n_total", "n_valid_pairs", "valid_pair_rate", "pairwise_accuracy", "tie_rate",
                  "corrupted_higher_rate", "mean_score_delta", "median_score_delta", "auroc"):
            same(actual[f], target[f], f)
    for target in rows("results", "track2_polarity_macro"):
        group = [r for r in standard + rules if r["model"] == target["model"] and r["corruption_type"] == "polarity"]
        same(macro_stat(group), target["macro_source_higher"], "Polarity macro")
        for direction in ("negative_to_positive", "positive_to_negative"):
            subset = [r for r in group if r["corruption_direction"] == direction]
            same(len(subset), target[direction + "_n"], "Polarity n")
            same(mean(r["source_higher"] for r in subset), target[direction + "_rate"], direction)

    lookup = {(r["model"], r["pair_key"]): r for r in spatial}
    joint = [{**r, **lookup[r["model"], r["pair_key"]]} for r in standard if r["corruption_type"] in ("laterality", "vertical")]
    assert len(joint) == 936
    for t in rows("results", "joint_spatial_score"):
        group = populate(choose(joint, t), t["population"])
        outcome = [r for r in group if r["behavioural_outcome"] == t["behavioural_outcome"]]
        hits = sum(r["score_response"] == t["score_response"] for r in outcome)
        same(len(group), t["n_population"], "Joint population")
        same(len(outcome), t["n_outcome"], "Joint outcome")
        same(hits, t["count"], "Joint count")
        if group:
            same(hits / len(group), t["joint_population_rate"], "Joint ratio")
        if outcome:
            same(hits / len(outcome), t["within_outcome_rate"], "Conditional ratio")

    for target in rows("results", "track1_nonspatial"):
        group = choose(native, target)
        same(len(group), target["n"], "Nonspatial n")
        for variant in ("source", "corrupted"):
            same(mean(int(r[variant + "_box_count"]) > 0 for r in group),
                 target[variant + "_box_rate"], "Nonspatial box rate")
        if target["source_region_hit_rate"]:
            same(mean(truthy(r["source_region_hit"]) for r in group), target["source_region_hit_rate"], "Source region hit")
            same(mean(truthy(r["corrupted_source_region_hit"]) for r in group),
                 target["corrupted_source_region_hit_rate"], "Corrupted region hit")
        if target["conditional_source_region_retention_rate"]:
            hits = [r for r in group if truthy(r["source_region_hit"])]
            same(mean(truthy(r["corrupted_source_region_hit"]) for r in hits),
                 target["conditional_source_region_retention_rate"], "Region retention")

    for name, evidence in (("biovilt_spatial_heatmap", bio_s), ("biovilt_nonspatial_heatmap", bio_n)):
        for t in rows("results", name):
            if t["metric"] not in evidence[0] or t["mean"] == "":
                continue
            group = choose(evidence, t)
            if "population" in t:
                group = populate(group, t["population"])
            values = [float(r[t["metric"]]) for r in group if r[t["metric"]] != ""]
            same(len(values), t["n_valid_pairs"], "Heatmap n")
            same(mean(values), t["mean"], t["metric"])

    for t in rows("results", "pgfc_fixed_e"):
        group = choose(standard, {**t, "model": PG})
        same(len(group), t["n"], "Fixed-E n")
        same(sum(r["source_score"] >= .5 for r in group), t["source_accept_n"], "Source accepted")
        same(sum(r["corrupted_score"] < .5 for r in group), t["corrupted_reject_n"], "Corrupted rejected")
        same(sum(r["source_score"] >= .5 and r["corrupted_score"] < .5 for r in group), t["both_decisions_correct_n"], "Both E decisions")
        if t["accepted_corrupted_text_following_n"]:
            same(sum(r["corrupted_score"] >= .5 and lookup[PG, r["pair_key"]]["behavioural_outcome"] == FOLLOW for r in group),
                 t["accepted_corrupted_text_following_n"], "Accepted following")
    for t in rows("results", "pgfc_zero_box"):
        group = choose(zero, t, ("variant", "corruption_type", "corruption_direction"))
        same(len(group), t["n"], "Zero-box n")
        for k in ("native_no_box", "derived_no_box"):
            same(sum(truthy(r[k]) for r in group), t[k + "_n"], k)
        if t["box_and_E_correct_n"]:
            both = sum(truthy(r["derived_no_box"]) == truthy(r["expected_no_box_under_training_policy"])
                       and truthy(r["E_accept"]) == (r["variant"] == "source") for r in group)
            same(both, t["box_and_E_correct_n"], "Both box and E")
    follow_bio = [r for r in joint if r["model"] == BIO and r["corruption_type"] == "laterality" and r["behavioural_outcome"] == FOLLOW]
    follow_pg = [r for r in joint if r["model"] == PG and r["corruption_type"] == "laterality" and r["behavioural_outcome"] == FOLLOW]
    assert (sum(r["source_higher"] for r in follow_bio), len(follow_bio)) == (101, 134)
    assert (sum(r["corrupted_score"] < .5 for r in follow_pg), len(follow_pg)) == (110, 245)
    assert sum(truthy(r["derived_no_box"]) for r in zero) == 98
    print(f"Verified {files} checksummed files and the principal published point estimates.")
    print("Original confidence intervals are archived, not regenerated for this check.")


if __name__ == "__main__":
    main()
