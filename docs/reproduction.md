# Reproduction guide

## Public verification

Run `python scripts/verify_evidence.py` from the repository root after installing `requirements.txt`. All inputs are English, local to the repository, and free of radiographs, report sentences, image paths, original identifiers and coordinates. `provenance/public_files.sha256` checks the numerical release payload. The evidence files contain scores or categorical/scalar metrics, not source dataset records or raw predictions.

The checker reproduces five-model spatial rates (including model-specific original-success subsets), Track 2 ranking, score deltas, AUROC and polarity macro point estimates, joint counts and percentages, box-presence and source retention rates, BioViL-T heatmap mean changes, PG-FC fixed-E counts and zero-box outcome counts. The original confidence intervals are carried over in the aggregate tables. Pair and patient identifiers are random aliases and the exact bootstrap draws are not promised to match across the public and private releases. Figure files are rendered summary statistics, not clinical images.

## Local scoring with authorised inputs

Obtain the dataset through the official access route. Place the inputs outside the Git working tree (or under ignored `data/`), then copy `configs/benchmark.json` to a local config and set its `paths` entries. Relative paths resolve from the repository root. `reports` and `metadata` point to the English groundings and corresponding split file; `image_root` points to the authorised PNG files. The frozen test set used `grounded_reports_20240819.json` and `master_table.csv` with the source hashes in [the manifest receipt](../provenance/benchmark_v2.json).

```sh
python scripts/build_manifest_v2.py --config /path/to/local-config.json --overrides /path/to/private-overrides.json
python scripts/validate_manifest_v2.py --config /path/to/local-config.json
python scripts/analyse_track1.py --config /path/to/local-config.json
python scripts/analyse_nonspatial.py --config /path/to/local-config.json
python scripts/analyse_track2.py --config /path/to/local-config.json
```

The historical study-specific overrides are deliberately excluded because they identify original studies. The private override JSON has `spatial_exclusions` (mapping `laterality` and `vertical` to arrays of study identifiers), `laterality_rewrites` and `vertical_rewrites` (mapping identifiers to replacement sentences). The historical override counts are 1 laterality exclusion, 2 vertical exclusions, 3 laterality rewrites and 1 vertical rewrite. Count checks may fail without the exact overrides. Possession of the public repository and a dataset account alone does not guarantee an exact rebuild. An existing authorised copy of the frozen manifest can be verified against the recorded SHA-256 instead.

Use per-model `normalized_predictions.csv` and `run_metadata.json` in the configured local results directory. Predictions must have exactly one row for each `(pair_id, variant)`. Required fields are `pair_id`, `variant`, `model`, `valid_output`, `boxes_xyxy_normalized`, `score`, `score_type` and `error`. Coordinates are in the normalised original-image frame. A valid no-box output is `[]`; use `supports_box_grounding=false` for heatmaps or score-only outputs. The recorded `inference_manifest_sha256` must equal the actual local manifest hash. Editing a run record cannot make predictions for a changed manifest valid.

The common scripts score native boxes and paired scores. BioViL-T's heatmap geometry and metrics are in `scripts/biovilt_track1_metrics.py`; learned inference and weight acquisition are upstream concerns. This repository is a numerical evidence release, not a verified end-to-end rerun of every model. Exact historical checkpoints, raw outputs, environment locks, cluster identities, and clinical inputs are not distributed here. The benchmark source should not be used for clinical decisions.
