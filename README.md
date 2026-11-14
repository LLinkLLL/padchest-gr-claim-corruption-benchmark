# PadChest-GR Claim Corruption Benchmark

Research code and numerical evidence for *Evaluating Medical Phrase Grounding under Incorrect Claims: A Controlled Benchmark and PG-FC-Inspired Reimplementation*. This benchmark asks whether chest X-ray grounding outputs and claim-support scores respond consistently when only the claim changes and the image stays fixed.

## Benchmark v2

| Controlled change | Pairs | Direction counts |
| --- | ---: | --- |
| Laterality | 245 | 150 right to left, 95 left to right |
| Strict vertical location | 67 | 31 upper to lower, 36 lower to upper |
| Polarity | 98 | 49 absent to present, 49 present to absent |
| Finding substitution (annotation-negative proxy) | 100 | Reported separately |

The test set contains 510 pairs, 1,020 claim inputs, 301 studies and 298 patients. The first three subsets form the main benchmark. Unannotated substitutions are not guaranteed to be clinically absent; there is no pooled headline score.

## Main results

| Model | Laterality following (245 pairs) | Vertical following (67 pairs) | Laterality following after a correct source hit |
| --- | ---: | ---: | ---: |
| MedGrounder | 93.47% | 98.51% | 149/158 |
| MedRPG | 100.00% | 80.60% | 78/78 |
| MAIRA-2 | 97.14% | 92.54% | 171/173 |
| PG-FC-inspired (epoch 22) | 100.00% | 97.01% | 111/111 |
| BioViL-T (heatmap peak) | 54.69% | 55.22% | 58/94 |

- **Correct source localisation did not ensure robustness.** Four box methods frequently followed the corrupted location even on their own originally successful pairs. Following describes the final location; it does not demonstrate that a model ignores the image or accepts the claim.
- **Localisation and ranking can disagree.** On 101/134 BioViL-T laterality pairs whose peak followed the corrupted location, the corrupted claim nevertheless received a lower global score. On 110 of the 245 laterality pairs, PG-FC-inspired followed the changed side while its native E output rejected the corrupted claim at the fixed 0.5 threshold.
- **A missing box does not establish claim validity.** PG-FC-inspired emitted no native empty boxes. Development-calibrated postprocessing decoded all 49 supported and all 49 unsupported negative polarity claims as empty. E still accepted 38/49 unsupported negatives. This supplementary decoder changes neither E scores nor the spatial results.

Polarity score preference was directional: BioViL-T ranked the supported claim higher on 38/49 absent-to-present and 39/49 present-to-absent pairs; PG-FC-inspired did so in 40/49 and 17/49; MedGrounder in 0/49 and 49/49. These are within-image ranking rates, not binary classification accuracy. See [results and interpretation](docs/results.md) for the full table index and limitations.

## Verify the evidence

Python 3.10+ is required. No dataset account, GPU or model download is necessary for the public analysis check.

```sh
python -m pip install -r requirements.txt
python scripts/verify_evidence.py
python -m unittest discover -s scripts -p "test_*.py"
```

The checker recomputes principal results using [derived analysis inputs](evidence/) and compares them with the [frozen summary tables](results/). The published tables retain the original patient-clustered confidence intervals. The [reproduction guide](docs/reproduction.md) distinguishes public numerical verification from scoring saved predictions, rebuilding the exact private manifest, and rerunning inference.

## Data access and scope

Obtain PadChest-GR through the [official project](https://bimcv.cipf.es/bimcv-projects/padchest-gr/) and follow the applicable terms. This repository contains no chest X-rays, radiology report text, ground-truth boxes, original dataset identifiers, model weights, or raw inference outputs. The [frozen manifest receipt](provenance/benchmark_v2.json) records the private benchmark hash and sample counts without redistributing the manifest. Opaque release identifiers exist only to join analysis rows and retain within-patient clustering.

- [Protocol and model settings](docs/protocol.md)
- [Results and table index](docs/results.md)
- [Reproduction and access boundaries](docs/reproduction.md)
- [PG-FC-inspired implementation repository](https://github.com/LLinkLLL/pgfc_reimplementation)
- [Citation metadata](CITATION.cff)

Research use only. This repository does not establish clinical safety or suitability for diagnosis. Third-party datasets, checkpoints and software retain their respective terms.
