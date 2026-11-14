# Evaluation protocol

## Paired construction

Use the official PadChest-GR test split and English finding annotations. Pair each supported source claim with one controlled corruption on the same image. Construction uses seed 42. Spatial eligibility requires agreement between the sentence, structured location and every source box. Exclude ambiguous, comparative, diffuse, compound or negated spatial descriptions, and cases with an annotated occurrence of the same finding in the intended target region. Strict vertical scoring uses upper and lower image halves, not a complete anatomical lung-zone ontology.

Polarity uses atomic `Evidence of <finding>.` and `No evidence of <finding>.` templates. Match counts in both directions: infiltrates 24, consolidation 4, pleural effusion 11, nodule 7, pneumothorax 3. The supported negative source has no lesion box. Retain the 100 finding substitutions as a separate lower-confidence proxy using absence from positive annotations and text. Do not interpret non-annotation as adjudicated absence. Five historical study-level adjustments are applied through a private override file by the public builder; see [reproduction](reproduction.md).

## Models and interfaces

| Model | Spatial output | Score used for Track 2 | Run settings |
| --- | --- | --- | --- |
| MedGrounder | Box centres | Maximum raw query groundability (proxy) | Published MSPC fine-tuned weights; grounding threshold 0.8, box-fusion IoU 0.1 |
| MedRPG | Box centre | Not applicable | `MedMPG_MS_CXR.pth`; 640-pixel padded input; 20-token query limit; corrected 16-bit conversion |
| MAIRA-2 | Generated phrase boxes | Not applicable | `microsoft/maira-2` revision `795a2b1cd4a310624b4e3d14b5a23e41fd273deb`; one beam and at most 150 new tokens |
| BioViL-T | Heatmap peak restored to original image | Global image-text cosine (proxy) | `microsoft/BiomedVLP-BioViL-T`, v1.0; current image only; 512-pixel resize, 448-pixel centre crop |
| PG-FC-inspired | Box centre, no E gating | Native veracity E | Selected epoch 22; deterministic structured claims; CXR-RePaiR initialisation; full-image 224-pixel resize |

MedRPG, MAIRA-2 and BioViL-T 16-bit inputs use division by 256 before uint8 RGB conversion and native transforms. The final MedRPG run has no source/corrupted token collisions. PG-FC-inspired uses 0.5/99.5-percentile intensity mapping and CXR-RePaiR normalisation. PG-FC-inspired is a resource-constrained implementation, not the original authors' model. The checkpoint was a development polarity candidate, but final adoption followed inspection of benchmark results from multiple checkpoints.

## Track 1: spatial and nonspatial behaviour

For laterality and vertical, source success means a predicted centre hits any source box. On the corrupted claim, hitting the source without hitting the corrupted half counts as adherence; hitting the corrupted half without the source counts as following; hitting both or neither counts as other/mixed. Valid no-box and invalid are separate. Patient left maps to image x > 0.5, upper to y < 0.5. Conditional results use each model's own successful-source subset. They are not common-success estimates.

BioViL-T uses the centre of the peak grid cell (main analysis with Gaussian smoothing sigma 1.5) and no native abstention. Raw cosine scales, visible-area weighting and unsmoothed checks accompany the peak analysis. Report mean similarity in the source region, signed contrast-to-noise, target-minus-source half contrast and the mean of the highest 5% of cell responses. Changes are corrupted minus source. Do not interpret local attenuation as a calibrated decision. For polarity and substitution, report output presence and available source-region retention without imposing the spatial following taxonomy.

## Track 2: ranking, fixed E, and joint outcomes

Compute the fraction of valid pairs with score(source) strictly greater than score(corrupted), and report ties and coverage separately. Score deltas are source minus corrupted. AUROC pools scores across pairs, gives half credit for ties, and answers a different question. Native E and proxy scores are not calibrated probabilities or directly comparable in magnitude. Use the two polarity directions and their equal-weight macro average. The two wording reference rules always prefer affirmative or negative claims, respectively, without reading images.

Historical 95% percentile intervals use 1,000 patient-clustered resamples and base seed 2026. They reflect sampling variation, not training-seed uncertainty. The joint analysis keeps spatial classification, score ranking, and E threshold decisions distinct. E >= 0.5 accepts; E < 0.5 rejects. E never gates the native boxes.

## PG-FC-inspired supplementary zero-box decoding

After benchmark exposure, a separate decoder was introduced. It keeps native geometric empties and decodes a normalised, unclipped regression output as no-box when `max(x, y, w, h) <= 0.21630859375`. The cutoff maximised balanced zero-target recall and nonzero-target retention on existing predictions for 1,092 development claims from 220 patients, disjoint from the test patients. It recognised 357/378 zero targets and retained 714/714 nonzero targets on development. This is fitted calibration performance, not an independent test score. No wording, reference box, or E gate is used at decoding. The original checkpoint, native results and E scores are unchanged.

## Primary references

- [PadChest-GR, NEJM AI, DOI 10.1056/AIdbp2401120](https://doi.org/10.1056/AIdbp2401120)
- [MedGrounder: Generalised Medical Phrase Grounding](https://arxiv.org/abs/2512.01085)
- [MedRPG: Medical Phrase Grounding with Region-Phrase Context Contrastive Alignment](https://arxiv.org/abs/2303.07618)
- [MAIRA-2: Grounded Radiology Report Generation](https://arxiv.org/abs/2406.04449)
- [BioViL-T: Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing](https://arxiv.org/abs/2301.04558)
- [Original PG-FC paper](https://arxiv.org/abs/2509.21356)
