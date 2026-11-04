# Phase 3 (stills stage) - report

Pre-registered in `phase3/PROTOCOL.md`; one look at the test blocks; raw output in `phase3/phase3_test_report.txt`, per-asana table `phase3/results_stills_per_asana.csv`.

## What was done
- Backbone (YOLO26m, imgsz 640) on all 8,259 usable stills of the 8 headline asanas (6.7 % no detection); x-correction K = 1.65; 10 angles with tau = 0.5 gating (`poseguard/angles.py`).
- Deterministic time-block split with guard bands: dev 5,234 / test 1,954 / guard 1,071 images; 187 dev blocks, 73 test blocks.
- Per-(asana, posture class, angle) percentile bands from **dev Right stills only**; any-angle-out-of-band => incorrect; < 3 judgeable angles => abstain (never a fault).
- Hyper-parameters chosen by 5-fold block-grouped CV on dev only. **Frozen: p = 15, delta = 5** (band = [P15, P85] widened by 5 deg). 170 of 270 possible (asana, class, angle) bands had enough dev data (`phase3/reference_ranges_stills.csv`; median band width 22 deg trunk tilt ... 69 deg left knee).
- **Amendment A1 (before any test result, documented in PROTOCOL sec. 7):** the first dev run selected a grid-corner setting and its F1 was below an always-'incorrect' predictor, so selection moved from F1 to balanced accuracy and the p grid was extended. First-run output kept in `phase3/phase3_dev_report_first_run.txt`.
- Tests: 11 library tests (hand-computed values; guard-band/abstain mutations verified to fail them); a smoke mode ran the whole test script with the real test rows removed before the real look.

## Result (test blocks, frozen configuration) - **negative**
| | value |
|---|---|
| headline test rows / scored / abstained | 1,876 / 1,165 / 37.9 % |
| **macro balanced accuracy** | **0.533, 95 % block-bootstrap CI [0.480, 0.593]** (chance = 0.500) |
| macro F1 (incorrect) | 0.613 [0.344, 0.736] |
| pooled precision / recall / specificity | 0.512 / 0.847 / **0.206** |
| always-'incorrect' baseline (base rate 0.495) | F1 0.662, precision 0.495 |

**Reading:** the CI on balanced accuracy includes chance and the F1 is below the trivial flag-everything predictor. The engine flags 85 % of incorrect images but also **79 % of correct ones**.
Per asana (n scored, balanced accuracy, precision, recall): Anantasana 71, 0.536, 0.385, 0.962 | Ardhakati Chakrasana 102, 0.468, 0.382, 0.707 | Kati Chakrasana 32, 0.575, 0.640, 0.842 | Marjariasana 296, 0.497, 0.461, 0.825 | Parvatasana 504, 0.514, 0.571, 0.891 | Tadasana 160, 0.606, 0.520, 0.757. **Bhujangasana and Vajrasana have no scored test rows** (no reference or every row abstained; Bhujangasana abstained on 233 rows). Per-asana CIs are wide (often 0-1) because there are only 3-17 independent test blocks per asana.
Angle coverage on test rows: computable 0.65-0.88 (trunk/shoulder line highest, elbows lowest); 0.71 of (asana, class, angle) cells have a reference.

**Fault types (authors' folder tags, not derived from the rule engine):** in-vocabulary Wrong rows containing legs / arms tags were flagged 81.7 % / 86.0 % of the time, but **out-of-vocabulary Wrong rows (neck/head/ankle/standing, which no angle can see) were flagged 82.5 %** - the same rate. So the flags are not tied to the actual fault. Attribution: the largest-deviation family is one of the tag's families in 74.4 % of 489 flagged rows; for true legs tags it named arms (54) about as often as legs (51), for arms tags it named arms (93) most often, which mostly reflects arm angles varying most.

**Sensitivity (reported, not used for selection):** K = 1.50 / 1.65 / 1.85 -> macro-BA 0.530 / 0.533 / 0.539 (the stretch correction is not what limits performance); delta -5 -> 0.507, +5 -> 0.530; the first-run F1-selected setting (p = 5, delta = 5) -> BA 0.560, F1 0.532.

## Why (evidence and hypotheses, not proven)
- Bands are wide: correct-pose IQR per angle is 12-31 deg on dev (`phase3/dev_auc_report.txt`), because stills mix unlabelled viewpoints and image-plane angles depend on view.
- Signal exists but is local (dev only, exploratory): 51 of 188 (asana, class, angle) cells have |AUC - 0.5| > 0.15 (e.g. Kati Chakrasana step 2 shoulders, Parvatasana step 4 elbows), median per angle only 0.04-0.13. A generic any-angle percentile rule dilutes it.
- Other candidate causes not tested: subtle faults, label noise, keypoint errors on stretched stills, single-performer variability.

## Limits
Single performer, single annotator (audit sec. 17), within-performer only; guards and block CIs reduce but cannot remove burst dependence; Bhujangasana and Vajrasana not evaluated; the test split has now been looked at once, so any further stills variant evaluated on it is a second look and must be flagged as such.
