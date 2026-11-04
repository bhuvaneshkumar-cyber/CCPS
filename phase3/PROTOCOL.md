# Phase 3 evaluation protocol (STILLS-ONLY stage) - written and frozen BEFORE any test-split result exists

Anything not stated here is not allowed to change after the test split is scored. The sha256 of this file and of `poseguard/phase3.py` is recorded in `phase3/frozen_config.json` when the dev stage finishes; the test script refuses to run without it and refuses a second look without `--second-look` (which is then logged in the results).

## 1. Scope
- Data: usable (deduped) stills of Anantasana, Ardhakati Chakrasana, Bhujangasana, Kati Chakrasana, Marjariasana, Parvatasana, Tadasana, Vajrasana. Sarvangasana and Viparita Karani are excluded (spec).
- Keypoints: pinned backbone through `poseguard.backbone` at imgsz 640 (`scripts/extract_stills.py`). Still x-coordinates corrected by `STILL_STRETCH_K = 1.65` before any angle. Confidence gating tau = 0.5.
- Labels: Right = correct, Wrong = incorrect (the dataset authors' labels; single annotator - see audit sec. 17). Step = folder step. Steps that are byte-identical copies of an earlier step share one **posture class**.
- Angles (10): left/right knee, hip, elbow, shoulder(arm elevation); trunk_tilt; shoulder_line_tilt. No head/neck.
- **This is not the final Phase 3.** Video-based ranges and the still-vs-video acceptance test are added after valid step boundaries exist (`audit/step_queue_phase3.txt`).

## 2. Split (deterministic, time-block based; there is no random component)
For every folder (asana, label, step): sort by camera timestamp; span = last - first; nb = clip(round(span / 40 s), 3, 10) equal-length blocks; block k = floor((t - t0) / span * nb).
Block k is **test** if k mod 3 == 2, otherwise **dev**. Images within g = min(4 s, 0.15 x block length) of an internal block edge are dropped (**guard**) from every split.
Consequences accepted: single performer, so results are within-performer; bursts are non-independent, so **every CI resamples blocks, never images**.

## 3. Reference ranges (dev Right stills only)
Per (asana, posture class, angle): band = [P_p, P_(100-p)] of the corrected, gated angle values, widened by delta degrees on each side. An angle gets a reference only with >= 40 dev Right values from >= 3 distinct blocks; otherwise it is **not judged** for that class.
Hyper-parameters chosen ONLY by 5-fold GroupKFold (groups = dev blocks) on dev, maximising mean-over-asanas F1 of the 'incorrect' class (tie -> higher precision, then larger delta):
p in {1, 2.5, 5} percent; delta in {5, 10, 15, 20} degrees (delta >= 5 covers the +-5 deg residual of the stretch correction, audit sec. 13).

## 4. Decision rule
An image is **incorrect** if any judged angle lies outside its widened band; **correct** if all judged angles are inside; **abstain** if fewer than 3 angles were judgeable (never counted as incorrect). Abstentions are reported per class and excluded from precision/recall.
(Any-angle rule; a k-of-n rule was considered and not adopted because k would be one more knob tuned on the same dev data.)

## 5. Fault families (ground truth = the authors' folder tags, NOT derived from the rule engine)
legs = tag contains 'leg' or 'knee' -> angles knee, hip. arms = 'hand' or 'palm' -> elbow, shoulder. trunk = 'back' -> trunk_tilt, shoulder_line_tilt.
Out-of-vocabulary (no angle can see it): 'neck', 'head', 'ankle', 'standing'. A Wrong image is **in-vocabulary** if its tag maps to >= 1 family; Wrong images with only out-of-vocabulary tags are excluded from the headline metrics and reported separately (their recall is expected to be low by construction).
Attribution: for a flagged in-vocabulary Wrong image, the predicted family is the family of the angle with the largest deviation beyond its band; the confusion matrix uses single-family tags only.

## 6. Reported metrics (all on the test blocks, one look)
Per asana and macro: accuracy, precision, recall, F1 of 'incorrect', abstention rate, per-angle coverage. 95 % percentile bootstrap CIs (2000 resamples of test blocks within asana; macro = resampling all blocks). Fault-family confusion matrix and per-family recall. Out-of-vocabulary recall. Any (asana, class) with no dev Right reference is reported as 'no reference', not scored.
**Sensitivity (reported, never used for selection):** K in {1.5, 1.65, 1.85}; delta +-5.
**Results are reported whatever they are**, including if the rule engine barely beats chance.

## 7. Amendments (each made before any test-split result existed; the test split has never been scored)
**A1 - selection criterion and p grid (after the first dev run).** First dev run under sections 3-4 as originally written: the best cross-validated setting was p = 5 (the largest value in the grid) with delta = 5 (the smallest), i.e. at the corner of the grid; and its macro-F1 (0.540) was **below the pooled F1 of a predictor that flags every image as incorrect (0.615; 2,262 of 5,096 dev rows are Wrong)**. With a positive-class F1 the optimum therefore drifts toward 'flag everything', which is not a meaningful classifier.
Changes: (a) the hyper-parameters are selected by mean-over-asanas **balanced accuracy** ((recall + specificity)/2, chance-level 0.5 for any constant predictor) instead of F1; tie -> higher F1, then larger delta. (b) the p grid is extended to {1, 2.5, 5, 10, 15, 20, 25}. delta stays {5, 10, 15, 20}: delta >= 5 is a fixed floor (stretch-correction residual), not a tuned value.
F1, precision, recall and accuracy are still reported for every asana; the always-'incorrect' baseline is reported next to them. The original F1-selected configuration is reported as a sensitivity row, not as the result. The first dev run's output is kept in `phase3/phase3_dev_report_first_run.txt`.
