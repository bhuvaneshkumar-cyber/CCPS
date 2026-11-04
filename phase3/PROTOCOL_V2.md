# geo_v2_lightly_supervised - protocol (written BEFORE the selection is run; nothing below may change after the freeze)

**Name.** `geo_v2_lightly_supervised` (v2). It is NOT "the geometric rule engine". `geo_v1_zero_supervision` (v1) is the engine whose stills result was negative (phase3/PHASE3_STILLS_REPORT.md). Every table, filename and sentence names which of the two it reports; v2 never replaces or averages with v1. v2 uses dev Wrong labels to decide WHICH angles are judged; the numeric bands still come from correct examples only.

## 1. Cell and data
Cell = (asana, posture class, angle). Selection data = **dev stills only** (dev split of phase3/PROTOCOL.md sec. 2): dev Right + dev in-vocabulary Wrong, K = 1.65-corrected, tau-gated angle values. The stills test split is **not used by v2 in any way** (it was scored once for v1; v2 is confirmed on video instead).

## 2. Qualification rule (a cell qualifies as informative only if ALL hold)
- **Q1 support:** >= 60 Right and >= 60 Wrong dev values, each class from >= 4 distinct blocks.
- **Q2 effect size:** |AUC - 0.5| >= 0.20, AUC of the raw angle value, Right vs Wrong (AUC >= 0.70 or <= 0.30).
- **Q3 stability:** 1,000 block-bootstrap resamples (Right blocks and Wrong blocks resampled separately, with replacement); the AUC's 2.5th percentile is >= 0.60 if the point AUC > 0.5, or its 97.5th percentile is <= 0.40 if < 0.5.
- **Q4 fault-family fit:** the angle's family (legs = knee/hip, arms = elbow/shoulder, trunk = trunk_tilt/shoulder_line_tilt) appears in the fault tags of >= 30 % of the cell's dev Wrong images, so a cell cannot qualify through a viewpoint or pose-mix artefact unrelated to the labelled fault.
Thresholds (60, 4, 0.20, 0.60/0.40, 30 %) are set here and are not tuned.

## 3. Gate against noise-chasing (label-permutation null)
Repeat the ENTIRE selection (Q1-Q4, with Q3 computed only for cells passing Q1+Q2) on 200 copies of dev in which the Right/Wrong labels are permuted **between blocks within each (asana, posture class)**, preserving the number of Right and Wrong blocks and all within-block correlation. N0 = mean number of qualifying cells under the null; N_obs = observed number.
**v2 proceeds only if N_obs >= 6 and N_obs >= 2 x N0.** Otherwise: v2 is declared unsupported by dev evidence, nothing is frozen, no video run is made, and we stop (Option B endpoint).

## 4. Decision rule of v2
Bands: [P15, P85] of correct (Right) values + 5 deg (the v1 values frozen from the stills dev stage; not re-tuned). A frame/image is **incorrect** if any *selected* angle of its (asana, posture class) is outside its band, **correct** if all judgeable selected angles are inside, **abstain** if the class has no selected angle or none is judgeable (never a fault).

## 5. Freeze
The selected cell list is written once to `phase3/v2/frozen_selection.json` together with the sha256 of this file and of `poseguard/geo_v2.py`, N_obs and N0. The video script refuses to run if either hash changed, and records a single-look marker (`phase3/v2/VIDEO_TOUCHED.json`); a second look needs `--second-look` and is reported as one.

## 6. Confirmation on video (one look; together with the v1 video analyses of PROTOCOL_VIDEO.md)
Selected cells are transferred by key (asana, posture class, angle); numeric bands come from the annotated video Right clips exactly as in PROTOCOL_VIDEO V2 (same-view, cross-fitted) and V3 (cross-view). v1 (all angles) and v2 are reported side by side on the same frames.
**Pre-registered success criterion ("clearly separates"):** in BOTH V2 and V3, v2's macro balanced accuracy over the three asanas is >= 0.70, the hold-level bootstrap 95 % CI lower bound is > 0.50, and v2's macro balanced accuracy exceeds v1's by >= 0.10.
**If not met: stop.** No refinement, no re-selection. Report v1 and v2 as negative geometric results; the BiLSTM temporal model carries primary weight for quality assessment.

## 7. Timebox
Started at the time recorded below; hard stop 24 h later. Whatever is unfinished by then is reported as not done.

**Timebox started:** 2026-09-20 19:55 (local time); hard stop 2026-09-21 19:55.
