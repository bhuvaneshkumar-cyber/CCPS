# Phase 4 - temporal model protocol

STATUS: DRAFT - not frozen. `scripts/phase4_train.py` refuses to run (except `--smoke`) while this line says DRAFT.
To freeze: the owner reviews every number below, resolves the two DECISION items, and replaces this line with `STATUS: FROZEN by <name> <date>`.
After that, any change to this file changes its SHA-256, and the test script refuses to score.

Written 2026-11-04 before any Phase 4 model was trained and before any Wrong-clip forecast error was computed.
Method name: **temporal_bilstm_att_v1** (`poseguard/temporal.py`). Uses **no Wrong labels**: it is trained, early-stopped and calibrated on Right clips only.

## 1. Data
- All 80 Mendeley videos (10 asanas x Right/Wrong x 4 angle numbers). Keypoints: pinned backbone (YOLO26m-Pose, imgsz 640, tau 0.5), **every frame**, extracted on the GPU machine by `scripts/extract_video_full.py` -> `phase4/kpts_video_full_cuda.npz`. Its SHA-256 is recorded at freeze time.
- The cross-device check (`tools/cross_device_check.py`) must have been run and its report kept before training; its verdict is reported with the results, whatever it is.
- Angles: the 10 angles of `poseguard/angles.py`, no stretch correction (video). Unavailable angle = NaN, never imputed as a value.
- Time step: 30 fps resampled to **10 fps** (every 3rd frame). No step labels are used.

## 2. Model (fixed, not tuned)
- Input per time step (30 values): 10 angles / 180 (NaN -> 0), 10 availability flags, one-hot asana (10).
- Window W = 20 steps (2.0 s) of past input; target = the 10 angles **H = 5 steps (0.5 s) after the last input step**. The target frame is never inside the window.
- Encoder: 1-layer bidirectional LSTM, hidden 64 per direction, over the window only (bidirectional within the past window, causal with respect to the target, so it can run live). Additive attention pooling over the 20 steps, linear head -> 10 angles.
- Loss: MSE over available target angles only. Adam, lr 1e-3, batch 256, max 200 epochs, early stopping on Right validation loss (patience 15). These values are set here once; **no grid search**.
- **Ensemble of 5 seeds (0-4) per arm/fold; the forecast is the mean of the 5.** All 5 are always used (no seed is chosen).
- Baseline (always reported next to the model): **persistence** = last observed value of each angle in the window.

## 3. Splits
Arm **CHRONO** (the Phase 0 split policy): inside every Right clip, train = first 70 %, val = 70-80 % (first 2 s dropped as guard), test = 80-100 % (first 5 s dropped as guard). Deviation from Phase 0: Phase 0 proposed a >= 5 s guard everywhere; a 10 % val segment is only 4.6-11 s long, so a 5 s guard would leave almost no validation windows. Wrong clips: the same 80-100 % segment (5 s guard) is scored, so Right and Wrong are compared at the same position in the sequence.
Arm **LOAO** (leave one angle number out, 4 folds): train on the Right clips of the other three angle numbers (first 90 % train, last 10 % val with a 2 s guard); test on the full Right and full Wrong clip of the held-out angle number. Each clip is scored in exactly one fold; the four folds are pooled.
Windows never cross a segment boundary. Expected window counts at 10 fps (from the inventory frame counts, before any detection loss): CHRONO Right train 17,071 / val 823 / test 2,213, Wrong test 2,033; LOAO val 541-680 per fold. Windows overlap (stride 1 step), so these are not independent samples.

**DECISION 1 (owner): which arm is the headline.**
Recommendation: **LOAO**. In CHRONO the Right test frames come from the same recording the model was trained on, while every Wrong frame comes from an unseen recording, so CHRONO separation can be "novel recording" rather than "incorrect execution" (the same optimism as Phase 3 V2). In LOAO both sides are unseen recordings from the same unseen view. Both arms are always reported side by side, whichever is chosen.

## 4. Score and calibration
- Per sample and angle: z = min(|forecast - observed| / sigma, 10). sigma = median |residual| / 0.6745 on the Right **validation** windows of the same arm/fold, **per (asana, angle)** when that cell has >= 20 judged validation values, else the angle's value pooled over asanas (frozen with the model). The persistence baseline gets its own sigma the same way.
- An angle is judged in a sample only if its target is available AND it was observed at least once in the window (so model and baseline are scored on exactly the same angles). Sample score = mean z over judged angles; samples with < 3 judged angles abstain.
- **Decision unit = 1 s**: the mean score of the non-abstained samples whose target step falls in the same 1-s interval of a clip (a unit with none abstains; share reported). AUCs and counts below are over units, not windows.
- Phase 5 flag (not used for the headline): TemporalFlag when z >= k, **k = 3.0** (provisional).

## 5. Analysis (one look)
- Per asana: AUC = P(score of a Wrong sample > score of a Right sample) (+ 0.5 ties), Right = test units of Right clips, Wrong = scored units of Wrong clips.
- **Headline = macro AUC over the 8 headline asanas** (Anantasana, Ardhakati Chakrasana, Bhujangasana, Kati Chakrasana, Marjariasana, Parvatasana, Tadasana, Vajrasana). Sarvangasana and Viparita Karani: reported separately as supplementary (Phase 0 decision 4), never in the macro.
- Interval: 95 % percentile bootstrap, 2000 resamples, **resampling unit = (clip, 5 s chunk)** (units within 5 s are strongly dependent; resampling units or windows is not allowed). Stated as approximate: one execution per (asana, view, label).
- Paired comparison: macro AUC(model) - macro AUC(persistence), same bootstrap resamples.

**DECISION 2 (owner): the success criterion (rule 4, "clearly separates").** Proposed, on the headline arm:
(a) macro AUC lower 95 % bound > 0.55, AND (b) the paired difference model - persistence has a lower 95 % bound > 0.
If (a) fails: negative result, reported as such; no architecture or hyper-parameter is changed afterwards (timebox rule). If (a) holds and (b) fails: report that the learned model adds nothing over persistence.

## 6. Single look
`scripts/phase4_test.py` refuses to run unless this file is frozen and unchanged, `poseguard/temporal.py` and the keypoint file match the hashes recorded at training time. It computes everything, then records the look in `phase4/TEST_TOUCHED_<device>.json`, then prints. A second run needs `--second-look` and is reported as one.

## 7. Known limitation of LOAO
The angle number is not a reliable view label (audit sec. 8: Tadasana Wrong has angles 3 and 4 swapped). In such a fold the Wrong clip may show a view the model WAS trained on while the Right clip does not, which lowers Wrong errors: the bias works against finding separation, not for it. Stated, not corrected.

## 8. Revision log (all before any Phase 4 training run or any Wrong-clip forecast error existed)
- 2026-11-04 r1: first draft.
- 2026-11-04 r2: 5-seed ensemble (removes seed luck); robust per-(asana, angle) sigma with pooled fallback and z cap 10 (one keypoint glitch or one noisy joint cannot dominate a score); 1-s decision unit (reduces frame jitter; the counted unit is closer to an independent observation). Each was chosen for a stated reason, not from any result; the only runs so far are plumbing smoke runs on Right clips with a time-reversed null.

## 9. Not in scope here
Arm B of Phase 0 decision 3 (pretraining on MPI-INF-3DHP) is not built; it needs the 3DHP download and its own protocol. Per-step analysis (no step labels for 56 clips). Feedback-engine weighting (Phase 5, needs these results first).
