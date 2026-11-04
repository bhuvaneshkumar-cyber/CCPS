# PoseGuard — Phase 0 dataset audit

All numbers below are produced by scripts in `audit/` and reproduce from `audit/*_report.txt`. Nothing here is projected.

## 1. Backbone actually used (pinned in `poseguard/config.py`)
**YOLO26m-Pose** (`yolo26m-pose.pt`, sha256 `2fbf1636…5443c4`). Existence verified in the official Ultralytics release
`ultralytics/assets` v8.4.0 (assets `yolo26{n,s,m,l,x}-pose.pt`); loads under ultralytics 8.4.24 with `kpt_shape=[17,3]`.
No fallback to YOLO11-Pose was needed. Size `m` is provisional until Phase 6 latency data exists.
Full-population statistics below were run at `imgsz=320` on CPU (this machine has no NVIDIA GPU). A 400-image stratified sample was also run at
`imgsz=640`: detection 88.5 % (640) vs 85.5 % (320), per-image detect/no-detect agreement 93.5 %, median keypoint disagreement 4.1 px (p95 37 px, on 256-px images).
**Treat all 320-px coverage figures as a lower bound.**

## 2. Dataset pin
"Yoga for all: A Comprehensive Collection of Yoga Images and Videos dataset", Kanorewala, Suryawanshi, Patil, Gunjal (Vishwakarma University).
DOI **10.17632/jc4mmnvcdk.1**, version 1, published 2023-04-21. `Yoga Postures Dataset.zip`, 3,258,915,524 B, sha256 `57851569…4be8c1` (verified after download).
Companion paper: *Yoga dataset: A resource for computer vision-based analysis of Yoga asanas*, Data in Brief 2023 (PMC10293982).

## 3. License and consent
- Mendeley API `data_licence`: **CC BY 4.0** (attribution required; state changes; no implied endorsement). Recorded from the API for this exact DOI/version.
- One performer, who is a dataset author; paper states written informed consent covering public sharing. No ethics approval was required per the paper.
- **Human3.6M and H3WB: role in this project is ZERO** (decision 1, §12). Not used for the stress test, pretraining or anything else. No H36M-derived data will exist in the repo or paper.
- **MPI-INF-3DHP:** no account or approval; license read from the shipped `license.txt`/`README.txt` (§12). Non-commercial; derivatives may not be redistributed; cite VNect (SIGGRAPH 2017) and the 3DV 2017 paper.

## 4. Composition (actual files)
| | count |
|---|---|
| still images (jpg, all 256×256) | 11,344 (6,013 Right / 5,331 Wrong) — matches the paper exactly |
| videos (mp4, ~30 fps) | 80 = 10 asanas × {Right,Wrong} × 4 angles; 720×1280 portrait (12 are 1280×720); 46–113 s each; 77,400 Right frames / 74,639 Wrong frames |

**The dataset is images AND video.** Stills are sparse 1-fps-ish photo bursts; videos are continuous.
Stills are NOT frames of the videos (thumbnail probe on Tadasana: image→nearest video frame L1 44–75 vs image→nearest other image 14–33, `image_vs_video_report.txt`),
so video is an independent modality with no frame-level leakage to the stills.

## 5. Label schema (exactly what exists)
- Stills: only folder names. `Images/<Asana>/<Asana> {Right|Wrong} Steps/<Asana> {Right|Wrong} Step <k>[- <fault tag>]/<camera timestamp>.jpg`.
  Labels = asana (10), Right/Wrong, **step index k** (2–5 sequential sub-postures per asana), and for Wrong images a free-text **fault tag** naming body parts
  (Legs, Knee(s), Ankle, Hand(s), Palms, Neck, Head, Back (Overbend/Bending), Standing, plus combinations; spelling is inconsistent, e.g. "Leg and Hand" / "Legs and Hand" / "Hands and Legs" → needs a normalisation map).
- Videos: only clip-level labels (asana, Right/Wrong, angle 1–4). **No step boundaries and no fault tags for video.**
- No keypoints, angles, bounding boxes, subject IDs or per-frame labels anywhere.
- Label provenance: author-declared — Right = "Daily Yoga Practice Routine" book; Wrong = mistakes deliberately performed; one author (Asst. Professor, Yoga & Naturopathy, Vishwakarma University) instructed the performer. Single performer, single instructor.

## 6. Per-class counts, and what deduplication does to them
Byte-identical (md5) image groups: 1,658 groups / 3,914 files. **612 groups occur in more than one asana folder** (Anantasana, Sarvangasana and Viparita Karani were filed from one session), and **104 groups carry both Right and Wrong labels** (same bytes, contradictory labels). `duplicates_report.txt`.
Policy (`dedupe.py`): a group is usable only if all members agree on (asana, label); keep one representative; drop ambiguous groups from training AND evaluation.

| asana | raw R / W | usable R / W | detect rate R / W (imgsz 320) |
|---|---|---|---|
| Anantasana | 640 / 535 | 283 / 294 | .95 / .97 |
| Ardhakati Chakrasana | 507 / 420 | 507 / 420 | .99 / .99 |
| Bhujangasana | 684 / 479 | 435 / 325 | .71 / .85 |
| Kati Chakrasana | 320 / 225 | 320 / 225 | .98 / 1.0 |
| Marjariasana | 845 / 748 | 845 / 748 | 1.0 / .95 |
| Parvatasana | 1375 / 1402 | 1375 / 1402 | .81 / .82 |
| Sarvangasana | 347 / 359 | **94 / 0** | .65 / — |
| Tadasana | 854 / 718 | 516 / 413 | .98 / 1.0 |
| Vajrasana | 65 / 86 | 65 / 86 | 1.0 / .94 |
| Viparita Karani | 376 / 359 | **123 / 0** | **.26** / — |
| **total** | 6,013 / 5,331 | 4,563 / 3,913 (8,476) | |

Flags: **Sarvangasana and Viparita Karani have no usable Wrong stills** (correct/incorrect cannot be evaluated on stills for them); Vajrasana is tiny (151 images);
Sarvangasana Right is 94. Effective sample size is far below image counts: 85 % of consecutive same-folder images are ≤1 s apart (`timestamps_report.txt`).
Every asana was shot in a single session on a single day, with Right and Wrong interleaved in time.

## 7. Backbone reliability per class (real detections, usable images, τ = 0.5)
Chain coverage = all constituent keypoints ≥ τ (missed detection counts as unavailable). Full tables in `analyze_kpts_report.txt`.
| asana | L arm | R arm | L leg | R leg | trunk |
|---|---|---|---|---|---|
| Ardhakati Chakrasana | .83 | .85 | .94 | .94 | .99 |
| Tadasana | .82 | .80 | .87 | .83 | .99 |
| Kati Chakrasana | .74 | .76 | .96 | .97 | .98 |
| Marjariasana | .67 | .60 | .89 | .85 | .97 |
| Anantasana | .63 | .66 | .94 | .94 | .81 |
| Vajrasana | .81 | .80 | .59 | .57 | .82 |
| Bhujangasana | .58 | .55 | .72 | .72 | .75 |
| Parvatasana | .65 | .65 | .45 | .45 | .72 |
| Sarvangasana | .33 | .30 | .60 | .62 | .60 |
| Viparita Karani | .07 | .04 | .22 | .21 | .17 |

Head keypoints (nose/eyes/ears) are ≥τ in only ~5–60 % of images in every class, so **neck/head angles are not reliably computable** at τ=0.5.
Visual inspection (`audit/kpt_viz/<asana>.jpg`, one sheet per class; I looked at Tadasana, Anantasana, Parvatasana, Sarvangasana, Viparita Karani): standing and most floor poses are good;
kneeling/seated poses lose the legs; **inverted poses give anatomically wrong skeletons at high confidence** (confidence gating does NOT catch these) and Viparita Karani is often not detected at all (close, overhead/tilted camera).
Visual sheets for the other five classes are generated but not yet individually reviewed.

## 8. Viewpoints
- **Videos:** fixed camera; the performer rotates. The 'Angle n' number is **mostly but not reliably** a view label: angle 1 = front and angle 3 = back in most upright clips, angles 2/4 = profile - **but Tadasana Wrong has angles 3 and 4 swapped (3 = profile, 4 = back) and Kati Chakrasana's twist confuses any single label** (`video_views_report.txt`, per-video keypoint classification; my earlier '1=front, 2=side, 3=back, 4=side' was verified on Right clips only). **Never use the angle number as a view label; classify per clip/frame.**
- **Stills:** no view label; handheld, wide-angle, frequently tilted or near-overhead (`sample_montage.jpg`). Viewpoint spans frontal, side, back and oblique. Not enumerable from labels.
- **Probable anisotropic resize in stills.** Frontal proxy (shoulder span / shoulder-to-hip length) for Tadasana: stills median 0.88, p95 1.25; front-angle video max ≈ 0.72 (`aspect_check_report.txt`).
  Consistent with the stills being stretched horizontally by roughly 1.7–2.2× (the paper quotes originals of 3264×1504 which, if portrait, gives 2.17×). **Not proven** (n and pose mix differ). *Superseded by §13:* a validated estimator gives an effective correction factor K = 1.65 (range 1.5–1.85), provisional, applied to stills only.
- Image-plane 2D angles are viewpoint dependent regardless; this goes in the paper's limitations.

## 9. Decisions
**Pose identification: user-selected (asana, step) mode.** Justification: the dataset is closed-set 10 asanas with known labels; a classifier would add an unspecified learned component and conflate classification error with quality error in every metric; nearest-match needs the same references we are deriving.
**Consequence found in this audit:** reference ranges must be per (asana, step), because steps within an asana are different postures — pooling them would produce ranges too wide to detect anything. Videos have no step labels, so a video-based path needs a step-assignment mechanism (not yet specified).

**Split policy (proposal, frozen before Phase 3):**
- Only `usable` stills. Sort each (asana, label, step-folder) by camera timestamp, cut into contiguous time blocks, assign whole blocks to train/val/test (≈60/15/25 %), and discard a ≥5 s guard band at each boundary. Never split within a burst.
- Videos: chronological split inside each clip (first 70 % train, next 10 % guard/val, last 20 % test, ≥5 s guard) as primary; leave-one-angle-out as a secondary view-generalisation test.
- Subject-independent splitting is **impossible** (one performer). All results are within-performer and must be labelled as such.
- Bootstrap CIs must **resample time blocks, not images** (bursts make images non-independent; i.i.d. resampling gives falsely narrow intervals).
- Perturbed samples never enter training.

**Annotation protocol / ground truth:**
- Existing Right/Wrong labels are performer-intent labels created under an expert's guidance, not derived from angles, so they are not circular with the rule engine. They are single-annotator; annotator credentials are recorded above (name not published in the paper).
- Fault-type ground truth = the folder fault tag (author-provided, body-part level), **not** derived from the rule engine. It is coarse (body-part, not angle) and Wrong-stills only. Video has none.
- Not yet available: a second independent annotator. Needed for (a) Cohen's κ, (b) any headline claim beyond within-author agreement. Proposed: an independent qualified yoga instructor re-labels a random test subset (stills, and video segments) under a written rubric; identity and credentials recorded. `kappa.py` will be written in Phase 3.

## 10. Consequences for later phases (things I will not silently work around)
1. **Phase 3 references from stills are unsafe until the stretch is resolved**; video gives undistorted angles but only 1 clip per (asana, label, view) and no step labels.
2. **Phase 4:** there are 40 Right clips (~2,580 s, 77k frames). Enough to train a small BiLSTM+attention forecaster, but the held-out set is small and single-performer.
3. Sarvangasana / Viparita Karani: **stills** have no usable Wrong images, low keypoint coverage and wrong skeletons at high confidence → excluded from stills-based evaluation. **Videos are a different story** (§12, decision 3): 100 % detection and leg/trunk coverage 0.85–1.0, so they get a separate, explicitly supplementary video evaluation.
4. Neck/head angle faults cannot be evaluated reliably (head keypoints).
5. Latency/memory: this machine has no NVIDIA GPU (Intel UHD only), torch is CPU-only. RTX 5080 measurements need the workstation with a CUDA ≥12.8 torch build.

## 11. Open items (state after the decisions in §12)
Pre-Phase-1 gate — none of these is resolved yet:
- [ ] Step boundaries for the 80 videos: **REDO REQUIRED.** The submitted file (annotator field `YOURNAME`, all 80 videos 'done') holds exactly one mark at frame 0 per video and no boundaries; it is quarantined in `audit/invalid/`. Tool now refuses such input (§15).
- [x] Second annotator: **explicitly DEFERRED with a written reason (§17)**; no second annotator is available before Phase 1. Not required for Phases 1-6; changes only how the headline claim may be worded.
- [~] Distortion-correction factor: **provisional k = 1.65 (range 1.5–1.85)**, §13. Derived from 3 standing asanas × all 4 video views; floor-pose asanas could not be validated; final acceptance test is Phase 3 still-vs-video reference-range agreement.
- [x] Pose identification: **user-selected (asana, step)**, confirmed by the project owner.
- [x] RTX 5080: deferred by the owner until the heavy-processing phases (Phase 4 training, Phase 6 benchmarking). Phases 0-3 and 5 are CPU-only per the spec. First-use check still required: `torch.cuda.is_available()` + a small GPU matmul with a CUDA >=12.8 build.
- [x] Dataset DOI confirmed by owner (spec now pins 10.17632/jc4mmnvcdk.1).
- [x] H36M / MPI access requests: superseded by §12.

## 12. Decisions log (post-audit, owner-confirmed)
1. **Human3.6M / H3WB: dropped entirely. Role = zero.** Evidence from the H3WB README (read in full, twice, independently): H3WB releases annotation JSONs only; pixels must come from the official H36M site under its EULA, which this project does not have; the RGB→3D test set carries `image_path` + bbox with no ground truth (leaderboard scoring); the 2D→3D / I2D→3D test releases are coordinate files with an `image_path` field, not images.
   It therefore supplies neither backbone stress-test images nor temporal sequences. Mirrors of H36M video are not treated as a substitute (EULA still applies). The paper does not use or mention H36M data as a source.
2. **Stress test uses MPI-INF-3DHP alone.** Test set TS1–TS6 (`mpi_inf_3dhp_test_set.zip`, 6,936,910,586 B by HTTP HEAD, live). Constraints to state in the paper: PCK/OKS on the **12 shared COCO body joints only** (3DHP has no nose/eyes/ears, so partial OKS); presence of 2D test annotations is expected but **not yet verified** (test-set file format inspected only after download); no rule-engine evaluation on 3DHP.
3. **BiLSTM pretraining is an explicit ablation, not the default.** Arms: (A) from scratch on Mendeley video, (B) pretrained on 3DHP-derived angle sequences then fine-tuned. Success criterion is not forecast MAE alone but **which arm gives cleaner separation between correct and incorrect execution** (forecast-error distributions on held-out Right vs Wrong clips). Rationale: a forecaster pretrained on generic motion may forecast incorrect yoga equally well, blunting the deviation signal. Constraints if arm B is run: angles derived from backbone output on 3DHP frames (not GT); resample 25/50 fps (3DHP) and 30 fps (Mendeley) to one common time step; a pretrained checkpoint inherits 3DHP's non-commercial restriction; the 3DHP-derived angle sequences may not be redistributed.
4. **Sarvangasana / Viparita Karani: exclusion scoped to stills.** Evidence: `inverted_video_report.txt`, `inverted_video_overlay.jpg` — 16 clips × 20 frames, detection 100 %, leg coverage 0.85–0.94, trunk ≈ 1.0, arm coverage 0.1–0.3 in side views (self-occlusion). **This is signal, not a metric**: 8 overlay frames inspected by eye, no ground-truth keypoints. Reporting rule: a separate **supplementary** video evaluation — leg and trunk angles only, clip-level correct/incorrect, 4 Right + 4 Wrong clips per asana, sample size stated up front — never merged into the headline confusion matrix. Before that evaluation is reported, a proper keypoint-correctness check for these clips is required (e.g. manual annotation of a fixed frame sample), since coverage does not detect confident-but-wrong skeletons.

## 13. Distortion-correction factor for the stills (derivation, validation, limits)
Correction applied to still-derived keypoints only: `x_corrected = x / STILL_STRETCH_K` (`poseguard/config.py`), **K = 1.65, plausible range 1.5–1.85, provisional.**
Scripts: `extract_video_kpts.py` (80 videos × 40 frames, imgsz 640), `distortion_pseudo.py`, `distortion.py`, `distortion_paired.py`; outputs `distortion_report.txt`, `distortion_paired_report.txt`.

**Method.** Two estimators, each checked against a *known* transformation before its number on the real stills is used (validation frames = real video frames resized to 256×256, which imposes a known geometric x-stretch of 1280/720 = 1.778, and a uniform-resize null with k = 1.0, both run through the real backbone; estimator validation only, never training/evaluation data).
Pass criterion fixed before any result was seen: |k̂ − k_true| ≤ 0.15. Bootstrap CIs resample time blocks (clips / 60-s still blocks), not images.

| estimator | null (k=1.0) | squash (k=1.778) | verdict |
|---|---|---|---|
| A: nearest-video-pose Procrustes matching, leave-one-view-out, all asanas | 3.00 (grid edge) | 3.00 (grid edge) | **FAIL — discarded.** Biased toward large k; its 1.80 on real stills is meaningless and is not used. |
| B: frontal shoulder-span/torso ratio (q90, upright torsos), standing asanas | 1.00 [0.96, 1.05] | 1.56 [1.47, 1.62] | null PASS, squash FAIL by −0.22 (−12 %) |

**Why B under-recovers (paired test, same frames under both conditions, both 256 px wide).** With identical geometry, the backbone's output deviates from the applied transform:
x-spans shrink (shoulder 0.93, hip 0.91 vs ideal 1.00) and vertical bone lengths come out at 0.52–0.59 vs ideal 0.563 (thigh/shin low, torso high).
The backbone partly regresses stretched bodies toward normal proportions, so the **effective keypoint-space stretch for a known 1.778 is ≈ 1.67 (bone-pair range ≈ 1.58–1.74)**, not 1.778. B's shortfall is therefore mostly a real backbone effect, not just estimator error. This also means **the right correction is the effective factor, and a single scalar cannot undo a bone-dependent distortion**.

**Real stills (B, usable stills, block-bootstrap 95 % CI), and what each hypothesis predicts in the same measurement space (pseudo-calibration 0.878):**
| asana | real B | implied geometric k |
|---|---|---|
| Tadasana | 1.71 [1.56, 2.28] | 1.95 [1.78, 2.60] |
| Kati Chakrasana | 1.52 [1.49, 1.64] | 1.73 [1.70, 1.87] |
| Ardhakati Chakrasana | 1.42 [1.38, 1.47] | 1.62 [1.57, 1.68] |
| mean | 1.55 | 1.77 → effective ≈ 1.66 |

- H1 "stills are 16:9 portrait frames squashed to 256×256" (k = 1.778, predicts B ≈ 1.56): consistent with 2 of 3 asanas (Ardhakati slightly below).
- H2 "originals 1504×3264 per the paper's 3264×1504" (k = 2.170, predicts B ≈ 1.90): rejected by 2 of 3 asanas. The paper's dimensions are probably not the real aspect ratio.
- Neither is proven. Across-asana spread (1.42–1.71) exceeds within-asana CIs, so unmodelled real-still factors exist (handheld perspective, close range, lens).

**Named limitations (state in the paper).**
1. Only 3 standing asanas were measurable; there is no validated estimator for floor poses. Applying one K to all asanas assumes a single capture/export pipeline (same device, all images 256×256), which is plausible but unverified.
2. Residual: K uncertainty ±0.15 (≈ ±9 %) → per-segment orientation error up to ≈ 2.6°, joint-angle error up to ≈ 5° (≈ 3.5° at ±6 %). Reference ranges and rule tolerances derived from stills must be wider than this, or should be cross-checked against video.
3. Correction is applied to keypoints after the backbone, so backbone localisation degradation on stretched inputs is not undone.
4. Head/neck and arm keypoints were not calibrated (torso/leg bones drive B and the paired test).

**Acceptance test (Phase 3, to be run before any still-derived range is used).** For asanas/steps that exist in both modalities, still-derived (K-corrected) reference angle ranges must agree with video-derived ranges; any angle whose still-vs-video difference exceeds the ±5° residual is treated as still-unreliable. Sensitivity of results to K ∈ [1.5, 1.85] must be reported.

## 14. Step-boundary annotation tool and a second structural finding
**Tool:** `tools/annotate_steps.py` (tkinter window, cv2 decode only; the installed cv2 is the headless build, so `imshow` is unavailable and the environment was not modified).
Keyboard-driven: play/pause, ±1 frame / ±1 s / ±10 s, click-to-seek timeline, mark step k or 0, delete, undo, Enter = done+next, autosave after every change (atomic write).
Per-annotator output file (`audit/step_annotations_<name>.json`; annotator identity is required), so a second annotator is comparable later. `--check` validates and exports `audit/step_segments_<name>.csv`
(flags: not done, missing steps in a Right clip, non-monotone step order, step > N, segments < 15 frames). Reference panel shows two labelled stills per step so the annotator knows which posture is which.
**Verified:** `--selftest` (hand-constructed mark fixtures: segments, delete, undo, missing/non-monotone/short-segment flags, CSV export); scripted-key run of the real window
(seek, mark, undo, invalid step rejected, playback ≈ real time, done→next video, JSON persisted); seek returns bit-identical frames to sequential decoding (mean abs diff 0.000 at frames 299/599/900/1199), so frame indices are reliable.
**Not verified:** a human annotating all 80 clips. All scripted-test annotations were deleted; none are in the repo.

**Protocol** (also in the script header): press the step number on the first frame where the posture is *reached* (main movement stopped), press 0 at the first frame of *leaving* it (0 = transition/idle); labels hold until the next mark; label by **sequence order**; do not view any rule-engine/model output.

**Finding — several steps are byte-identical copies of earlier steps** (`step_duplicates_report.txt`): Tadasana step 5 = step 1 and step 4 = step 2 (Right and Wrong), Bhujangasana step 3 = step 1 (Right and Wrong).
12 of 70 step folders (4 of 20 asana×label combinations) are 100 % copies; the other six asanas have none. Consequences:
1. Distinct postures are fewer than the step counts: Tadasana 3 (not 5), Bhujangasana 2 (not 3). Per-(asana, step) reference ranges should be built per **posture class** (steps merged when their stills are identical) and the copies must not be counted as independent samples.
2. The Wrong stills for a step and for its copy are the same files, so they carry no independent information about the return phase.
3. In video the return phase is still a real, separately annotated segment; whether it should share a reference range with its first visit is decided in Phase 3 from the annotated video, not assumed.

## 15. Step-annotation incident, safeguards, and the second-annotator decision
**What was found.** `step_annotations_YOURNAME.json` (annotator name is the usage-example placeholder): 80/80 videos marked done, each with **one mark at frame 0** (labels inconsistent across asanas: e.g. '1' for Anantasana, '0' for Ardhakati Chakrasana, Sarvangasana Right angles 1-3 '0' but angle 4 '1'); no step boundaries at all. File times (tool created 13:35, last save 13:41) are also inconsistent with marking 80 clips of 47-113 s. It is therefore **not step-boundary data** and is not used anywhere; it was moved (not deleted) to `audit/invalid/`. **Cause (later established): a protocol misunderstanding, not a fast pass.** The annotator's own description was 'mark the shown steps; if not shown assume step 1 and 0', i.e. pressing 1 (or 0) once at frame 0 as a default. A second file (`step_annotations_Bhuvanesh.json`, 13:50) shows the same pattern - 10 videos each with one mark at frame 0 labelled step 1, all forced through the first version of the safeguard. The clarified protocol: 0 is not a default (it means moving/idle/no matching step); step 1 is never assumed; press a step number only when the performer ARRIVES at that posture and 0 when they LEAVE; a mark holds until the next mark. A still-open annotator window from before the fix kept re-saving `step_annotations_YOURNAME.json`; all annotator windows must be closed before files are moved or renamed.
**Tool safeguards added** (`tools/annotate_steps.py`, verified by `--selftest` and a scripted run of the real window): Enter refuses a video whose marks fail `problems()` (fewer than 2 marks, a Right clip missing steps, non-monotone order, step > N, segment < 15 frames) - a second Enter forces it and records `forced: true`, which `--check` reports - **except fewer than 2 marks, which is a hard block that cannot be forced** (added after the single-mark files above); a permanent legend in the status bar states what 1..N and 0 mean; placeholder annotator names (`YOURNAME`, `NAME`, `ANNOTATOR`) are rejected.
**Agreement script** (`tools/agreement.py`, `--selftest` passes on hand-computed kappa values; labels-mode smoke-tested on a hand-built fixture; steps-mode not yet run on real data because no second annotation exists): frame-level Cohen's kappa over frames labelled by both annotators plus start/end boundary offsets for matched segments, and kappa for correct/incorrect or fault-type labels; 95 % CIs resample whole clips/blocks, not frames.

**Second annotator (decision).**
- *Step boundaries (needed now):* the project teammate. This task needs the written protocol and the reference thumbnails, not expert yoga judgement, so a teammate is appropriate. They annotate a stratified subset - 20 of the 80 clips (both labels, all four angles, every asana represented) - in their own file (`--annotator <their real name>`), without seeing the first annotator's file or any rule-engine output. Reported: frame-level kappa with CI and boundary-offset medians/p90.
- *Correct/incorrect kappa (needed for the headline claim, later):* the spec requires 'genuine familiarity with correct asana form', so this must be someone with real yoga training (the teammate only if that is true; otherwise a yoga instructor - certified if possible). Blind protocol: items drawn from the frozen test split, shuffled, with folder names, fault tags and the other labeler's answers hidden; label = correct/incorrect (+ body part of the fault if incorrect). Sample large enough for a usable CI (target >= 100 items across >= 20 independent time blocks; fewer blocks give intervals like the [-0.23, 1.00] seen in the 4-block smoke test). Record name, credentials and years of practice with the labels.
- The same person doing both roles is acceptable if they qualify, but the credential must be recorded either way.

## 16. Feedback engine (Phase 5, built ahead of Phases 1-4 at the owner's request) and left/right verification
**Built (all verified by runnable tests):** `poseguard/lr.py` (the single L/R convention), `poseguard/angles.py` (Phase 2: 10 angles with confidence gating, still x-correction hook), `poseguard/rules.py`
(Phase 3 comparison logic only - **no reference ranges exist in code**; they must come from real data), `poseguard/feedback.py` (fusion + wording + severity), `tests/test_angles.py` (8 tests, hand-computed values),
`tests/test_feedback.py` (10 tests, hand-built TEST keypoints incl. the spec's 'right knee 20 deg beyond reference' -> says right, not left), `tools/verify_lr.py`.
Mutation check: forcing the wrong convention makes the spec's L/R tests fail, so they can catch a mirrored-side bug. Angles on real video keypoints look plausible (e.g. standing knees median ~174 deg, trunk tilt 1-4 deg) - sanity only, nothing judged.

**Left/right, what is and is not established.**
- *Established (objective, real data):* `tools/verify_lr.py --dataset`, criteria fixed before running (upright Tadasana/Ardhakati, per-frame view from face visibility, profile excluded): facing the camera 172/175 = 0.983 frames have left_shoulder.x > right_shoulder.x; facing away 159/164 = 0.970 have left < right.
  => the backbone's 'left_*' is the subject's anatomical left on non-mirrored input (`BACKBONE_LEFT_IS_ANATOMICAL_LEFT = True`). The first attempt (view taken from the angle number) gave 0.875 on the back-facing side; that was my error - one clip was a profile view - and is why the check now classifies view per frame.
- *Residual risk:* 8 of 339 frames (2.4 %) violate the rule, in profile-ish or arm-occluded clips. Keypoints alone cannot say whether these are genuine left/right swaps or view-classification errors. If swaps, live feedback could flicker between sides; Phase 6 should smooth the side per limb over a window. Not implemented.
- *NOT established:* whether a live camera feed is mirrored. `INPUT_MIRRORED = False` is only correct for un-flipped sources. The spec's raised-arm recording has not been done; the script exists (`--video clip --raised left|right`) and was smoke-tested on a Mendeley clip (mechanics only, the raised side there was inferred, not independently known). **Required before Phase 6:** run it on the actual camera pipeline with your own known raised arm.

**Provisional design choices (none fitted to data; recalibrate after Phases 3-4):** severity tiers at 10 and 25 deg deviation; temporal-flag fusion (a flag raises the tier by one; a flag on an in-range angle gives a low-severity 'hold steady' note; a flag on an unavailable angle is ignored; temporal=None is the geometric-only ablation arm); the wording templates (generic per joint, not per asana).
**Not done / not claimed:** no real evaluation of the engine exists (there are no reference ranges yet); wording has not been reviewed by a yoga-knowledgeable person; no on-screen overlay (Phase 6).

## 17. Deferral: second annotator (written reason, per the spec's 'resolved or explicitly deferred with a written reason')
**Decision (project owner):** proceed without a second annotator; none can be obtained in the available time. **Reason it is safe to defer:** no technical phase (1-6) depends on it; it affects only the strength of the correct/incorrect and step-boundary *validity* claims.
**Consequences accepted, and the wording the paper must use:**
- Correct/incorrect results are reported as **agreement with the dataset authors' labels** (one performer, one instructor, labels record performer intent, annotator unnamed in the source paper). They are not reported as 'the system correctly judges yoga posture'.
- **No Cohen's kappa / inter-annotator agreement is reported.** Limitations must state this, that generalization is within-performer only, and that step boundaries are single-annotator.
- Fault-type ground truth = the authors' folder tags (not derived from the rule engine); same single-source caveat.
**Mitigations retained:** (1) cite the measured source-label inconsistency (104 byte-identical image groups labelled both Right and Wrong, `duplicates_report.txt`); (2) guard band of +-1 s around every step boundary when deriving reference ranges and scoring (so imprecise boundaries cannot bias results); (3) optional intra-rater check - the owner re-annotates ~10 clips after >= 48 h under a different annotator name and `tools/agreement.py steps` measures their own timing noise (reported as intra-rater, never as inter-rater); (4) block-bootstrap CIs; (5) still-vs-video reference agreement test in Phase 3.
**Kept open:** protocol, `tools/annotate_steps.py` and `tools/agreement.py` are ready; a small blind second-labeler pass (~60-100 items by someone with real yoga training) can be added before submission without changing any code.
**Not acceptable substitutes:** the rule engine, perturbed samples, or the AI assistant as a 'second annotator'.
**Still owed by the owner (not blocking Phases 1-2, needed at Phase 3):** valid step boundaries for the 80 videos; annotator identity must be a real name.

## 18. Step-boundary requirement narrowed from 80 to 24 clips (owner decision + rationale)
The owner will annotate only the videos that require boundaries. Rule used: boundaries are needed only where the **geometric per-(asana, step) analysis runs on video**; the temporal model (Phase 4) forecasts angle vectors and needs no step labels, so all 80 clips are still used there.
**Queue (`audit/step_queue_phase3.txt`, run with `--queue`)**: Tadasana, Ardhakati Chakrasana, Kati Chakrasana, 4 angles each. **Priority 1 = the 12 Right clips** (video reference ranges + the still-vs-video acceptance test for the stretch factor K, which was only measurable on these standing asanas); **Priority 2 = the 12 Wrong clips** (video-based correct/incorrect evaluation for the same asanas).
**What is lost for the other 56 clips:** no video-based per-step geometric ranges or geometric evaluation for Anantasana, Bhujangasana, Marjariasana, Parvatasana, Vajrasana, Sarvangasana, Viparita Karani; their headline per-class metrics come from the stills (steps are labelled in the folder names) and the video supplementary evaluation of the two inverted asanas is limited to the temporal signal. This must be stated in the paper.
Tooling: `tools/annotate_steps.py --queue FILE`; a clip only counts as done with >= 2 marks (single-mark 'forced' clips are re-queued automatically).
Phase 3 stills stage is pre-registered in `phase3/PROTOCOL.md` (split, grid, decision rule, fault families, single look at the test split).

## 19. Annotation timing precision and the step-1 omission (owner's annotation, in progress)
**Timing.** The step marks land on arbitrary frames (e.g. 688, 1006, 1344), not on multiples of 30/300 that the seek keys would give: they were pressed while the video played, without frame-stepping (a/d). Expected effect: a **late bias** of roughly 0.3-0.5 s (human reaction), about double at 2x playback. A late arrival mark only trims the start of a held posture; a late departure mark lets ~0.5 s of movement into the segment. Segments annotated so far last 7-20 s, so the effect is small. Frame-stepping was a suggestion in the tool guidance, not a protocol requirement, so the annotations are **not invalidated**.
**Mitigation adopted:** before deriving ranges or scoring on video, trim **1.5 s from both ends of every annotated segment** (segments shorter than 4 s dropped). **Planned check (RTX 5080 session):** with dense keypoints for the annotated clips, measure each mark's lag to the real motion onset (joint-angle speed), report the median lag, and flag any 'held' segment that still moves; nothing is re-annotated unless that check flags it.
**Step-1 omission.** Ardhakati Chakrasana Right angles 1 and 2 were finished without step 1 and forced past the 'MISSING STEPS [1]' warning. Step 1 in this asana is the arm extended out to the side (visible at ~15 s in angle 1: `audit/ardhakati_step1_check.jpg`), not the initial standing posture. Marks for steps 2-5 matched the reference stills.
**Cleared (owner request):** all entries flagged forced were reset in `step_annotations_Bhuvanesh.json` (backup in `audit/invalid/`): 8 single-mark Anantasana entries removed (no content, outside the queue); the two Ardhakati clips kept their 7 marks each but were reopened (done=False, forced flag cleared) so step 1 can be added. `--check --queue` now reads 0/24 truly done.

## 20. Phase 3 stills stage: result (details in phase3/PHASE3_STILLS_REPORT.md)
Pre-registered, one look at the test blocks: **the geometric rule engine on stills does not separate correct from incorrect above chance** - macro balanced accuracy 0.533, 95 % block-bootstrap CI [0.480, 0.593]; F1 below an always-'incorrect' baseline; 79 % of correct images flagged; out-of-vocabulary faults flagged as often as detectable ones. Bhujangasana and Vajrasana not evaluable. Amendment A1 (selection by balanced accuracy, extended p grid) was made before any test result and is documented. Stretch correction K is not the limiting factor (BA 0.530-0.539 across K 1.5-1.85).
**Consequence:** the stills cannot support a positive claim for the geometric engine. Video (fixed camera, native aspect, per-view) is the remaining route; per-view/per-step ranges need the owner's step boundaries (queue in audit/step_queue_phase3.txt). Any stills variant evaluated on the same test blocks is a flagged second look; the cleaner confirmation is the video data.

## 21. Phase 3 closed (details: phase3/PHASE3_REPORT.md)
v1 stills: negative (macro BA 0.533 [0.480, 0.593]). v2: stopped at its dev gate (1 qualifying cell vs null mean 0.94), never run on test or video. v1 video pilot (one look, 24 clips): same-view cross-fitted macro BA 0.664 [0.565, 0.749], cross-view 0.472 [0.391, 0.562]. K acceptance test: legs/trunk within the +-5 deg residual, arm angles 8-10 deg apart (view mixing likely, not separable from K). Agreed fallback taken: the temporal model carries primary weight (Phase 4). Migration tooling: MIGRATION.md, tools/make_release.py, verify_release.py, env_check.py, fetch_data.py, probe_check.py.
