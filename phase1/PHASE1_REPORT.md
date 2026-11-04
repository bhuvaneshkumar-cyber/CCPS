# Phase 1 report - pose backbone integration

## Implemented
- `poseguard/backbone.py`: frozen pretrained **YOLO26m-Pose** wrapper. Refuses to run unless the weights match the pinned SHA-256; fixed input size (`BACKBONE_IMGSZ = 640`, batch 1, never dynamic); returns every person as COCO-17 (x, y, conf) in input-frame pixels; `primary()` = highest box confidence (single-user policy); `video()` iterator for a path or camera index; device 'auto' = CUDA after a small matmul health check that fails loudly on a missing-kernel wheel, else CPU with a printed notice. No training code exists anywhere.
- `poseguard/overlay.py`: skeleton drawing (left_* blue / right_* red by backbone name, sub-tau joints grey). `poseguard/config.py`: `BACKBONE_IMGSZ`, `BACKBONE_BOX_CONF` made explicit.

## Verified against real data
- `tests/test_backbone.py`, 8 tests pass: weights hash; unpinned weights refused; person-selection rule (hand-built fixture); well-formed output on a real video frame; **the wrapper reproduces the Phase 0 extraction on 9 real video frames to 0.0004 px (confidence diff 0.00000)**; blank image -> nobody found (edge-case fixture); video iterator indexing/stride; explicit device selection.
- `scripts/phase1_verify.py` -> `phase1/phase1_verify_report.txt`, `phase1/overlays/<asana>.jpg` (10 sheets: 4 video angles + 1 Right still + 1 Wrong still per asana). **I inspected 4 of the 10 by eye** (Anantasana, Sarvangasana, Viparita Karani, Parvatasana) plus 5 Phase 0 sheets; the rest are generated but not individually reviewed.

## Actual numbers
- **Video detection: 3,196 / 3,200 sampled frames = 99.88 %** (80 videos x 40 frames, imgsz 640).
- **Video angle coverage at tau = 0.5** (mean over all clips): knees 0.90-0.91, hips 0.93, trunk tilt 0.98, shoulder-line 0.98, arm elevation 0.80-0.83, elbows 0.74-0.75. Per asana table in the report. The two inverted asanas, which failed on stills, are good on video: Sarvangasana knees 0.93 / hips 0.97-0.98; Viparita Karani knees 0.90 / hips 0.98.
- **tau sensitivity** (mean coverage): tau 0.3 -> 0.5 costs only 3-6 points (knees 0.94 -> 0.90, elbows 0.81 -> 0.74); 0.7 costs a further 6-7. **Decision: tau stays 0.5, not tuned.** There are no ground-truth keypoints in the dataset, so tau cannot be tuned against accuracy; coverage changes little and a higher tau is the safer side for hallucinated joints.
- Stills sample (20 mid-sequence stills at 640 px): 5 of 20 had no detection (Anantasana Right and Wrong, Bhujangasana Right, Parvatasana Wrong, Viparita Karani Right) - lying-flat or close-up poses. Small sample, consistent with Phase 0's still-image weakness.
- CPU time 0.7-1.4 s/frame at 640 (noisy: other jobs were running). **Not an RTX 5080 number**; no latency claim is made.

## Deviations / limits
- Person selection is by box confidence only (no tracking). Phase 0 saw 1-4 % multi-person frames for a single performer, so a false second person is possible.
- Keypoint accuracy against ground truth is not measured (none exists for Mendeley); accuracy claims for the backbone are limited to the published COCO figures, cited in the paper, not re-derived.
- Backbone on the 5080: unverified (no GPU here); the GPU health check exists but has never run on CUDA.

## Consequence for the rest of the plan
The backbone is markedly more reliable on the **videos** (fixed camera, native aspect, 99.9 % detection) than on the **stills**. Videos should be the primary source for angle statistics; stills mainly supply step and fault labels and need the x-correction.

## Phase 2 status
`poseguard/angles.py` (10 angles, gating, still correction) is implemented with 8 hand-computed unit tests passing; sanity-checked on real keypoints; coverage tables above. Angle definitions and conventions are in the module docstring.
