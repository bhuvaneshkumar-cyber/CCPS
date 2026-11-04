# Phase 3 - final report (geometric rule engine)

Two named methods, never interchanged: **geo_v1_zero_supervision** (bands from correct examples only, every angle judged) and **geo_v2_lightly_supervised** (v1 bands, angle subset chosen with dev Wrong labels).
Each result below is one pre-registered look (PROTOCOL.md, PROTOCOL_V2.md, PROTOCOL_VIDEO.md; single-look markers `TEST_TOUCHED.json`, `VIDEO_TOUCHED.json`).

## Summary
| Stage | Method | Outcome |
|---|---|---|
| Stills test blocks | v1 | **Negative.** Macro balanced accuracy 0.533, 95 % block-bootstrap CI [0.480, 0.593] (chance 0.500); F1 below the always-'incorrect' baseline; 79 % of correct images flagged. (`PHASE3_STILLS_REPORT.md`) |
| Dev gate (stills) | v2 | **Stopped.** 1 qualifying cell vs. a label-permutation null mean of 0.94 (gate: >= 6 and >= 2 x null). Nothing frozen; v2 never run on stills-test or video. Thresholds were not relaxed afterwards. (`v2/STOPPED.json`, `v2/selection_report.txt`) |
| Video pilot (24 annotated clips) | v1 | **Mixed.** Same-view, cross-fitted: macro balanced accuracy 0.664 [0.565, 0.749]. Cross-view: 0.472 [0.391, 0.562] = chance. |

## Video stage (one look, v1 only) - `phase3_video_report.txt`
Data: 3,906 held, trimmed, 5 fps frames (Right 2,182, Wrong 1,724) from 92 annotated hold segments (Tadasana, Ardhakati Chakrasana, Kati Chakrasana x 4 camera angles; single annotator, marks placed during playback; 1.5 s trimmed from each segment end; segments < 4 s dropped). Bands [P15, P85] + 5 deg, fixed from the stills dev stage.
- **V2 same-view, cross-fitted** (reference = other holds / earlier part of the same Right clip): Tadasana 0.802 (flags 30 % of held-out Right, 91 % of Wrong), Ardhakati Chakrasana 0.624 (50 % / 75 %; 18 % abstained), Kati Chakrasana 0.567 (54 % / 68 %). Macro 0.664, CI lower bound 0.565.
- **V3 cross-view** (reference = the other three angle numbers of the same asana): Tadasana 0.415, Ardhakati 0.482, Kati 0.520; macro 0.472; 70-83 % of correct frames flagged.
- **V1 acceptance test for the stretch factor K** (stills K = 1.65 vs pooled Right video, 89 cells): median |difference| 4.5 deg; 43 of 89 cells (48 %) differ by > 5 deg; median band overlap 0.38. By angle (median |difference|): trunk tilt 1.7, hips 1.5-2.8, shoulder line 2.7, knees 3.0-3.7, right elbow 4.6, **left elbow 8.4, shoulders 8.3-9.5**.

**Reading (evidence-based, not proven):**
1. The stretch correction holds up for leg and trunk angles (median differences <= 3.7 deg, inside the +-5 deg residual). Arm angles disagree by 8-10 deg; that is more likely viewpoint mixing in the stills than K, but the test cannot separate the two (stated in advance).
2. With the camera view known and references taken from the same view, the engine has weak-to-moderate signal; **it does not transfer across views** (0.472). Viewpoint dependence of image-plane 2D angles is the dominant obstacle, matching the stills result.
3. Even in the most favourable design 30-54 % of correct frames are flagged.

## Limits (must appear in the paper)
Single performer; single annotator with late-biased marks (audit sec. 19); one execution per (asana, view, label) so no significance claims beyond the hold-level intervals; V2's Right-side numbers are optimistic (same execution); only three asanas have video step labels; five headline asanas have no video evaluation; Bhujangasana and Vajrasana were not evaluable on stills; Sarvangasana and Viparita Karani excluded by spec; head/neck angles excluded.

## Consequence for the architecture
The specification makes the geometric rule engine the primary quality signal. On this data neither variant supports that: v1 is at chance on stills and across views, v2 is not supported by dev evidence. Following the agreed fallback, **the BiLSTM + attention temporal model should carry primary weight for quality assessment (Phase 4)**, evaluated on video where the camera is fixed. The geometric engine remains useful only where the camera view is known and per-view references exist, and as the source of interpretable joint-level wording for the feedback engine. Whether the feedback engine should be driven by the temporal model's per-joint forecast errors instead is a Phase 5 design decision that needs the Phase 4 results first.
