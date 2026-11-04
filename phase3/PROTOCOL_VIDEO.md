# Phase 3 video stage - protocol (written BEFORE any video result exists)

## Data and preparation
- 24 step-annotated clips: Tadasana, Ardhakati Chakrasana, Kati Chakrasana x 4 camera angles x {Right, Wrong}. Annotator: Bhuvanesh (project owner), single annotator, marks placed during playback (late bias, audit sec. 19). Keypoints: pinned backbone, imgsz 640, every 6th frame (5 fps), tau = 0.5, **no stretch correction (video is native aspect)**.
- Held segments = annotated segments with step label > 0. Consecutive marks with the same label are merged (stray repeated key presses). Segments shorter than 4 s are dropped; **1.5 s is trimmed from both ends of every kept segment**. Only trimmed frames are used anywhere below. Steps map to posture classes with the same table as the stills (`posture_classes`).
- The 'Angle n' number is used only as a clip-grouping id, never as a view label (audit sec. 8).
- Two clips were force-closed by the annotator with junk marks (Ardhakati Right angle 2: stray zero marks; Tadasana Wrong angle 1: a step-5 mark on the last frame); junk is handled by the merge/drop rules above, not edited by hand.

## Parameters (fixed now, NOT tuned on video)
Band = [P15, P85] of the held frames' angle values + 5 deg (the values frozen from the stills dev stage); decision rule and abstention (< 3 judgeable angles) exactly as in PROTOCOL.md sec. 4. Angles: the same 10; head/neck excluded.

## Analyses (all descriptive: 1 execution per (asana, view, label); no significance claims)
- **V1 acceptance test for K (Right clips):** per (asana, posture class, angle) with >= 20 video frames and >= 40 dev Right still values: median(still, K = 1.65) - median(video) and overlap of the two [P15, P85] bands. Cells with |difference| > 5 deg are listed as 'still-unreliable'. Limitation stated in advance: stills mix unlabelled views while video has four fixed views, so a difference cannot be attributed to K alone.
- **V2 same-view, cross-fitted (per asana x clip-angle):** reference bands from the Right clip; evaluated on (a) held-out Right frames and (b) the Wrong clip of the same angle number, both judged against the class of their annotated step. Cross-fit: a posture class with two holds in the clip (e.g. Tadasana steps 1 and 5) uses hold A as reference for hold B and vice versa; a class with one hold uses its first half as reference and the second half as test, with a 1.5 s gap. Right-side numbers are therefore optimistic (same execution).
- **V3 cross-view:** reference bands from the Right clips of the other three angle numbers of the same asana (pooled), evaluated on the held-out clip (Right and Wrong).
- Reported per asana: flag rate on Right frames, flag rate on Wrong frames, balanced accuracy, abstention, number of hold segments (the resampling unit for any interval; hold-level percentile bootstrap, stated as very approximate). No hyper-parameter, angle subset or threshold is chosen using these clips.
- Results are reported whatever they are.

## Clarifications (written before any video result existed; v2 was stopped at its dev gate, so **only the v1 analyses run on video**)
- Video bands need >= 10 reference frames per (posture class, angle) (2 s at 5 fps); otherwise that angle is not judged for that class. Decision rule = PROTOCOL.md sec. 4 unchanged (abstain if < 3 judgeable angles).
- Single-hold cross-fit: frames sorted by index; reference = frames before (midpoint - 0.75 s), test = frames after (midpoint + 0.75 s), i.e. a 1.5 s gap. Two-hold classes: each hold is tested with bands from the other hold(s) of the same clip.
- Wrong frames are judged with bands from ALL Right-clip frames of the same class in the same view (V2) or from the other three clips (V3); Right test frames use the cross-fitted / cross-view bands. Every frame is judged against the class of its own annotated step.
- Resampling unit for every interval = hold segment (clip, hold id); the macro figure is the mean over the three asanas. Balanced accuracy = (Right-frame specificity + Wrong-frame recall) / 2; abstained frames are excluded and their share reported.
- V1 acceptance test uses stills dev Right values (K = 1.65) versus the pooled Right video frames of all four angle numbers.
- Single look: `phase3/VIDEO_TOUCHED.json`; a second look needs `--second-look` and is reported as one.
