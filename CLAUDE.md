# PoseGuard: standing research rules

This project is a publication effort. The owner wants tuning-until-it-works to be impossible by construction, because an earlier version was corrupted by silent substitution of fake data and the paper's claims must survive a hostile reviewer. Apply these rules without being reminded. (Copied from the owner's Claude memory so they travel with the project to the RTX 5080 machine; keep the two in sync.)

1. **Name variants distinctly, everywhere** (code, config, results tables, paper). A method that uses any label supervision beyond "correct examples define the ranges" is never called "the geometric rule engine". The zero-supervision `geo_v1_zero_supervision` and the lightly-supervised `geo_v2_lightly_supervised` (see `poseguard/config.py`) are never conflated or substituted for each other in reporting. Both results are always shown side by side.
2. **Write the selection rule down before running it**: an explicit minimum effect size and minimum sample count per cell, plus a control for the many implicit comparisons (e.g. a label-permutation null of the whole selection). Never pick cells or features by eyeballing which looked good in dev.
3. **Freeze from dev alone, then evaluate once on confirmation data** (the annotated videos). No iteration after seeing that result. A second look at an already-scored split is allowed only if flagged as a second look.
4. **Timebox exploratory variants** (about a day). If the frozen variant does not clearly separate correct from incorrect on the confirmation data, stop and report the negative result; the fallback endpoint is to report both negative geometric results and let the BiLSTM temporal model carry the primary weight for quality assessment.

## Migration rules (see MIGRATION.md)
- Copy, don't move. One source of truth after cut-over; the old machine becomes read-only archive.
- Frozen artefacts (`phase3/PROTOCOL*.md`, `frozen_config.json`, `TEST_TOUCHED.json`, `VIDEO_TOUCHED.json`, `v2/STOPPED.json`) must stay byte-identical. Once created, so must `phase4/PROTOCOL_PHASE4.md` (after its STATUS line says FROZEN), `phase4/frozen_temporal_*.json`, `phase4/model_*.pt` and `phase4/TEST_TOUCHED_*.json`.
- Every GPU result is stored next to, never over, the CPU result, and labelled with the device.
