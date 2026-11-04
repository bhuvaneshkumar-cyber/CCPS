# Moving PoseGuard to the RTX 5080 machine without corruption

**Principle:** nothing is trusted because a copy "looks fine". Every file is checked against a SHA-256 manifest, the environment is checked against the pins, the tests are re-run on the new machine, and the backbone is re-run on real frames and compared with the keypoints this machine produced. The source machine is never modified or deleted until the target passes every check (**copy, don't move**).

> **Transport is now GitHub (2026-11-04).** Clone the repository and follow [GPU_CHECKLIST.md](GPU_CHECKLIST.md). The same checks apply. The committed `MANIFEST.sha256`/`MANIFEST.json`, `ENVIRONMENT.json` and `PROBES.json` (built by `python tools/make_release.py --manifest-only`) replace the zip's internal manifest, and `python tools/verify_release.py --checkout . --gpu` runs steps 3-7 below on the clone; git's own object hashes replace steps 1-2. `.gitattributes` (`* -text`) stops git from rewriting line endings, which would otherwise change frozen files. The zip route below still works as a fallback.

## What travels, what does not
| Item | How |
|---|---|
| Code, tests, tools, audit/phase reports, annotations, keypoint caches, frozen configs, protocols | the bundle (`release/poseguard_<time>.zip` + its `.sha256`) |
| Dataset (3.2 GB) and backbone weights | **not in the bundle**: `tools/fetch_data.py` re-downloads and checks them against the pinned SHA-256 in `poseguard/config.py` (both URLs checked reachable on 2026-09-21, sizes match). Offline fallback: copy `data/yoga.zip` and `weights/yolo26m-pose.pt` by USB; `fetch_data.py` still verifies their hashes and never trusts them otherwise |
| `CLAUDE.md` | in the bundle: carries the owner's research-discipline rules (naming of v1/v2, pre-written selection rules, freeze-then-confirm, timebox). Claude memory is machine-local and does NOT travel |
| Python environment | rebuilt from `ENVIRONMENT.json` (pins); **torch must be a CUDA >= 12.8 build** for the RTX 5080 (Blackwell, sm_120; 16 GB VRAM - far more than needed: the backbone has 24.2 M parameters, the Phase 4 model 50.6 k) |
| `__pycache__`, `.venv`, `release/` | never |

## On this (source) machine
1. Stop every job (annotator windows, extraction, training). The builder refuses to run while a project process is alive.
2. Run the tests: `python tests/test_angles.py`, `test_feedback.py`, `test_backbone.py`, `test_phase3.py`, `test_geo_v2.py`, `test_videostage.py`, and `python tools/annotate_steps.py --selftest`, `python tools/agreement.py --selftest`.
3. Build: `python tools/make_release.py` -> `release/poseguard_<time>.zip` and `.zip.sha256` (records file hashes, exact environment, and 8 real-frame probes).
4. Copy **both** files to the target with a checksum-verified method (`robocopy <src> <dst> /V` on Windows, `rsync -c` on Linux, or a USB stick followed by step 3 below). Do not use a method that can silently truncate (interrupted copy) without the `.sha256` check.

## On the target (5080) machine
1. Check the zip hash by hand first (`certutil -hashfile poseguard_<time>.zip SHA256` or `sha256sum -c poseguard_<time>.zip.sha256`), then pull only the verifier and pin list out of the checked zip:
   `python -m zipfile -e poseguard_<time>.zip boot` (or extract just `tools/verify_release.py` and `ENVIRONMENT.json`).
2. Create a fresh virtual environment with **Python 3.14** (the source is 3.14.2; wheels below exist for cp314 on Windows and Linux). Install torch first, exactly the pinned version with the CUDA 12.8 build (verified available 2026-09-21):
   `pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128`
   then the other pinned libs (`libs` in `ENVIRONMENT.json`): `ultralytics==8.4.24 opencv-python==4.13.0.92 numpy==2.4.0 pandas==2.3.3 scipy==1.16.3 scikit-learn==1.8.0 onnx==1.22.0 onnxruntime==1.29.0 psutil==7.2.1`.
3. `python boot/tools/verify_release.py poseguard_<time>.zip --extract-to PoseGuard --gpu` (the `.sha256` file must be next to the zip). It checks, in order: zip hash -> zip CRC -> **every file's SHA-256 vs the manifest** -> no unexpected files -> re-fetch + verify weights and dataset -> environment (CUDA available, GPU matmul, ultralytics/torch versions vs pins) -> **every test suite** -> **probes on real frames** (worst keypoint difference vs the source machine; tolerance 3 px on GPU because GPU and CPU numerics differ).
4. Only when it prints `ALL CHECKS PASSED` start any GPU work. If anything fails: do not fix by hand; re-copy the bundle and re-run.
5. First GPU task = cross-device check: re-extract the stills and the 24 annotated clips' keypoints at full rate on the GPU and compare with the CPU caches (`phase3/kpts_stills_640.npz`, `kpts_video_s6.npz`). Differences within the probe tolerance confirm the CPU-derived results carry over; larger ones are reported, not hidden.

## GPU work, in this order (after `ALL CHECKS PASSED`)
Target changed from the RTX 5070 to the **RTX 5080** on 2026-11-04 (owner). Same architecture (Blackwell, sm_120), same CUDA >= 12.8 requirement, same pins; nothing in the code depends on which of the two it is.
1. **Dense keypoints**: `python scripts/extract_video_full.py` -> `phase4/kpts_video_full_cuda.npz` (80 clips, every frame, ~152k frames; resumable, atomic saves). This is the only heavy job; on the CPU it would take ~30-60 h.
2. **Cross-device check**: `python tools/cross_device_check.py` -> `phase4/cross_device_report.txt` (criterion in its docstring, fixed before the first run). FAIL is reported, not fixed.
3. **Owner freezes `phase4/PROTOCOL_PHASE4.md`** (resolve DECISION 1 headline arm, DECISION 2 success criterion, change the STATUS line). Training refuses to run before this.
4. **Train** (Right clips only): `python scripts/phase4_train.py` -> `phase4/frozen_temporal_cuda.json`, `phase4/model_*_cuda.pt`. Small model; minutes on the GPU (also runs on CPU).
5. **One look**: `python scripts/phase4_test.py` -> `phase4/phase4_test_report_cuda.txt`, marker `phase4/TEST_TOUCHED_cuda.json`.
6. Later: Phase 0 sec. 19 mark-lag check on the dense keypoints (not written yet: its onset rule must be written down before it runs); Phase 6 latency benchmark.
Plumbing smoke runs (`--smoke` on steps 4-5) use the CPU stride-6 cache and time-reversed Right copies as a null 'Wrong'; they never read a Wrong clip and write only to `phase4/smoke/` (excluded from bundles).

## Rules after cut-over
- **One source of truth.** From the moment work continues on the 5080, the old machine is read-only archive; never edit both copies (that is how two histories silently diverge).
- Frozen artefacts stay frozen: `phase3/PROTOCOL*.md`, `frozen_config.json`, `TEST_TOUCHED.json`, `VIDEO_TOUCHED.json` and `v2/STOPPED.json` must arrive byte-identical (the manifest proves it). A new look at any already-scored split is a flagged second look on the new machine too.
- Any GPU result is stored next to, not over, the CPU result, and labelled with the device.
- Keep the last verified bundle and its `.sha256`: it is the rollback point.

## Open question for the owner
Which OS does the 5080 machine run? Some CSVs (`audit/inventory_*.csv`, `manifest_dedup.csv`) store Windows-style `\` paths. Windows -> works as is. Linux -> paths need a one-time normalisation before use (tell me and I will add and test `tools/normalize_paths.py`; it is not needed on Windows).

## Tested on 2026-09-20 (source machine, simulated target)
Good bundle: all 7 verification steps passed (138 files; 6 test suites; 7 real-frame probes, worst keypoint difference 0.0000 px). Deliberate corruptions: (a) one flipped byte inside the zip -> stopped at step 1, nothing extracted; (b) one file altered inside the archive with the zip hash recomputed -> step 3 named the file and reported FAILED. Not yet tested: a GPU target (needs the 5080 machine; run with `--gpu`).
Note: as of 2026-09-21 `data/` holds only the Mendeley Yoga dataset (`yoga.zip`, `raw/`) plus two small metadata JSONs; nothing else needs to be transferred separately. If third-party files are added later (e.g. MPI-INF-3DHP helpers), give them their own pinned hashes before migrating.
