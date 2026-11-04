# PoseGuard

Yoga posture quality assessment from video: a frozen pretrained pose backbone (YOLO26m-Pose) gives COCO-17 keypoints, from which 10 joint angles are computed. Two families of quality signal are evaluated:

- **Geometric rule engine.** Per-(asana, posture) angle bands from correct examples: `geo_v1_zero_supervision`, and the lightly supervised `geo_v2_lightly_supervised`.
- **Temporal model.** `temporal_bilstm_att_v1`, a BiLSTM + attention forecaster trained on correct executions only. Its forecast error is the deviation signal.

The project is run as a pre-registered study. Every evaluation has a written protocol, a freeze step that records hashes, and a single look at held-out data. The rules are in [CLAUDE.md](CLAUDE.md).

## Status
| Phase | What | State |
|---|---|---|
| 0 | Dataset audit | done: [audit/PHASE0_REPORT.md](audit/PHASE0_REPORT.md) |
| 1-2 | Backbone and joint angles | done: [phase1/PHASE1_REPORT.md](phase1/PHASE1_REPORT.md) |
| 3 | Geometric rule engine | done, **negative** on stills and across views: [phase3/PHASE3_REPORT.md](phase3/PHASE3_REPORT.md) |
| 4 | Temporal model | code and tests done; protocol is a **draft** ([phase4/PROTOCOL_PHASE4.md](phase4/PROTOCOL_PHASE4.md)); GPU steps in [GPU_CHECKLIST.md](GPU_CHECKLIST.md) |
| 5-6 | Feedback fusion, live latency | later |

## Data (not in this repository)
"Yoga for all: A Comprehensive Collection of Yoga Images and Videos dataset", Kanorewala, Suryawanshi, Patil and Gunjal, Mendeley Data V1 (2023), doi:[10.17632/jc4mmnvcdk.1](https://doi.org/10.17632/jc4mmnvcdk.1), licensed **CC BY 4.0**.

This repository contains material derived from that dataset: keypoint caches, angle tables, and overlay and montage images in `audit/` and `phase1/`. It does not contain the dataset itself.

`python tools/fetch_data.py` downloads the dataset (3.2 GB) and the backbone weights and checks both against the SHA-256 values pinned in [poseguard/config.py](poseguard/config.py).

## Setup
Requires Python 3.14. On Windows, run:

```powershell
py -3.14 -m venv .venv; .venv\Scripts\Activate.ps1
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128   # CPU-only machines: drop the index URL
pip install -r requirements.txt
pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93                          # see the OpenCV note in requirements.txt
python tools/fetch_data.py
python tools/verify_release.py --checkout .        # add --gpu on a CUDA machine; must end with ALL CHECKS PASSED
```

## Layout
| Path | Contents |
|---|---|
| `poseguard/` | library: backbone, angles, left/right convention, rules, Phase 3 stages, temporal model, feedback |
| `scripts/` | phase pipelines: extraction, dev/freeze, the single look |
| `tools/` | annotation tool, agreement, release/verify, environment and cross-device checks |
| `tests/` | `python tests/test_<name>.py` (no framework needed) |
| `audit/`, `phase1/`, `phase3/`, `phase4/` | reports, protocols, frozen configurations, keypoint caches, single-look markers |

`MANIFEST.sha256` lists the SHA-256 of every tracked file at the last manifest build. Frozen artefacts must never change; [MIGRATION.md](MIGRATION.md) says which files those are.

## Licence
No licence has been chosen yet, so all rights are reserved by the author.
