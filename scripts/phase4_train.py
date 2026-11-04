"""Phase 4 dev stage: train temporal_bilstm_att_v1 on RIGHT clips only (phase4/PROTOCOL_PHASE4.md sec. 2-4): arm CHRONO (1 model) + arm LOAO (4 folds).
Early stopping and sigma use Right validation windows only; Wrong clips are never loaded. Freezes phase4/frozen_temporal_<device>.json + model_<arm>_s<seed>_<device>.pt (5-seed ensemble per arm).
Refuses to run while the protocol is a DRAFT, and never retrains over an existing freeze.
usage: python scripts/phase4_train.py [--device auto|cuda:0|cpu] [--kpts phase4/kpts_video_full_cuda.npz] [--smoke]
--smoke: plumbing check on phase3/kpts_video_s6.npz (12 Right clips, 5 fps), 2 seeds x 3 epochs, writes only to phase4/smoke/ - NOT RESULTS (delete before bundling)."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poseguard import temporal as T
from poseguard.backbone import pick_device

sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest(); PH = ROOT / "phase4"; PROT = PH / "PROTOCOL_PHASE4.md"
ap = argparse.ArgumentParser(); ap.add_argument("--device", default="auto"); ap.add_argument("--kpts", default=str(PH / "kpts_video_full_cuda.npz")); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
if not a.smoke and "STATUS: DRAFT" in PROT.read_text(encoding="utf-8"): sys.exit("phase4/PROTOCOL_PHASE4.md is still a DRAFT: the owner must resolve its DECISION items and freeze it first (see its first lines)")
kpts = ROOT / "phase3" / "kpts_video_s6.npz" if a.smoke else Path(a.kpts).resolve(); out = PH / "smoke" if a.smoke else PH; out.mkdir(exist_ok=True)
if not kpts.exists(): sys.exit(f"{kpts} missing: run scripts/extract_video_full.py first")
device = pick_device(a.device); tag = "cuda" if device.startswith("cuda") else "cpu"; frozen = out / f"frozen_temporal_{tag}.json"
if frozen.exists() and not a.smoke: sys.exit(f"{frozen.name} exists: the model was already trained and frozen; it is never retrained - report what exists")
if a.smoke: print("*** SMOKE RUN: stride-6 cache, 3 epochs, output in phase4/smoke/ - NOT RESULTS ***")

R = T.load_clips(kpts, pd.read_csv(ROOT / "audit" / "inventory_videos.csv"), labels=("Right",))
print(f"device {device} | Right clips {len(R)} | 10-fps steps {sum(len(c['ang']) for c in R)} | kpts {kpts.name}")
arms = {"chrono": (R, T.chrono_bounds), **{f"loao{k}": ([c for c in R if c["angle"] != k], T.loao_bounds) for k in (1, 2, 3, 4)}}
meta = {}
for arm, (clips, bounds) in arms.items():
    tr, va = T.collect(clips, bounds, "train"), T.collect(clips, bounds, "val"); print(f"\n[{arm}] train windows {len(tr['t'])}, val windows {len(va['t'])}, clips {len(clips)}", flush=True)
    t0 = time.time(); models, infos, files = [], [], []
    for sd in T.SEEDS[:2] if a.smoke else T.SEEDS:
        m, info = T.fit(tr, va, device, max_epochs=3 if a.smoke else T.MAX_EPOCHS, seed=sd); f = out / f"model_{arm}_s{sd}_{tag}.pt"; torch.save(m.state_dict(), f)
        models.append(m); infos.append(dict(info, seed=sd)); files.append(dict(file=f.name, sha256=sha(f)))
        print(f"[{arm}] seed {sd}: best epoch {info['best_epoch']} val MSE {info['best_val_mse']:.5f}", flush=True)
    em, j = T.errors(T.predict(models, va["X"], device), va); ep, _ = T.errors(va["last"], va)
    meta[arm] = dict(seeds=infos, models=files, seconds=round(time.time() - t0, 1), sigma_model=T.sigma(em, j, va["a"]).tolist(), sigma_persist=T.sigma(ep, j, va["a"]).tolist(), clips=[c["key"] for c in clips])
    print(f"[{arm}] ensemble of {len(models)} | Right val mean |err| deg: model {np.nanmean(em):.2f}, persistence {np.nanmean(ep):.2f}")
json.dump(dict(method=T.METHOD, device=tag, gpu=torch.cuda.get_device_name(0) if tag == "cuda" else None, torch=torch.__version__, smoke=a.smoke, kpts=kpts.relative_to(ROOT).as_posix(), kpts_sha256=sha(kpts),
               protocol_sha256=sha(PROT), temporal_py_sha256=sha(ROOT / "poseguard" / "temporal.py"), constants=dict(W=T.W, H=T.H, FPS=T.FPS, HIDDEN=T.HIDDEN, LR=T.LR, BATCH=T.BATCH, SEEDS=list(T.SEEDS), PATIENCE=T.PATIENCE, Z_CAP=T.Z_CAP, MIN_SIGMA_N=T.MIN_SIGMA_N, UNIT_S=T.UNIT_S),
               arms=meta, frozen_at=time.strftime("%Y-%m-%d %H:%M:%S")), open(frozen, "w"), indent=1)
print(f"\nfrozen: {frozen.relative_to(ROOT)}. No Wrong clip was read.")
