"""Phase 4 input (GPU machine): backbone keypoints on ALL 80 videos at EVERY frame (or --stride N), through poseguard.backbone at imgsz 640.
Resumable per clip; every save is atomic (temp file + rename), so an interrupted run never leaves a half-written cache.
The output file name carries the device (GPU results are stored next to, never over, CPU results): phase4/kpts_video_full_<cuda|cpu>.npz
(stride > 1: kpts_video_s<N>_<device>.npz). Layout = phase3/kpts_video_s6.npz: kp_<i> (n,17,3) NaN if nobody found, fr_<i> frame indices, paths.
usage: python scripts/extract_video_full.py [--device auto|cuda:0|cpu] [--stride 1]"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poseguard import config as C
from poseguard.backbone import Backbone, primary

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="auto"); ap.add_argument("--stride", type=int, default=1); a = ap.parse_args()
bb = Backbone(device=a.device); tag = "cuda" if str(bb.device).startswith("cuda") else "cpu"
OUT = ROOT / "phase4"; OUT.mkdir(exist_ok=True); f = OUT / (f"kpts_video_full_{tag}.npz" if a.stride == 1 else f"kpts_video_s{a.stride}_{tag}.npz")
vid = pd.read_csv(ROOT / "audit" / "inventory_videos.csv"); keys = vid.path.str.replace("\\", "/").to_numpy()
store = dict(np.load(f, allow_pickle=True)) if f.exists() else {}
if store and list(store["paths"]) != list(keys): sys.exit(f"{f} was made for a different clip list; refusing to mix")
print(f"device {bb.device} -> {f.name} | {len(keys)} clips, {int(vid.frames.sum() // a.stride)} frames | done already: {sum(k.startswith('kp_') for k in store)}", flush=True)
for i, k in enumerate(keys):
    if f"kp_{i}" in store: continue
    t = time.time(); kps, frs = [], []
    for fi, _, ps in bb.video(str(C.media(k)), stride=a.stride):
        p = primary(ps); kps.append(p.kp if p is not None else np.full((17, 3), np.nan, np.float32)); frs.append(fi)
    if abs(len(frs) - vid.frames[i] / a.stride) > 2: print(f"  WARNING clip {i}: decoded {len(frs)} frames, inventory says {vid.frames[i]} (reported, not fixed)")
    store[f"kp_{i}"], store[f"fr_{i}"], store["paths"] = np.array(kps, np.float32), np.array(frs), keys
    tmp = f.with_name(f.stem + ".tmp.npz"); np.savez(tmp, **store); os.replace(tmp, f)
    print(f"{i:2d} {k.split('/')[-1][:55]:55s} {len(frs):5d} frames  {len(frs) / (time.time() - t):5.1f} fps  no-det {int(np.isnan(np.array(kps)[:, 0, 0]).sum())}", flush=True)
print("done:", f)
