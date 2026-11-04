"""Keypoints on the 24 step-annotated videos at stride 6 (5 fps) through poseguard.backbone at imgsz 640. Right clips first. Resumable per clip.
Stride 6 is a CPU-time compromise (full 30 fps is for the RTX 5070); holds last >= 4 s so >= 20 frames per kept segment.
Output: phase3/kpts_video_s6.npz  -> for clip i: kp_<i> (n,17,3) NaN if nobody found, fr_<i> (n,) frame indices, plus `paths`."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poseguard import config as C
from poseguard.backbone import Backbone, primary

STRIDE = 6; f = ROOT / "phase3" / "kpts_video_s6.npz"
q = [l.strip() for l in open(ROOT / "audit" / "step_queue_phase3.txt") if l.strip() and not l.startswith("#")]
vid = pd.read_csv(ROOT / "audit" / "inventory_videos.csv"); vid["key"] = vid.path.str.replace("\\", "/")
print("clips:", len(q), "| frames to run:", int(sum(vid[vid.key == k].frames.iloc[0] // STRIDE for k in q)), flush=True)
store = dict(np.load(f, allow_pickle=True)) if f.exists() else {}; bb = Backbone(device="cpu")
for i, k in enumerate(q):
    if f"kp_{i}" in store: continue
    cap = cv2.VideoCapture(str(C.DATA_DIR / "raw" / "Yoga Postures Dataset" / k)); kps, frs, n = [], [], 0
    while True:
        if n % STRIDE:
            if not cap.grab(): break
        else:
            ok, fr = cap.read()
            if not ok: break
            p = primary(bb(fr)); kps.append(p.kp if p is not None else np.full((17, 3), np.nan, np.float32)); frs.append(n)
        n += 1
    store[f"kp_{i}"], store[f"fr_{i}"], store["paths"] = np.array(kps, np.float32), np.array(frs), np.array(q)
    np.savez(f, **store); print(i, k.split("/")[-1][:50], len(frs), "frames", flush=True)
print("done")
