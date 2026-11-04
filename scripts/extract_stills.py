"""Phase 3 input: backbone keypoints for every USABLE (deduped) still of the 8 headline asanas, through poseguard.backbone at the deployed imgsz (640).
Sarvangasana and Viparita Karani are excluded (stills-based evaluation excluded by spec). Checkpoints every 500 images; resumable.
Output: phase3/kpts_stills_640.npz  (path, kp[N,17,3] NaN if nobody found, npers, box conf)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poseguard import config as C
from poseguard.backbone import Backbone, primary

OUT = ROOT / "phase3"; OUT.mkdir(exist_ok=True); f = OUT / "kpts_stills_640.npz"
man = pd.read_csv(ROOT / "audit" / "manifest_dedup.csv"); man = man[man.usable & ~man.asana.isin(["Sarvangasana", "Viparita Karani"])].reset_index(drop=True)
bb = Backbone(device="cpu"); N = len(man); kp = np.full((N, 17, 3), np.nan, np.float32); npers = np.zeros(N, np.int16); bconf = np.zeros(N, np.float32); start = 0
if f.exists():
    z = np.load(f, allow_pickle=True)
    if len(z["path"]) == N and (z["path"] == man.path.values).all(): kp, npers, bconf, start = z["kp"], z["npers"], z["bconf"], int(z["done"])
print("total", N, "resuming from", start, flush=True)
for i in range(start, N):
    ps = bb(cv2.imread(str(C.DATA_DIR / "raw" / "Yoga Postures Dataset" / man.path[i]))); p = primary(ps); npers[i] = len(ps)
    if p is not None: kp[i], bconf[i] = p.kp, p.conf
    if (i + 1) % 500 == 0 or i == N - 1:
        np.savez(f, path=man.path.values, kp=kp, npers=npers, bconf=bconf, done=i + 1); print(i + 1, "/", N, flush=True)
print("done; no-detection:", int((npers == 0).sum()))
