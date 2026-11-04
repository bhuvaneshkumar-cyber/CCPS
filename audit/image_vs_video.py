"""Phase 0: are the still images frames of the videos? Tadasana probe: 32x32 gray thumbnails (aspect ignored) of
every 3rd video frame vs every Tadasana image; report nearest-neighbour L1 distance image->video, compared with
image->image nearest distance (same-session burst neighbours) as the reference scale."""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard.config import DATA_DIR, ROOT

B = DATA_DIR / "raw" / "Yoga Postures Dataset"
th = lambda im: cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
inv = pd.read_csv(ROOT / "audit" / "manifest_dedup.csv"); inv = inv[(inv.asana == "Tadasana") & inv.usable]
I = np.stack([th(cv2.imread(str(B / p))) for p in inv.path])
V = []
for p in sorted((B / "Videos" / "Tadasana").rglob("*.mp4")):
    cap = cv2.VideoCapture(str(p)); k = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if k % 3 == 0: V.append(th(f))
        k += 1
V = np.stack(V); print("images", len(I), "video thumbs", len(V))
dv = np.array([np.abs(V - i).mean(1).min() for i in I])
D = np.abs(I[:, None] - I[None]).mean(2); np.fill_diagonal(D, 1e9); di = D.min(1)
q = lambda a: np.percentile(a, [5, 25, 50, 75, 95]).round(2)
print("image->nearest video frame  L1 pct[5,25,50,75,95]:", q(dv))
print("image->nearest OTHER image  L1 pct[5,25,50,75,95]:", q(di))
