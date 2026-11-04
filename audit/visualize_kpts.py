"""Phase 0/1: overlay backbone keypoints on real dataset images, one sheet per asana (8 Right + 8 Wrong, fixed seed).
green = kpt conf >= tau, red = below tau (would be gated out); 'NO DET' = backbone found no person.
usage: python audit/visualize_kpts.py kpts_640_sample.npz"""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard.config import DATA_DIR, KPT_CONF_TAU, ROOT

B = DATA_DIR / "raw" / "Yoga Postures Dataset"
z = np.load(ROOT / "audit" / sys.argv[1], allow_pickle=True)
d = pd.DataFrame({"path": z["path"]}).merge(pd.read_csv(ROOT / "audit" / "manifest_dedup.csv")[["path", "asana", "label", "usable"]], on="path")
SKEL = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
out = ROOT / "audit" / "kpt_viz"; out.mkdir(exist_ok=True)
S = 384  # 256 -> 384 for legibility
for a, g in d.groupby("asana"):
    tiles = []
    for lab in ("Right", "Wrong"):
        gg = g[(g.label == lab) & g.usable]
        for i in gg.index[:8]:
            im = cv2.resize(cv2.imread(str(B / d.path[i])), (S, S)); k = z["kp"][list(z["path"]).index(d.path[i])].copy()
            if np.isnan(k).all():
                cv2.putText(im, "NO DET", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                k[:, :2] *= S / 256  # ultralytics returns keypoints in original-image pixels (256) -> scale to tile
                for u, v in SKEL:
                    if k[u, 2] >= KPT_CONF_TAU and k[v, 2] >= KPT_CONF_TAU:
                        cv2.line(im, tuple(k[u, :2].astype(int)), tuple(k[v, :2].astype(int)), (255, 200, 0), 2)
                for x, y, c in k:
                    cv2.circle(im, (int(x), int(y)), 4, (0, 255, 0) if c >= KPT_CONF_TAU else (0, 0, 255), -1)
            cv2.putText(im, lab, (6, S - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            tiles.append(im)
        while len(tiles) % 8: tiles.append(np.zeros((S, S, 3), np.uint8))  # keep Right/Wrong on separate rows
    sheet = np.vstack([np.hstack(tiles[r:r + 8]) for r in range(0, len(tiles), 8)])
    cv2.imwrite(str(out / f"{a.replace(' ', '_')}.jpg"), cv2.resize(sheet, None, fx=0.5, fy=0.5))
    print("wrote", a, len(tiles))
