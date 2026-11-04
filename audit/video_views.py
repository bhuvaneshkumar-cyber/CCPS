"""Phase 0 correction: classify the ACTUAL camera view of each of the 80 videos from keypoints, instead of trusting the 'Angle n' number.
Per video (median over its 40 frames): shoulder span / torso length (large => subject faces or turns its back to the camera, small => profile) and mean
confidence of nose+eyes (high => face visible => FRONT, low => BACK). Only meaningful for upright asanas; floor/inverted asanas are listed but flagged 'n/a'.
Rules (frozen before looking at the table): profile if ratio < 0.30; else front if face conf >= 0.5 else back."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poseguard.config import KPT_CONF_TAU as TAU

UPRIGHT = ("Tadasana", "Ardhakati Chakrasana", "Kati Chakrasana")
vid = pd.read_csv(ROOT / "audit" / "inventory_videos.csv"); kp = np.load(ROOT / "audit" / "kpts_video_640.npz", allow_pickle=True)["kp"]
rows = []
for i, r in vid.iterrows():
    k = kp[i]; m = np.nan_to_num(k[:, [5, 6, 11, 12], 2]).min(1) >= TAU; k = k[m]
    if not len(k): rows.append((r.asana, r.label, r.angle, np.nan, np.nan, "n/a")); continue
    ratio = np.median(np.linalg.norm(k[:, 5, :2] - k[:, 6, :2], axis=1) / np.linalg.norm(k[:, [5, 6], :2].mean(1) - k[:, [11, 12], :2].mean(1), axis=1))
    face = np.median(np.nan_to_num(k[:, 0:3, 2]).mean(1)); view = "profile" if ratio < 0.30 else ("front" if face >= 0.5 else "back")
    rows.append((r.asana, r.label, r.angle, round(ratio, 2), round(face, 2), view if r.asana in UPRIGHT else f"({view}) n/a-not-upright"))
d = pd.DataFrame(rows, columns=["asana", "label", "angle", "span/torso", "face_conf", "view"]); pd.set_option("display.width", 200, "display.max_rows", 200)
u = d[d.asana.isin(UPRIGHT)]
print(u.pivot_table(index=["asana", "label"], columns="angle", values="view", aggfunc="first").to_string())
print("\nview by angle number, RIGHT clips:", u[u.label == "Right"].groupby("angle").view.agg(lambda s: dict(s.value_counts())).to_dict())
print("view by angle number, WRONG clips:", u[u.label == "Wrong"].groupby("angle").view.agg(lambda s: dict(s.value_counts())).to_dict())
print("\nfull table:\n", d.to_string(index=False))
