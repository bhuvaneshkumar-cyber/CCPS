"""Phase 0: per-class backbone reliability from REAL detections on the usable (deduped) images.
- detection rate, multi-person rate
- per-keypoint coverage: fraction of usable images where the keypoint conf >= tau (denominator = ALL usable images,
  so a missed detection counts as 'unavailable', matching the confidence-gating semantics)
- agreement of imgsz=320 (full population) with imgsz=640 (400-image stratified sample)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard.config import KPT_CONF_TAU as TAU, ROOT

NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri",
         "l_hip", "r_hip", "l_kne", "r_kne", "l_ank", "r_ank"]
man = pd.read_csv(ROOT / "audit" / "manifest_dedup.csv")
pd.set_option("display.width", 250, "display.max_columns", 40)

def load(f):
    z = np.load(ROOT / "audit" / f, allow_pickle=True)
    return pd.DataFrame({"path": z["path"]}).merge(man[["path", "asana", "label", "usable"]], on="path"), z["kp"], z["npers"]

d, kp, npers = load("kpts_320_all.npz")
u = d.usable.values; d, kp, npers = d[u].reset_index(drop=True), kp[u], npers[u]
conf = np.nan_to_num(kp[:, :, 2], nan=0.0)
ok = conf >= TAU
d["detected"] = npers > 0; d["multi"] = npers > 1
print(f"imgsz=320, usable images={len(d)}  tau={TAU}")
print("\nDETECTION rate / multi-person rate per asana x label")
print(d.groupby(["asana", "label"]).agg(n=("path", "size"), det=("detected", "mean"), multi=("multi", "mean")).round(3).to_string())
cov = pd.DataFrame(ok, columns=NAMES); cov["asana"] = d.asana; cov["label"] = d.label
print("\nPER-KEYPOINT COVERAGE (conf>=tau), per asana (all usable images)")
print(cov.groupby("asana")[NAMES].mean().round(2).to_string())
lim = {"L_arm(sho,elb,wri)": [5, 7, 9], "R_arm": [6, 8, 10], "L_leg(hip,kne,ank)": [11, 13, 15], "R_leg": [12, 14, 16], "trunk(sho+hip x4)": [5, 6, 11, 12]}
print("\nJOINT-CHAIN COVERAGE: all constituent keypoints >= tau")
c2 = pd.DataFrame({k: ok[:, v].all(1) for k, v in lim.items()}); c2["asana"] = d.asana
print(c2.groupby("asana").mean().round(2).to_string())
print("\noverall chain coverage:", c2.drop(columns="asana").mean().round(3).to_dict())

# 320 vs 640 agreement on the sample (same images)
d6, kp6, np6 = load("kpts_640_sample.npz")
d3, kp3, np3 = load("kpts_320_all.npz")
i3 = {p: i for i, p in enumerate(d3.path)}; idx = [i3[p] for p in d6.path]
a, b = kp3[idx], kp6
both = ~np.isnan(a[:, 0, 0]) & ~np.isnan(b[:, 0, 0])
dist = np.linalg.norm(a[:, :, :2] - b[:, :, :2], axis=2); m = (a[:, :, 2] >= TAU) & (b[:, :, 2] >= TAU) & both[:, None]
print(f"\n320 vs 640 on {len(idx)} sample images: detection agree={((np3[idx] > 0) == (np6 > 0)).mean():.3f}"
      f" | det@320={(np3[idx] > 0).mean():.3f} det@640={(np6 > 0).mean():.3f}"
      f" | median kpt disagreement (px, conf>=tau in both)={np.median(dist[m]):.2f}, p95={np.percentile(dist[m], 95):.2f}")
