"""Paired diagnostic on the KNOWN-transform validation frames (not a system result). null and squash images come from the SAME video frames
and are both 256 px wide, so x is comparable in pixels; y is scaled by 256/455 = 0.5625 in squash relative to null.
Perfect backbone => dx_sq/dx_null = 1, dy_sq/dy_null = 0.5625, k_eff = (dx ratio)/(dy ratio) = 1.778.
Per-bone medians over frames (block bootstrap over clips) show WHERE the backbone deviates from the applied geometry."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard.config import KPT_CONF_TAU as TAU, ROOT

z = np.load(ROOT / "audit" / "kpts_pseudo.npz"); a, b, vi = z["squash_kp"], z["null_kp"], z["squash_vi"]
assert (vi == z["null_vi"]).all() and len(a) == len(b)  # same frame order in both conditions
BONES = {"shoulder span (x)": (5, 6, 0), "hip span (x)": (11, 12, 0), "L thigh (y)": (11, 13, 1), "R thigh (y)": (12, 14, 1),
         "L shin (y)": (13, 15, 1), "R shin (y)": (14, 16, 1), "L torso (y)": (5, 11, 1), "R torso (y)": (6, 12, 1)}
MINLEN = {0: 8.0, 1: 20.0}  # px in null image; drop near-degenerate (foreshortened) bones where the ratio is noise
rng = np.random.default_rng(0); GEO = 256 / round(256 * 1280 / 720)  # y scale squash/null = 0.5625
res = {}
for n, (i, j, ax) in BONES.items():
    m = (np.nan_to_num(a[:, [i, j], 2]) >= TAU).all(1) & (np.nan_to_num(b[:, [i, j], 2]) >= TAU).all(1)
    da, db = np.abs(a[:, i, ax] - a[:, j, ax]), np.abs(b[:, i, ax] - b[:, j, ax]); m &= db > MINLEN[ax]
    r = da[m] / db[m]; v = vi[m]; u = np.unique(v)
    bs = [np.median(r[np.isin(v, rng.choice(u, len(u)))]) for _ in range(300)]
    res[n] = (np.median(r), *np.percentile(bs, [2.5, 97.5]), m.sum())
    print(f"{n:18s} median ratio squash/null = {np.median(r):.3f} CI[{res[n][1]:.3f},{res[n][2]:.3f}] n={m.sum():4d}   ideal {'1.000' if ax == 0 else f'{GEO:.3f}'}")
xr = np.mean([res[k][0] for k in BONES if "(x)" in k]); yr = np.mean([res[k][0] for k in BONES if "(y)" in k])
print(f"\nmean x-ratio {xr:.3f} (ideal 1.000), mean y-ratio {yr:.3f} (ideal {GEO:.3f}) -> k_eff = {xr / yr:.2f} vs geometric 1.778")
