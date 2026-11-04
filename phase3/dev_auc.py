"""Dev-only exploratory diagnostic (NOT a result, test rows never used): per (asana, posture class, angle) cell, AUC of the raw angle value for Right vs Wrong (in-vocab) on DEV rows."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard import phase3 as P
S = P.load_stills(); d = S[S.split == "dev"].copy(); d["invocab"] = [(l == "Right") or bool(P.tag_families(f)[0]) for l, f in zip(d.label, d.fault)]; d = d[d.invocab]
rows = []
for (a, c), g in d.groupby(["asana", "pclass"]):
    for n in P.ANGLE_NAMES:
        v = g[["label", n]].dropna(); r, w = v[v.label == "Right"][n], v[v.label == "Wrong"][n]
        if len(r) >= 30 and len(w) >= 30: rows.append(dict(asana=a, pclass=c, angle=n, n_r=len(r), n_w=len(w), auc=roc_auc_score(np.r_[np.zeros(len(r)), np.ones(len(w))], np.r_[r, w]), iqr_right=np.subtract(*np.percentile(r, [75, 25]))))
R = pd.DataFrame(rows); R["strength"] = (R.auc - 0.5).abs()
print(f"cells with >=30 Right and >=30 Wrong dev values: {len(R)}")
print("median |AUC-0.5| by angle:"); print(R.groupby("angle").strength.median().round(3).sort_values(ascending=False).to_string())
print(f"\ncells with |AUC-0.5| > 0.15: {(R.strength > .15).sum()} / {len(R)} ({(R.strength > .15).mean():.2f}); > 0.25: {(R.strength > .25).sum()}")
print("strongest 8 cells:"); print(R.sort_values("strength", ascending=False).head(8).round(3).to_string(index=False))
print("median IQR of Right angle values per angle (deg):"); print(R.groupby("angle").iqr_right.median().round(0).to_string())
