"""Phase 0: burst density + right/wrong session overlap from camera-timestamp filenames (leakage analysis)."""
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parent.parent
d = pd.read_csv(ROOT / "audit" / "inventory_images.csv", dtype={"date": str, "time": str})
print("files without camera timestamp:", int(d.date.isna().sum()))
d["t"] = pd.to_datetime(d.date + d.time, format="%Y%m%d%H%M%S")
d["folder"] = d.path.map(lambda p: str(Path(p).parent))
gaps = np.concatenate([x.t.sort_values().diff().dt.total_seconds().dropna().values for _, x in d.groupby("folder")])
print(f"intra-folder consecutive gaps (s): median={np.median(gaps)} frac<=1s={(gaps <= 1).mean():.3f} frac<=2s={(gaps <= 2).mean():.3f} frac<=5s={(gaps <= 5).mean():.3f}")
print(d.groupby(["asana", "label"]).t.agg(["min", "max"]).to_string())
for a, x in d.groupby("asana"):
    r, w = x[x.label == "Right"].t, x[x.label == "Wrong"].t
    print(f"{a:22s} right/wrong time ranges", "OVERLAP" if (r.min() <= w.max() and w.min() <= r.max()) else "disjoint", "| dates:", sorted(x.date.unique()))
print("files with (n) duplicate-name suffix:", int(d.path.str.contains(r"\(\d\)\.jpg").sum()))
