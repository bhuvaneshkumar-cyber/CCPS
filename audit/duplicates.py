"""Phase 0: exact-duplicate detection (md5 of file bytes) within/across classes and Right/Wrong labels."""
import hashlib, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard.config import DATA_DIR, ROOT

B = DATA_DIR / "raw" / "Yoga Postures Dataset"
f = ROOT / "audit" / "inventory_images.csv"
d = pd.read_csv(f)
d["md5"] = [hashlib.md5((B / p).read_bytes()).hexdigest() for p in d.path]
d["name"] = d.path.map(lambda p: Path(p.replace("\\", "/")).name)
d.to_csv(f, index=False)
dup = d[d.duplicated("md5", keep=False)]
print(f"exact-duplicate files: {len(dup)} in {dup.md5.nunique()} groups")
g = dup.groupby("md5").agg(n=("path", "size"), asanas=("asana", lambda s: sorted(set(s))),
                           labels=("label", lambda s: sorted(set(s))), steps=("path", lambda s: sorted(set(x.replace("\\", "/").split("/")[-2] for x in s))))
print("groups spanning >1 asana:", int((g.asanas.str.len() > 1).sum()), "| spanning both Right and Wrong:", int((g.labels.str.len() > 1).sum()))
print(g[(g.asanas.str.len() > 1) | (g.labels.str.len() > 1)].head(10).to_string())
print("groups within a single folder:", int((g.steps.str.len() == 1).sum()))
print("filenames that occur in >1 asana:", int((d.groupby("name").asana.nunique() > 1).sum()))
