"""Phase 0: dedupe policy. An md5 group is USABLE only if all its members agree on (asana, label);
then one representative is kept. Groups whose members span >1 asana or both Right and Wrong are
label-ambiguous and excluded from training AND evaluation. Writes audit/manifest_dedup.csv."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
d = pd.read_csv(ROOT / "audit" / "inventory_images.csv")
g = d.groupby("md5")
d["n_asana"] = g.asana.transform("nunique"); d["n_label"] = g.label.transform("nunique"); d["grp"] = g.path.transform("size")
d["ambiguous"] = (d.n_asana > 1) | (d.n_label > 1)
d["rep"] = ~d.duplicated("md5")  # first occurrence within group
d["usable"] = ~d.ambiguous & d.rep
d.to_csv(ROOT / "audit" / "manifest_dedup.csv", index=False)
print("raw", len(d), "| ambiguous (cross-asana or Right/Wrong-conflicting):", int(d.ambiguous.sum()),
      "| redundant same-label copies dropped:", int((~d.ambiguous & ~d.rep).sum()), "| usable:", int(d.usable.sum()))
t = d.pivot_table(index="asana", columns="label", values="path", aggfunc="count", margins=True).rename(columns={"Right": "raw_Right", "Wrong": "raw_Wrong", "All": "raw_All"})
u = d[d.usable].pivot_table(index="asana", columns="label", values="path", aggfunc="count", margins=True).rename(columns={"Right": "usable_Right", "Wrong": "usable_Wrong", "All": "usable_All"})
print(pd.concat([t, u], axis=1).to_string())
