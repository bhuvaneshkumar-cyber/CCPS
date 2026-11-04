"""Phase 0 follow-up: are some step folders byte-identical copies of earlier steps of the same asana/label?
(Found when the annotator's reference panel showed no stills for Tadasana steps 4-5.) Uses exact md5 groups from inventory_images.csv.
Reports, per asana x label x step: n images, and the fraction whose bytes also occur in a DIFFERENT step of the same asana+label (and in which)."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
d = pd.read_csv(ROOT / "audit" / "manifest_dedup.csv")
rows = []
for (a, lab), g in d.groupby(["asana", "label"]):
    if lab == "Wrong": g = g.assign(step=g.step)  # keep step index; fault tag ignored
    by_md5 = g.groupby("md5").step.agg(lambda s: sorted(set(s)))
    for k, gk in g.groupby("step"):
        other = [set(by_md5[m]) - {k} for m in gk.md5]
        dup = sum(bool(o) for o in other); into = sorted(set().union(*other)) if other else []
        rows.append(dict(asana=a, label=lab, step=k, n=len(gk), n_dup_of_other_step=dup, frac=round(dup / len(gk), 2), also_in_steps=into))
r = pd.DataFrame(rows); pd.set_option("display.width", 200, "display.max_rows", 200)
print(r[r.n_dup_of_other_step > 0].to_string(index=False))
print("\nasana/label combos with ANY cross-step duplication:", r[r.n_dup_of_other_step > 0].groupby(["asana", "label"]).ngroups, "of", r.groupby(["asana", "label"]).ngroups)
print("steps that are >=90% copies of another step:", int((r.frac >= 0.9).sum()), "of", len(r))
