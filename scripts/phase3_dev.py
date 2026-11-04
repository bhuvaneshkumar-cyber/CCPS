"""Phase 3 dev stage (stills). Uses DEV blocks only: test rows are dropped before anything is computed. Selects (p, delta) by 5-fold GroupKFold (groups = blocks)
per phase3/PROTOCOL.md, writes the actual reference ranges, and FREEZES phase3/frozen_config.json (with the sha256 of the protocol and of poseguard/phase3.py).
usage: python scripts/phase3_dev.py"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poseguard import config as C
from poseguard import phase3 as P

sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
zz = np.load(ROOT / "phase3" / "kpts_stills_640.npz", allow_pickle=True); COMPLETE = int(zz["done"]) == len(zz["path"])
if not COMPLETE: print(f"*** PARTIAL KEYPOINTS ({int(zz['done'])}/{len(zz['path'])}): DRY RUN - nothing will be written or frozen ***")
S = P.load_stills(); print("stills loaded:", len(S), "| split counts:", S.split.value_counts().to_dict())
print(S.pivot_table(index="asana", columns="split", values="path", aggfunc="count", fill_value=0).to_string())
print("blocks per split:", S.groupby("split").block.nunique().to_dict())
dev = S[S.split == "dev"].copy()                                   # test rows are never used below this line
dev["y"] = (dev.label == "Wrong").astype(int); dev["invocab"] = [(l == "Right") or bool(P.tag_families(f)[0]) for l, f in zip(dev.label, dev.fault)]
ev = dev[dev.invocab].reset_index(drop=True); print("dev evaluation rows:", len(ev), "| Right:", int((ev.y == 0).sum()), "Wrong in-vocab:", int((ev.y == 1).sum()), "| dev Wrong out-of-vocab dropped:", int((~dev.invocab).sum()))

def macro(df):
    sc = df[df.status != "abstain"]; rows = [P.prf(g.y, g.status == "incorrect") for _, g in sc.groupby("asana")]
    return dict(ba=np.nanmean([r["bal_acc"] for r in rows]), f1=np.nanmean([r["f1"] for r in rows]), precision=np.nanmean([r["precision"] for r in rows]), recall=np.nanmean([r["recall"] for r in rows]), abstain=float((df.status == "abstain").mean()))

grid = []
for p in P.GRID_P:
    for dl in P.GRID_DELTA:
        st = pd.Series("abstain", index=ev.index, dtype=object)
        for tr, va in GroupKFold(n_splits=5).split(ev, groups=ev.block):
            train = ev.iloc[tr]; refs = P.derive_refs(train[train.label == "Right"], p, dl)
            st.iloc[va] = P.judge_frame(ev.iloc[va], refs).status.values
        m = macro(ev.assign(status=st)); grid.append(dict(p=p, delta=dl, **m)); print(f"p={p:>3} delta={dl:>4}: BA={m['ba']:.3f} macroF1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} abstain={m['abstain']:.3f}", flush=True)
triv = np.nanmean([2 * g.y.mean() / (1 + g.y.mean()) for _, g in ev.groupby("asana")]); print(f"baseline: always-incorrect macro-F1 = {triv:.3f}, balanced accuracy = 0.500 (any constant predictor)")
G = pd.DataFrame(grid)
if COMPLETE: G.to_csv(ROOT / "phase3" / "dev_grid.csv", index=False)
best = G.sort_values(["ba", "f1", "delta"], ascending=False).iloc[0]; p, dl = float(best.p), float(best.delta); print("\nSELECTED (dev CV only): p =", p, " delta =", dl, f" macroF1 = {best.f1:.3f}")

if not COMPLETE: sys.exit("dry run finished (partial data); nothing frozen")
refs = P.derive_refs(ev[ev.label == "Right"], p, dl)
R = pd.DataFrame([dict(asana=a, posture_class=c, angle=n, lo=r.lo, hi=r.hi, width=r.hi - r.lo) for (a, c, n), r in refs.items()])
R.to_csv(ROOT / "phase3" / "reference_ranges_stills.csv", index=False)
print(f"\nreference ranges derived: {len(R)} (asana, class, angle) bands of {len(ev.groupby(['asana', 'pclass'])) * len(P.ANGLE_NAMES)} possible; median width by angle (deg):")
print(R.groupby("angle").width.median().round(0).to_string())
json.dump(dict(p=p, delta=dl, K=C.STILL_STRETCH_K, tau=C.KPT_CONF_TAU, min_n=P.MIN_N, min_blocks=P.MIN_BLOCKS, min_judged=P.MIN_JUDGED, dev_cv_bal_acc=float(best.ba), dev_cv_macro_f1=float(best.f1), first_run_f1_selected=dict(p=5.0, delta=5.0),
               protocol_sha256=sha(ROOT / "phase3" / "PROTOCOL.md"), phase3_py_sha256=sha(ROOT / "poseguard" / "phase3.py"), frozen_at=time.strftime("%Y-%m-%d %H:%M:%S")),
          open(ROOT / "phase3" / "frozen_config.json", "w"), indent=1)
print("frozen_config.json written. The test split has not been touched.")
