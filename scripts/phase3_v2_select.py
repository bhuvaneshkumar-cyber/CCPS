"""geo_v2_lightly_supervised: run the PRE-REGISTERED selection (phase3/PROTOCOL_V2.md) on DEV stills only, run the label-permutation gate, and freeze or stop.
Never reads test rows or video. usage: python scripts/phase3_v2_select.py"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poseguard import config as C
from poseguard import geo_v2 as G
from poseguard import phase3 as P

sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest(); OUT = ROOT / "phase3" / "v2"; OUT.mkdir(exist_ok=True)
if (OUT / "frozen_selection.json").exists() or (OUT / "STOPPED.json").exists(): sys.exit("selection already run (frozen or stopped); it is never re-run - delete nothing, report what exists")
S = P.load_stills(); dev = S[S.split == "dev"].copy()
dev["fams"] = [frozenset(P.tag_families(f)[0]) if l == "Wrong" else frozenset() for l, f in zip(dev.label, dev.fault)]
dev = dev[(dev.label == "Right") | dev.fams.map(len).gt(0)].reset_index(drop=True)          # in-vocabulary Wrong only, as in v1
print(f"dev rows for selection: {len(dev)} (Right {int((dev.label == 'Right').sum())}, in-vocab Wrong {int((dev.label == 'Wrong').sum())}); cells: {dev.groupby(['asana', 'pclass']).ngroups * len(P.ANGLE_NAMES)}")
t0 = time.time(); obs = G.select_cells(dev, np.random.default_rng(0)); n_obs = int(obs.qualifies.sum())
print(f"observed: Q1 {int(obs.q1.sum())} | Q1&Q2 {int((obs.q1 & obs.q2).sum())} | Q1&Q2&Q4 {int((obs.q1 & obs.q2 & obs.q4).sum())} | qualifying (all four) {n_obs}   [{time.time() - t0:.0f}s]")
null = G.permutation_null(dev, n_perm=200, seed=1); n0 = float(np.mean(null))
print(f"permutation null (200 label-permuted copies of dev, whole selection re-run): mean {n0:.2f}, median {np.median(null):.0f}, 95th pct {np.percentile(null, 95):.0f}, max {max(null)}")
obs.to_csv(OUT / "selection_table.csv", index=False); passes = n_obs >= 6 and n_obs >= 2 * n0
print(f"\nGATE: N_obs = {n_obs} >= 6 and >= 2 x N0 ({2 * n0:.2f})?  -> {'PASS' if passes else 'FAIL'}")
meta = dict(n_obs=n_obs, n0_mean=n0, n0_percentiles=[float(x) for x in np.percentile(null, [50, 95, 100])], protocol_sha256=sha(ROOT / "phase3" / "PROTOCOL_V2.md"), geo_v2_py_sha256=sha(ROOT / "poseguard" / "geo_v2.py"),
            phase3_py_sha256=sha(ROOT / "poseguard" / "phase3.py"), name=C.GEO_V2, at=time.strftime("%Y-%m-%d %H:%M:%S"))
if not passes:
    json.dump(dict(**meta, verdict="v2 unsupported by dev evidence; stopped per protocol section 3"), open(OUT / "STOPPED.json", "w"), indent=1); print("STOPPED: nothing frozen, no video run.")
else:
    sel = obs[obs.qualifies][["asana", "pclass", "angle", "auc", "n_right", "n_wrong", "blocks_right", "blocks_wrong"]]
    json.dump(dict(**meta, selected=[[a, int(c), n] for a, c, n in zip(sel.asana, sel.pclass, sel.angle)]), open(OUT / "frozen_selection.json", "w"), indent=1)
    print("FROZEN. selected cells:"); print(sel.round(3).to_string(index=False)); print("\nselected cells per asana:", sel.groupby("asana").size().to_dict(), "| the video and stills-test sets were not touched")
