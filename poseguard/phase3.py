"""Phase 3 (stills stage): split, reference bands, decision rule, fault families, metrics. Everything here is specified in phase3/PROTOCOL.md
(frozen before any test result); pure functions except load_stills(). Vectorised judging is unit-tested against poseguard.rules.judge."""
import re
import numpy as np, pandas as pd
from poseguard import config as C
from poseguard.angles import ANGLES, compute_angles
from poseguard.rules import Reference

ASANAS = ["Anantasana", "Ardhakati Chakrasana", "Bhujangasana", "Kati Chakrasana", "Marjariasana", "Parvatasana", "Tadasana", "Vajrasana"]
ANGLE_NAMES = [a.name for a in ANGLES]
JOINT_FAMILY = {"knee": "legs", "hip": "legs", "elbow": "arms", "shoulder": "arms", "trunk_tilt": "trunk", "shoulder_line_tilt": "trunk"}
ANGLE_FAMILY = {a.name: JOINT_FAMILY[a.joint] for a in ANGLES}
GRID_P, GRID_DELTA = (1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0), (5.0, 10.0, 15.0, 20.0)   # p grid extended by amendment A1 (protocol sec. 7)
MIN_N, MIN_BLOCKS, MIN_JUDGED = 40, 3, 3
BLOCK_S, GUARD_S, GUARD_FRAC = 40.0, 4.0, 0.15

def posture_classes(man):
    """{(asana, step): representative step}. Steps of one asana are one posture if any byte-identical image (md5) occurs in both (union-find, min step = representative)."""
    out = {}
    for a, g in man.groupby("asana"):
        parent = {s: s for s in g.step.unique()}
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for _, gg in g.groupby(["label", "md5"]):
            s = sorted(gg.step.unique())
            for x in s[1:]: parent[find(x)] = find(s[0])
        for s in parent: out[(a, s)] = min(x for x in parent if find(x) == find(s))
    return out

def assign_blocks(df):
    """Adds `block` (unique id), `k` (block index in folder) and `split` in {dev, test, guard}. Deterministic; needs asana, label, step, t (datetime)."""
    parts = []
    for (a, l, s), g in df.groupby(["asana", "label", "step"], sort=False):
        t = (g.t - g.t.min()).dt.total_seconds().to_numpy(); span = float(t.max()); nb = int(np.clip(round(span / BLOCK_S), 3, 10)); L = span / nb if span > 0 else 1.0
        k = np.minimum(nb - 1, np.floor(t / L)).astype(int) if span > 0 else np.zeros(len(t), int)
        gm = min(GUARD_S, GUARD_FRAC * L); near = ((k > 0) & (t - k * L < gm)) | ((k < nb - 1) & ((k + 1) * L - t < gm))
        split = np.where(k % 3 == 2, "test", "dev").astype(object); split[near] = "guard"
        parts.append(g.assign(k=k, block=[f"{a}|{l}|{s}|{x}" for x in k], split=split))
    return pd.concat(parts)

def band(values, p, delta):
    return Reference(float(np.percentile(values, p)) - delta, float(np.percentile(values, 100 - p)) + delta)

def derive_refs(dev_right, p, delta):
    """{(asana, pclass, angle): Reference} from dev Right rows (columns asana, pclass, block + ANGLE_NAMES with NaN = unavailable)."""
    refs = {}
    for (a, c), g in dev_right.groupby(["asana", "pclass"]):
        for n in ANGLE_NAMES:
            v = g[n].dropna()
            if len(v) >= MIN_N and g.loc[v.index, "block"].nunique() >= MIN_BLOCKS: refs[(a, c, n)] = band(v.to_numpy(), p, delta)
    return refs

def deviations(values, lo, hi):
    """values (n, A) with NaN = unavailable; lo/hi (A,) with NaN = no reference -> signed deviation beyond the band (0 inside, NaN = not judged). Same arithmetic as rules.judge."""
    d = np.where(values > hi, values - hi, np.where(values < lo, values - lo, 0.0))
    return np.where(np.isnan(values) | np.isnan(lo) | np.isnan(hi), np.nan, d)

def judge_frame(df, refs):
    """Row-wise decision for df (asana, pclass + angle columns). Returns DataFrame: status (correct/incorrect/abstain), n_judged, top_family (family of the largest |deviation|, incorrect only)."""
    st = pd.Series("abstain", index=df.index, dtype=object); nj = pd.Series(0, index=df.index); fam = pd.Series(None, index=df.index, dtype=object)
    for (a, c), g in df.groupby(["asana", "pclass"]):
        lo = np.array([refs[(a, c, n)].lo if (a, c, n) in refs else np.nan for n in ANGLE_NAMES]); hi = np.array([refs[(a, c, n)].hi if (a, c, n) in refs else np.nan for n in ANGLE_NAMES])
        D = deviations(g[ANGLE_NAMES].to_numpy(float), lo, hi); judged = ~np.isnan(D); n = judged.sum(1); bad = (np.nan_to_num(D) != 0).any(1)
        s = np.where(n < MIN_JUDGED, "abstain", np.where(bad, "incorrect", "correct")); st[g.index], nj[g.index] = s, n
        arg = np.nanargmax(np.abs(np.where(judged, D, -1.0)), axis=1); fam[g.index] = np.where(s == "incorrect", [ANGLE_FAMILY[ANGLE_NAMES[i]] for i in arg], None)
    return pd.DataFrame({"status": st, "n_judged": nj, "top_family": fam})

def tag_families(tag):
    """Fault tag text (folder name after 'Step k-') -> (set of families, has out-of-vocabulary part)."""
    t = str(tag).lower(); fams = set()
    if "leg" in t or "knee" in t: fams.add("legs")
    if "hand" in t or "palm" in t: fams.add("arms")
    if "back" in t: fams.add("trunk")
    return fams, any(w in t for w in ("neck", "head", "ankle", "standing"))

def prf(y_true, y_pred):
    """Positive class = incorrect (1). Precision/recall NaN when undefined; F1 = 2tp/(2tp+fp+fn) (NaN if no positives anywhere)."""
    y, p = np.asarray(y_true, bool), np.asarray(y_pred, bool); tp, fp, fn, tn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum(), (~y & ~p).sum()
    f = lambda a, b: a / b if b else np.nan
    rec, spec = f(tp, tp + fn), f(tn, tn + fp)
    return dict(precision=f(tp, tp + fp), recall=rec, specificity=spec, bal_acc=(0.5 * (rec + spec) if not (np.isnan(rec) or np.isnan(spec)) else np.nan), f1=f(2 * tp, 2 * tp + fp + fn), accuracy=f(tp + tn, len(y)), n=int(len(y)), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

def boot_stat(df, stat, block_col="block", nb=2000, seed=0):
    """Point estimate + 95% percentile CI of stat(df_resample) resampling whole blocks with replacement (never single images)."""
    rng = np.random.default_rng(seed); groups = {b: g.index.to_numpy() for b, g in df.groupby(block_col)}; keys = np.array(list(groups)); vals = []
    for _ in range(nb):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))]); vals.append(stat(df.loc[idx]))
    return stat(df), tuple(np.nanpercentile(vals, [2.5, 97.5])) if not np.all(np.isnan(vals)) else (np.nan, np.nan)

def load_stills(k=None):
    """Usable headline stills + keypoints (phase3/kpts_stills_640.npz) + corrected angles + posture class + block/split."""
    man_all = pd.read_csv(C.ROOT / "audit" / "manifest_dedup.csv"); pc = posture_classes(man_all[~man_all.ambiguous])
    z = np.load(C.ROOT / "phase3" / "kpts_stills_640.npz", allow_pickle=True); n = int(z["done"]); k = C.STILL_STRETCH_K if k is None else k
    m = man_all.set_index("path").loc[z["path"][:n]].reset_index(); m["kp_i"] = np.arange(n)
    m["t"] = pd.to_datetime(m.date.astype(str).str.zfill(8) + m.time.astype(str).str.zfill(6), format="%Y%m%d%H%M%S")
    m["pclass"] = [pc[(a, s)] for a, s in zip(m.asana, m.step)]
    A = np.array([[v if v is not None else np.nan for v in compute_angles(kp, stretch_k=k).values()] for kp in z["kp"][:n]], float)
    return assign_blocks(pd.concat([m, pd.DataFrame(A, columns=ANGLE_NAMES)], axis=1))
