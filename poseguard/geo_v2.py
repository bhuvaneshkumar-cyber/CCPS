"""geo_v2_lightly_supervised (config.GEO_V2): informative-angle selection per (asana, posture class), exactly as written in phase3/PROTOCOL_V2.md.
NOT 'the geometric rule engine' - v1 (config.GEO_V1) judges every angle with zero label supervision; v2 uses dev Wrong labels to choose WHICH angles are judged.
The numeric bands still come from correct examples only (phase3.derive_refs / video Right clips). All thresholds below are pre-registered constants."""
import numpy as np, pandas as pd
from scipy.stats import rankdata
from poseguard import phase3 as P

Q1_MIN_N, Q1_MIN_BLOCKS = 60, 4          # per class (Right and Wrong)
Q2_EFFECT = 0.20                         # |AUC - 0.5|
Q3_BOUND, NBOOT = 0.60, 1000             # bootstrap: AUC 2.5th pct >= .60 (AUC>.5) or 97.5th pct <= .40 (AUC<.5)
Q4_FAMILY_FRAC = 0.30                    # share of the cell's Wrong images whose fault tags include the angle's family

def auc(right, wrong):
    """P(random Wrong value > random Right value) + 0.5 * P(tie) (Mann-Whitney U / (n_r n_w))."""
    r, w = np.asarray(right, float), np.asarray(wrong, float)
    ranks = rankdata(np.r_[r, w]); return float((ranks[len(r):].sum() - len(w) * (len(w) + 1) / 2) / (len(r) * len(w)))

def _by_block(g, col): return {b: x[col].dropna().to_numpy() for b, x in g.groupby("block")}

def _boot_auc_interval(rb, wb, rng, nboot):
    rk, wk = np.array(list(rb)), np.array(list(wb)); vals = []
    for _ in range(nboot):
        r = np.concatenate([rb[k] for k in rng.choice(rk, len(rk))]); w = np.concatenate([wb[k] for k in rng.choice(wk, len(wk))])
        if len(r) and len(w): vals.append(auc(r, w))
    return np.percentile(vals, [2.5, 97.5])

def evaluate_cell(g, angle, rng, nboot=NBOOT):
    """g: dev rows of one (asana, posture class) with columns label, block, fams (set) and the angle column. Returns dict with the four checks."""
    r, w = g[g.label == "Right"], g[g.label == "Wrong"]; rv, wv = r[angle].dropna(), w[angle].dropna()
    info = dict(n_right=len(rv), n_wrong=len(wv), blocks_right=r.loc[rv.index, "block"].nunique(), blocks_wrong=w.loc[wv.index, "block"].nunique(), auc=np.nan, q1=False, q2=False, q3=False, q4=False, qualifies=False)
    info["q1"] = len(rv) >= Q1_MIN_N and len(wv) >= Q1_MIN_N and info["blocks_right"] >= Q1_MIN_BLOCKS and info["blocks_wrong"] >= Q1_MIN_BLOCKS
    fam = P.ANGLE_FAMILY[angle]; info["q4"] = bool(len(w)) and np.mean([fam in s for s in w.fams]) >= Q4_FAMILY_FRAC
    if not info["q1"]: return info
    a = auc(rv, wv); info["auc"] = a; info["q2"] = abs(a - 0.5) >= Q2_EFFECT
    if info["q2"] and info["q4"]:                                            # bootstrap only where it can still matter
        lo, hi = _boot_auc_interval(_by_block(r, angle), _by_block(w, angle), rng, nboot); info["q3"] = (lo >= Q3_BOUND) if a > 0.5 else (hi <= 1 - Q3_BOUND)
    info["qualifies"] = info["q1"] and info["q2"] and info["q3"] and info["q4"]; return info

def select_cells(dev, rng, nboot=NBOOT):
    """dev: DataFrame (asana, pclass, block, label, fams, angle columns) -> DataFrame of every cell with its checks and `qualifies`."""
    rows = []
    for (a, c), g in dev.groupby(["asana", "pclass"]):
        for n in P.ANGLE_NAMES: rows.append(dict(asana=a, pclass=c, angle=n, **evaluate_cell(g, n, rng, nboot)))
    return pd.DataFrame(rows)

def permute_labels(dev, rng):
    """Null copy: within each (asana, posture class), the (label, fault-family) bundle of each BLOCK is shuffled among that cell's blocks
    (block counts per label and within-block correlation preserved; a permuted Wrong block keeps the fault tags of the Wrong block it took its label from)."""
    out = dev.copy()
    for _, g in dev.groupby(["asana", "pclass"]):
        b = g.groupby("block").agg(label=("label", "first"), fams=("fams", "first")); perm = rng.permutation(len(b))
        new = pd.DataFrame({"label": b.label.to_numpy()[perm], "fams": b.fams.to_numpy()[perm]}, index=b.index)
        out.loc[g.index, "label"] = g.block.map(new.label).to_numpy(); out.loc[g.index, "fams"] = pd.Series(g.block.map(new.fams).to_numpy(), index=g.index)
    return out

def permutation_null(dev, n_perm=200, seed=0):
    """Counts of qualifying cells when the whole selection is run on label-permuted dev. Returns the list of counts."""
    rng = np.random.default_rng(seed); return [int(select_cells(permute_labels(dev, rng), rng).qualifies.sum()) for _ in range(n_perm)]

def judge_v2(df, refs, selected):
    """v2 decision. selected: set of (asana, pclass, angle). refs: {(asana, pclass, angle): Reference}. Per row: 'incorrect' if any selected+referenced angle is out of band,
    'correct' if at least one is judgeable and all are inside, 'abstain' if the class has no selected angle or none is judgeable (never a fault)."""
    st = pd.Series("abstain", index=df.index, dtype=object); nj = pd.Series(0, index=df.index)
    for (a, c), g in df.groupby(["asana", "pclass"]):
        names = [n for n in P.ANGLE_NAMES if (a, c, n) in selected and (a, c, n) in refs]
        if not names: continue
        lo = np.array([refs[(a, c, n)].lo for n in names]); hi = np.array([refs[(a, c, n)].hi for n in names]); D = P.deviations(g[names].to_numpy(float), lo, hi)
        n_j = (~np.isnan(D)).sum(1); bad = (np.nan_to_num(D) != 0).any(1); st[g.index] = np.where(n_j < 1, "abstain", np.where(bad, "incorrect", "correct")); nj[g.index] = n_j
    return pd.DataFrame({"status": st, "n_judged": nj})
