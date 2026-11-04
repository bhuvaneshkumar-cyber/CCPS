"""Phase 0 follow-up: estimate the horizontal stretch k of the Mendeley stills (corrected x = x / k) against undistorted video.

Estimator A (shape matching, all asanas/views): un-stretch still keypoints by candidate k, score = mean over stills of the distance to the
  nearest video pose (same asana, torso+leg joints, similarity-normalised: translation, scale, rotation; no reflection). k_hat = argmin.
Estimator B (frontal-ratio, standing asanas only, independent): k = q90(shoulder-span/torso-length | upright torso) stills / video.
Uncertainty: bootstrap over TIME BLOCKS (stills: 60 s blocks per asana/label; video/pseudo: whole clips), never over images.
Validation (distortion_pseudo.py frames, KNOWN k): squash -> k_true=1280/720, null -> 1.0. Pre-registered tolerance |k_hat - k_true| <= 0.15.
usage: python audit/distortion.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard.config import KPT_CONF_TAU as TAU, ROOT

J = [5, 6, 11, 12, 13, 14, 15, 16]  # shoulders, hips, knees, ankles (arms/head dropped: often occluded / low confidence)
GRID = np.round(np.arange(0.8, 3.0001, 0.05), 2)
EXCL = ("Sarvangasana", "Viparita Karani"); STANDING = ("Tadasana", "Kati Chakrasana", "Ardhakati Chakrasana")
TOL, NB, RNG = 0.15, 300, np.random.default_rng(0)  # RNG: bootstrap resampling only, no data involved

def ok(kp, idx=J): c = np.nan_to_num(kp[..., idx, 2]); return (c >= TAU).all(-1)

def sim(A, B):
    """max over rotations of <A,R B> for unit-norm centred shapes; A (n,J,2), B (m,J,2) -> (n,m). Closed form for 2x2 cross-covariance."""
    nz = lambda X: (X - X.mean(1, keepdims=True)) / np.linalg.norm(X - X.mean(1, keepdims=True), axis=(1, 2), keepdims=True)
    M = np.einsum("nja,mjb->nmab", nz(A), nz(B))
    return np.sqrt((M[..., 0, 0] + M[..., 1, 1]) ** 2 + (M[..., 0, 1] - M[..., 1, 0]) ** 2)

def dcurve(S, V):
    """S (n,17,3) still-like kp, V (m,17,3) reference kp -> (n,K) nearest-pose distance for each candidate k."""
    out = np.empty((len(S), len(GRID)))
    for c, k in enumerate(GRID):
        Sk = S[:, J, :2].copy(); Sk[..., 0] /= k
        out[:, c] = np.sqrt(np.clip(2 - 2 * sim(Sk, V[:, J, :2]).max(1), 0, None))
    return out

def boot_argmin(curves, blocks):
    """curves: {group: (n,K)}, blocks: {group: (n,) block ids}. Equal-weight over groups. -> point k_hat, 95% CI, pooled curve."""
    pooled = np.mean([c.mean(0) for c in curves.values()], 0); ks = []
    for _ in range(NB):
        cs = []
        for g, c in curves.items():
            u = np.unique(blocks[g]); pick = RNG.choice(u, len(u)); idx = np.concatenate([np.flatnonzero(blocks[g] == b) for b in pick])
            cs.append(c[idx].mean(0))
        ks.append(GRID[np.mean(cs, 0).argmin()])
    return GRID[pooled.argmin()], np.percentile(ks, [2.5, 97.5]), pooled

def ratio(kp, up=10):
    """shoulder span / torso length, upright torsos only (torso within `up` deg of image vertical)."""
    m = ok(kp, [5, 6, 11, 12]); kp = np.where(m[:, None, None], kp, np.nan)
    sh, hp = kp[:, [5, 6], :2].mean(1), kp[:, [11, 12], :2].mean(1); t = sh - hp
    upright = np.degrees(np.abs(np.arctan2(t[:, 0], -t[:, 1]))) <= up  # y grows downward: hip->shoulder points up
    r = np.linalg.norm(kp[:, 5, :2] - kp[:, 6, :2], axis=1) / np.linalg.norm(t, axis=1)
    return r, m & upright

def estimator_B(rs, bs, rv, bv):
    """q90 ratio still/video with block bootstrap on both sides."""
    q = lambda r: np.nanpercentile(r, 90)
    ks = []
    for _ in range(NB):
        ps = RNG.choice(np.unique(bs), len(np.unique(bs))); pv = RNG.choice(np.unique(bv), len(np.unique(bv)))
        ks.append(q(rs[np.concatenate([np.flatnonzero(bs == b) for b in ps])]) / q(rv[np.concatenate([np.flatnonzero(bv == b) for b in pv])]))
    return q(rs) / q(rv), np.percentile(ks, [2.5, 97.5])

# ---------- data ----------
man = pd.read_csv(ROOT / "audit" / "manifest_dedup.csv")
man["t"] = pd.to_datetime(man.date.astype(str).str.zfill(8) + man.time.astype(str).str.zfill(6), format="%Y%m%d%H%M%S")
man["block"] = man.groupby(["asana", "label"]).t.transform(lambda s: ((s - s.min()).dt.total_seconds() // 60).astype(int)).astype(str) + man.asana + man.label
z = np.load(ROOT / "audit" / "kpts_320_all.npz", allow_pickle=True)
st = pd.DataFrame({"path": z["path"], "i": np.arange(len(z["path"]))}).merge(man[["path", "asana", "label", "usable", "block"]], on="path")
st = st[st.usable & ~st.asana.isin(EXCL)]; SKP = z["kp"][st.i.values]; st = st.reset_index(drop=True)
vid = pd.read_csv(ROOT / "audit" / "inventory_videos.csv"); zv = np.load(ROOT / "audit" / "kpts_video_640.npz", allow_pickle=True); VKP = zv["kp"]  # (80,40,17,3)
asanas = [a for a in sorted(st.asana.unique())]

def vset(asana, exclude_angle=None):
    sel = vid[(vid.asana == asana) & (vid.angle != exclude_angle)].index if exclude_angle else vid[vid.asana == asana].index
    v = VKP[sel].reshape(-1, 17, 3); return v[ok(v)]

def run_A(groups):
    """groups: {name: (S_kp, V_kp, block_ids)} -> per-group k_hat (+CI) and pooled."""
    curves, blocks, rows = {}, {}, []
    for n, (S, V, b) in groups.items():
        m = ok(S); S, b = S[m], b[m]
        if len(S) > 400: sel = np.linspace(0, len(S) - 1, 400).astype(int); S, b = S[sel], b[sel]  # deterministic thin, keeps time coverage
        curves[n], blocks[n] = dcurve(S, V), np.asarray(b)
        k, ci, pc = boot_argmin({n: curves[n]}, {n: blocks[n]}); rows.append((n, len(S), len(V), k, *ci, pc[GRID == 1.0][0] - pc.min()))
    kp_, cip, pooled = boot_argmin(curves, blocks)
    return pd.DataFrame(rows, columns=["group", "n_still", "n_ref", "k_hat", "ci_lo", "ci_hi", "depth(d(1)-min)"]), (kp_, cip)

pd.set_option("display.width", 200)
# ---------- validation on known transforms ----------
zp = np.load(ROOT / "audit" / "kpts_pseudo.npz")
for cond, ktrue in (("null", 1.0), ("squash", 1280 / 720)):
    kp, vi = zp[f"{cond}_kp"], zp[f"{cond}_vi"]; g = {}
    for a in asanas:
        for ang in (1, 2, 3, 4):  # leave-one-view-out: pseudo-stills of angle `ang`, reference = the other three angles
            sel = np.flatnonzero(np.isin(vi, vid[(vid.asana == a) & (vid.angle == ang)].index))
            if len(sel): g[f"{a}|{ang}"] = (kp[sel], vset(a, ang), vi[sel])
    # pool the four LOVO folds per asana: sum curves by giving each fold its own group, then report per-asana medians
    t, (kh, ci) = run_A(g)
    t["asana"] = t.group.str.split("|").str[0]
    per = t.groupby("asana").k_hat.median()
    print(f"\n[VALIDATION A/{cond}] known k={ktrue:.3f}: pooled k_hat={kh:.2f} CI[{ci[0]:.2f},{ci[1]:.2f}] | per-asana median k_hat range [{per.min():.2f},{per.max():.2f}] median {per.median():.2f} -> {'PASS' if abs(kh - ktrue) <= TOL else 'FAIL'} (tol {TOL})")
    print(t.groupby("asana")[["k_hat", "depth(d(1)-min)"]].median().round(2).T.to_string())
    # B on standing asanas
    rs, ms = ratio(kp); rv_, mv = ratio(VKP[vid[vid.asana.isin(STANDING)].index].reshape(-1, 17, 3))
    sel_s = np.isin(vi, vid[vid.asana.isin(STANDING)].index) & ms
    kb, cib = estimator_B(rs[sel_s], vi[sel_s], rv_[mv], np.repeat(vid[vid.asana.isin(STANDING)].index, VKP.shape[1])[mv])
    print(f"[VALIDATION B/{cond}] known k={ktrue:.3f}: k_hat={kb:.2f} CI[{cib[0]:.2f},{cib[1]:.2f}] -> {'PASS' if abs(kb - ktrue) <= TOL else 'FAIL'}")

# ---------- real stills ----------
g = {a: (SKP[st.asana == a], vset(a), st.block[st.asana == a].values) for a in asanas}
t, (kh, ci) = run_A(g)
print("\n[REAL STILLS A] per asana (usable stills, block-bootstrap 95% CI):\n", t.round(2).to_string(index=False))
print(f"pooled k_hat={kh:.2f} CI[{ci[0]:.2f},{ci[1]:.2f}] | across-asana median {t.k_hat.median():.2f}, range [{t.k_hat.min():.2f},{t.k_hat.max():.2f}]")
for a in STANDING:
    rs, ms = ratio(SKP[st.asana == a]); v = VKP[vid[vid.asana == a].index].reshape(-1, 17, 3)
    rv_, mv = ratio(v); kb, cib = estimator_B(rs[ms], st.block[st.asana == a].values[ms], rv_[mv], np.repeat(vid[vid.asana == a].index, VKP.shape[1])[mv])
    print(f"[REAL STILLS B] {a:22s} k={kb:.2f} CI[{cib[0]:.2f},{cib[1]:.2f}]  (n_still={ms.sum()}, n_video={mv.sum()})")
