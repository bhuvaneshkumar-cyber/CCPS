"""Inter-annotator agreement (Cohen's kappa + bootstrap CI). Two modes, one kappa implementation.
  steps  : python tools/agreement.py steps NAME_A NAME_B        frame-level kappa over frames BOTH labelled + boundary offsets per matched segment
  labels : python tools/agreement.py labels file.csv --a colA --b colB [--block colBLOCK]   e.g. correct/incorrect or fault-type labels of two labelers
  test   : python tools/agreement.py --selftest
Bootstrap resamples whole CLIPS (steps) or BLOCKS (labels; pass time-block/clip ids) - never individual frames/images, which are highly correlated."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotate_steps as S

RNG = np.random.default_rng(0)  # bootstrap resampling only

def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0: return float("nan")
    cats = np.union1d(a, b); po = (a == b).mean(); pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)

def frame_labels(marks, n):
    """per-frame label array, -1 before the first mark."""
    out = np.full(n, -1, int)
    for s0, s1, lab in Marks(n, marks).segments(): out[s0:s1] = lab
    return out
Marks = S.Marks

def boundary_offsets(ma, mb, n, fps):
    """For every segment (step>0) of A, the segment of B with the same step and max overlap -> (start_off_s, end_off_s) = B - A."""
    sa = [s for s in Marks(n, ma).segments() if s[2] > 0]; sb = [s for s in Marks(n, mb).segments() if s[2] > 0]; out = []
    for a0, a1, k in sa:
        c = [(min(a1, b1) - max(a0, b0), b0, b1) for b0, b1, kb in sb if kb == k and min(a1, b1) > max(a0, b0)]
        if c: _, b0, b1 = max(c); out.append(((b0 - a0) / fps, (b1 - a1) / fps))
    return out

def boot_kappa(units, nb=1000):
    """units: list of (a_labels, b_labels) per clip/block. Point kappa on pooled data + 95% percentile CI over clip resampling."""
    pool = lambda us: cohen_kappa(np.concatenate([u[0] for u in us]), np.concatenate([u[1] for u in us]))
    ks = [pool([units[i] for i in RNG.integers(0, len(units), len(units))]) for _ in range(nb)]
    return pool(units), np.nanpercentile(ks, [2.5, 97.5])

def steps_mode(na, nb):
    vid, _, nsteps = S.load_meta(); A, B = S.load_ann(na)["videos"], S.load_ann(nb)["videos"]; units, offs, rows = [], [], []
    for _, r in vid.iterrows():
        k = r.path.replace("\\", "/")
        if not (A.get(k, {}).get("done") and B.get(k, {}).get("done")): continue
        la, lb = frame_labels(A[k]["marks"], int(r.frames)), frame_labels(B[k]["marks"], int(r.frames)); m = (la >= 0) & (lb >= 0)
        if m.sum() == 0: continue
        units.append((la[m], lb[m])); rows.append((r.asana, cohen_kappa(la[m], lb[m]), m.mean()))
        offs += [(r.asana, *o) for o in boundary_offsets(A[k]["marks"], B[k]["marks"], int(r.frames), r.fps)]
    if not units: sys.exit("no video is done by both annotators")
    k, ci = boot_kappa(units); print(f"videos in common: {len(units)} | frame-level kappa (frames labelled by both) = {k:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(pd.DataFrame(rows, columns=["asana", "kappa", "frac_frames_both"]).groupby("asana").mean().round(3).to_string())
    if offs:
        o = pd.DataFrame(offs, columns=["asana", "start_off_s", "end_off_s"]); ab = o[["start_off_s", "end_off_s"]].abs()
        print(f"\nmatched segments: {len(o)} | |start offset| median {ab.start_off_s.median():.2f}s, p90 {ab.start_off_s.quantile(.9):.2f}s | |end offset| median {ab.end_off_s.median():.2f}s, p90 {ab.end_off_s.quantile(.9):.2f}s")
        print(f"within 0.5 s: start {(ab.start_off_s <= .5).mean():.2f}, end {(ab.end_off_s <= .5).mean():.2f}; within 1 s: start {(ab.start_off_s <= 1).mean():.2f}, end {(ab.end_off_s <= 1).mean():.2f}")

def labels_mode(f, ca, cb, block):
    d = pd.read_csv(f).dropna(subset=[ca, cb]); bl = d[block] if block else pd.Series(np.arange(len(d)))
    if not block: print("WARNING: no --block given, resampling individual items; CI will be too narrow if items are bursts/frames of the same clip")
    units = [(g[ca].values, g[cb].values) for _, g in d.groupby(bl)]; k, ci = boot_kappa(units)
    print(f"n={len(d)} items in {len(units)} blocks | kappa = {k:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}] | raw agreement {(d[ca] == d[cb]).mean():.3f}")
    print(pd.crosstab(d[ca], d[cb], rownames=[ca], colnames=[cb]).to_string())

def selftest():
    assert cohen_kappa([1, 1, 1, 0], [1, 1, 1, 0]) == 1.0
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) - 0.5) < 1e-12          # hand-computed: po=.75, pe=.5*.25+.5*.75=.5 -> (.75-.5)/.5 = .5
    assert abs(cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) - (-1.0)) < 1e-12        # total disagreement on balanced labels
    assert np.isnan(cohen_kappa([1, 1], [1, 1]))                                # pe == 1: kappa undefined, must not divide by zero
    fl = frame_labels({100: 1, 300: 0}, 400); assert (fl[:100] == -1).all() and (fl[100:300] == 1).all() and (fl[300:] == 0).all()
    off = boundary_offsets({100: 1, 300: 0}, {130: 1, 290: 0}, 400, 30.0)      # B starts 30 frames late (+1.0 s), ends 10 frames early (-0.333 s)
    assert len(off) == 1 and abs(off[0][0] - 1.0) < 1e-9 and abs(off[0][1] + 10 / 30) < 1e-9, off
    assert boundary_offsets({100: 1, 300: 0}, {100: 2, 300: 0}, 400, 30.0) == []  # different step -> not matched
    print("agreement selftest OK")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", nargs="?"); ap.add_argument("x", nargs="?"); ap.add_argument("y", nargs="?")
    ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--block"); ap.add_argument("--selftest", action="store_true"); g = ap.parse_args()
    if g.selftest: selftest()
    elif g.mode == "steps": steps_mode(g.x, g.y)
    elif g.mode == "labels": labels_mode(g.x, g.a, g.b, g.block)
    else: ap.print_help()
