"""Phase 3 test stage (stills): ONE look at the test blocks with the frozen configuration. Refuses to run if the protocol or library changed since the freeze,
and refuses a second look unless --second-look is given (which is then recorded in the output).
usage: python scripts/phase3_test.py [--second-look]"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poseguard import config as C
from poseguard import phase3 as P

sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest(); PH = ROOT / "phase3"
smoke = "--smoke" in sys.argv; second = "--second-look" in sys.argv
if smoke:   # bug-shaking only: the REAL test rows are removed, dev blocks are re-split; nothing here is a result and no marker is written
    cfg = dict(p=2.5, delta=10.0, K=C.STILL_STRETCH_K, tau=C.KPT_CONF_TAU); print("*** SMOKE RUN: real test rows removed, dev blocks re-split - NOT RESULTS ***")
else:
    cfg = json.load(open(PH / "frozen_config.json"))
    assert cfg["protocol_sha256"] == sha(PH / "PROTOCOL.md") and cfg["phase3_py_sha256"] == sha(ROOT / "poseguard" / "phase3.py"), "protocol or library changed after the freeze - refusing to score the test split"
    if (PH / "TEST_TOUCHED.json").exists() and not second: sys.exit("the test split was already scored once; a second look needs --second-look and will be recorded")
    looks = json.load(open(PH / "TEST_TOUCHED.json")) if (PH / "TEST_TOUCHED.json").exists() else []
    looks.append(dict(at=time.strftime("%Y-%m-%d %H:%M:%S"), second_look=second, frozen_config=cfg)); json.dump(looks, open(PH / "TEST_TOUCHED.json", "w"), indent=1)
def load(k=None):
    S = P.load_stills(k)
    if smoke: S = S[S.split != "test"].copy(); S["split"] = np.where(S.k % 3 == 1, "test", "dev")
    return S
p, dl = cfg["p"], cfg["delta"]

def evaluate(S, p, dl):
    refs = P.derive_refs(S[(S.split == "dev") & (S.label == "Right")], p, dl)
    T = S[S.split == "test"].copy(); T["y"] = (T.label == "Wrong").astype(int)
    T["fams"] = [P.tag_families(f)[0] if l == "Wrong" else set() for l, f in zip(T.label, T.fault)]; T["oov_only"] = (T.label == "Wrong") & T.fams.map(len).eq(0)
    T = T.join(P.judge_frame(T, refs)); return refs, T

def per_asana(T):
    sc = T[(~T.oov_only) & (T.status != "abstain")]; return {a: P.prf(g.y, g.status == "incorrect") for a, g in sc.groupby("asana")}
def macro_f1(T):
    sc = T[(~T.oov_only) & (T.status != "abstain")]; return float(np.nanmean([P.prf(g.y, g.status == "incorrect")["f1"] for _, g in sc.groupby("asana")]))

def macro_ba(T):
    sc = T[(~T.oov_only) & (T.status != "abstain")]; return float(np.nanmean([P.prf(g.y, g.status == "incorrect")["bal_acc"] for _, g in sc.groupby("asana")]))

S = load(); refs, T = evaluate(S, p, dl); head = T[~T.oov_only]; sc = head[head.status != "abstain"]
print(f"frozen config: p={p} delta={dl} K={cfg['K']} tau={cfg['tau']}  (second look: {second})")
print(f"test rows {len(T)} | headline (Right + in-vocabulary Wrong) {len(head)} | scored (not abstained) {len(sc)} | abstain rate {float((head.status == 'abstain').mean()):.3f}")
rows = []
for a in P.ASANAS:
    g = sc[sc.asana == a]
    if g.empty: rows.append(dict(asana=a, note="no scored test rows (no reference or all abstained)")); continue
    m = P.prf(g.y, g.status == "incorrect"); ci = {k: P.boot_stat(g, lambda d, k=k: P.prf(d.y, d.status == "incorrect")[k])[1] for k in ("f1", "precision", "recall")}
    rows.append(dict(asana=a, n=m["n"], n_right=int((g.y == 0).sum()), n_wrong=int((g.y == 1).sum()), blocks=g.block.nunique(), accuracy=m["accuracy"], bal_acc=m["bal_acc"], precision=m["precision"], p_ci=ci["precision"], recall=m["recall"], r_ci=ci["recall"], f1=m["f1"], f1_ci=ci["f1"],
                     abstain=float((head[head.asana == a].status == "abstain").mean())))
R = pd.DataFrame(rows); pd.set_option("display.width", 250, "display.max_colwidth", 40); print(R.round(3).to_string(index=False))
mp, mci = P.boot_stat(sc, macro_f1); print(f"\nMACRO F1 (incorrect) over asanas = {mp:.3f}  95% block-bootstrap CI [{mci[0]:.3f}, {mci[1]:.3f}]")
bp, bci = P.boot_stat(sc, macro_ba); print(f"MACRO BALANCED ACCURACY over asanas = {bp:.3f}  95% block-bootstrap CI [{bci[0]:.3f}, {bci[1]:.3f}]  (chance / any constant predictor = 0.500)")
allm = P.prf(sc.y, sc.status == "incorrect"); print("pooled (micro):", {k: round(v, 3) if isinstance(v, float) else v for k, v in allm.items()})
base = float(sc.y.mean()); print(f"base rate of 'incorrect' among scored test rows = {base:.3f} (an always-'incorrect' predictor has F1 = {2 * base / (1 + base):.3f}, precision = {base:.3f})")

print("\nPER-ANGLE COVERAGE on headline test rows (fraction computable / fraction judged with a reference):")
cov = pd.DataFrame({n: [head[n].notna().mean(), np.mean([(a, c, n) in refs for a, c in zip(head.asana, head.pclass)])] for n in P.ANGLE_NAMES}, index=["computable", "has_reference"]).round(2); print(cov.to_string())

print("\nFAULT FAMILIES (in-vocabulary Wrong test rows, not abstained):")
w = sc[sc.y == 1]
for f in ("legs", "arms", "trunk"):
    g = w[w.fams.map(lambda s: f in s)]
    if len(g):
        att = g[g.status == "incorrect"]
        print(f"  tag contains {f:5s}: n={len(g):4d}  flagged incorrect {float((g.status == 'incorrect').mean()):.3f}  | of the flagged, largest-deviation family is one of the tag's families: "
              f"{float(np.mean([tf in s for tf, s in zip(att.top_family, att.fams)])) if len(att) else float('nan'):.3f}")
flag = w[(w.status == "incorrect") & (w.fams.map(len) == 1)]; flag = flag.assign(true_family=flag.fams.map(lambda s: next(iter(s))))
print("  confusion (rows: true single-family tag; cols: family of the largest deviation), flagged only:"); print(pd.crosstab(flag.true_family, flag.top_family).to_string() if len(flag) else "  (none)")
att = w[w.status == "incorrect"]; print(f"  attribution: largest-deviation family is one of the tag's families in {float(np.mean([f in s for f, s in zip(att.top_family, att.fams)])):.3f} of {len(att)} flagged in-vocabulary Wrong rows")
oo = T[T.oov_only & (T.status != 'abstain')]; print(f"\nOUT-OF-VOCABULARY Wrong (neck/head/ankle/standing only), not in headline: n={len(oo)}, flagged incorrect {float((oo.status == 'incorrect').mean()):.3f}" if len(oo) else "\nno out-of-vocabulary Wrong rows scored")
nr = head[head.status == "abstain"].groupby("asana").size(); print("abstained headline rows by asana:", nr.to_dict())

print("\nSENSITIVITY (reported, not used for selection): macro-F1 on the same test rows")
for name, k, pp, d in (("K=1.50", 1.5, p, dl), ("K=1.65 (frozen)", 1.65, p, dl), ("K=1.85", 1.85, p, dl), ("delta-5 (floor 0)", None, p, max(0.0, dl - 5)), ("delta+5", None, p, dl + 5),
                       ("first-run F1-selected p=5,d=5", None, 5.0, 5.0)):
    _, T2 = evaluate(S if k is None else load(k), pp, d); print(f"  {name:32s} macro-F1 = {macro_f1(T2):.3f}   macro-BA = {macro_ba(T2):.3f}")
if not smoke:
    R.to_csv(PH / "results_stills_per_asana.csv", index=False); print("saved phase3/results_stills_per_asana.csv")
else: print("smoke run: nothing saved")
