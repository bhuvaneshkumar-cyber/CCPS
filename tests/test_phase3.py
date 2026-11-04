"""Phase 3 library tests. Hand-built timestamps/values are explicit TEST fixtures (pure-function checks), never data used for results.
Real-manifest tests use the actual dataset tables (posture classes, fault tags). Run: python tests/test_phase3.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard import config as C
from poseguard import phase3 as P
from poseguard.angles import ANGLE_BY_NAME
from poseguard.rules import Reference, judge

def folder(span_s, step=1, n=None):   # TEST fixture: one 1-image-per-second folder
    t0 = pd.Timestamp("2023-03-10 10:00:00"); n = span_s + 1 if n is None else n
    return pd.DataFrame({"asana": "X", "label": "Right", "step": step, "t": [t0 + pd.Timedelta(seconds=float(s)) for s in np.linspace(0, span_s, n)]})

def test_blocks_120s_folder():
    d = P.assign_blocks(folder(120)); assert set(d.split) <= {"dev", "test", "guard"} and d.k.nunique() == 3   # nb = round(120/40) = 3
    sec = (d.t - d.t.min()).dt.total_seconds()
    assert (d.split[d.k == 2].isin(["test", "guard"])).all() and (d.split[d.k < 2].isin(["dev", "guard"])).all()   # k mod 3 == 2 -> test
    t_test, t_dev = sec[d.split == "test"].to_numpy(), sec[d.split == "dev"].to_numpy()
    assert min(abs(a - b) for a in t_test for b in t_dev) >= 4 - 1e-9               # guard band: nearest test/dev images are >= 4 s apart
    assert set(sec[d.split == "guard"].round()) >= {37, 38, 39, 40, 41, 42, 43, 77, 78, 79, 81, 82, 83}   # images near the two internal block edges are dropped
    assert (P.assign_blocks(folder(120)).split.values == d.split.values).all()       # deterministic

def test_blocks_short_folder_uses_scaled_guard():
    d = P.assign_blocks(folder(45)); assert d.k.nunique() == 3                       # span 45 -> clip(round(1.125), 3, 10) = 3 blocks of 15 s; guard = min(4, 2.25) = 2.25 s
    sec = (d.t - d.t.min()).dt.total_seconds(); assert (d.split == "test").any() and (d.split == "dev").any()
    assert min(abs(a - b) for a in sec[d.split == "test"] for b in sec[d.split == "dev"]) >= 2.25 - 1e-9

def test_blocks_many_blocks_alternate():
    d = P.assign_blocks(folder(400)); assert d.k.nunique() == 10                     # round(400/40) = 10
    assert set(d.k[d.split == "test"]) == {2, 5, 8} and set(d.k[d.split == "dev"]) == {0, 1, 3, 4, 6, 7, 9}

def test_band_hand_computed():
    r = P.band(np.arange(101.0), 5, 5); assert (r.lo, r.hi) == (0.0, 100.0)          # 5th pct of 0..100 = 5, 95th = 95, widened by 5

def test_vectorised_deviation_equals_rules_judge():
    lo, hi = np.array([170.0, 100.0, np.nan]), np.array([180.0, 150.0, 90.0]); vals = np.array([[150.0, 160.0, 50.0], [175.0, 100.0, np.nan], [181.0, 99.0, 50.0]])
    D = P.deviations(vals, lo, hi); spec = ANGLE_BY_NAME["left_knee"]
    for i in range(3):
        for j in range(3):
            if np.isnan(lo[j]) or np.isnan(vals[i, j]): assert np.isnan(D[i, j]); continue
            v = judge(spec, vals[i, j], Reference(lo[j], hi[j])); assert abs(v.deviation - D[i, j]) < 1e-12, (i, j)

def refs_for(names, lo=170.0, hi=180.0): return {("X", 1, n): Reference(lo, hi) for n in names}
def rowframe(**vals):
    r = {n: np.nan for n in P.ANGLE_NAMES}; r.update(vals); return pd.DataFrame([{"asana": "X", "pclass": 1, **r}])

def test_decision_rule():
    refs = refs_for(["left_knee", "right_knee", "left_elbow", "right_elbow"])
    assert P.judge_frame(rowframe(left_knee=175, right_knee=172, left_elbow=180, right_elbow=171), refs).status[0] == "correct"
    r = P.judge_frame(rowframe(left_knee=150, right_knee=172, left_elbow=175, right_elbow=171), refs)
    assert r.status[0] == "incorrect" and r.top_family[0] == "legs"                  # knee 20 off vs nothing else -> legs
    r = P.judge_frame(rowframe(left_knee=150, right_knee=172, left_elbow=160, right_elbow=171), refs)   # knee 20 off, elbow 10 off -> largest is the knee
    assert r.top_family[0] == "legs"
    r = P.judge_frame(rowframe(left_knee=175, left_elbow=160, right_knee=172), refs); assert r.status[0] == "incorrect" and r.top_family[0] == "arms" and r.n_judged[0] == 3
    r = P.judge_frame(rowframe(left_knee=150, right_knee=172), refs); assert r.status[0] == "abstain" and r.n_judged[0] == 2   # only 2 angles judgeable: never a fault
    r = P.judge_frame(rowframe(trunk_tilt=90, left_knee=175, right_knee=172, left_elbow=175), refs); assert r.n_judged[0] == 3 and r.status[0] == "correct"   # angle without a reference is not judged

def test_reference_needs_enough_samples_and_blocks():
    mk = lambda n, nblocks: pd.DataFrame({"asana": "X", "pclass": 1, "block": [f"b{i % nblocks}" for i in range(n)], **{a: np.linspace(170, 180, n) for a in P.ANGLE_NAMES}})
    assert len(P.derive_refs(mk(40, 3), 5, 5)) == len(P.ANGLE_NAMES)                 # exactly the minimum: 40 values, 3 blocks
    assert P.derive_refs(mk(39, 3), 5, 5) == {} and P.derive_refs(mk(40, 2), 5, 5) == {}

def test_prf_hand_computed():
    m = P.prf([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])                                      # tp=2 fp=1 fn=1 tn=1
    assert abs(m["precision"] - 2 / 3) < 1e-12 and abs(m["recall"] - 2 / 3) < 1e-12 and abs(m["f1"] - 4 / 6) < 1e-12 and abs(m["accuracy"] - 3 / 5) < 1e-12
    assert abs(m["specificity"] - 1 / 2) < 1e-12 and abs(m["bal_acc"] - 7 / 12) < 1e-12   # recall 2/3, specificity tn/(tn+fp) = 1/2 -> (2/3 + 1/2)/2 = 7/12
    assert np.isnan(P.prf([0, 0], [0, 0])["f1"]) and np.isnan(P.prf([0, 0], [0, 0])["precision"])   # undefined, not 0 or an error

def test_block_bootstrap_resamples_blocks_not_images():
    df = pd.DataFrame({"block": ["a"] * 10 + ["b"] * 10, "v": [1.0] * 10 + [0.0] * 10})
    pt, (lo, hi) = P.boot_stat(df, lambda d: d.v.mean(), nb=500)
    assert pt == 0.5 and lo in (0.0, 0.5) and hi in (0.5, 1.0) and lo < hi          # with 2 blocks the mean can only be 0, .5 or 1: per-image resampling would give a narrow CI near .5
    assert P.boot_stat(df, lambda d: d.v.mean(), nb=500) == (pt, (lo, hi))          # seeded -> reproducible

def test_posture_classes_on_real_manifest():
    man = pd.read_csv(C.ROOT / "audit" / "manifest_dedup.csv"); pc = P.posture_classes(man[~man.ambiguous])
    assert pc[("Tadasana", 5)] == 1 and pc[("Tadasana", 4)] == 2 and pc[("Tadasana", 3)] == 3 and pc[("Bhujangasana", 3)] == 1 and pc[("Bhujangasana", 2)] == 2
    assert all(pc[("Ardhakati Chakrasana", s)] == s for s in range(1, 6)) and pc[("Marjariasana", 4)] == 4   # no copies -> each step is its own posture

def test_fault_tag_mapping_on_every_real_tag():
    man = pd.read_csv(C.ROOT / "audit" / "manifest_dedup.csv"); tags = sorted(set(man[man.label == "Wrong"].fault.dropna()))
    for t in tags: f, oov = P.tag_families(t); assert f or oov, f"tag {t!r} maps to nothing"     # nothing may fall through silently
    tf = P.tag_families; assert tf("Legs") == ({"legs"}, False) and tf("Knee") == ({"legs"}, False) and tf("Palms") == ({"arms"}, False)
    assert tf("Leg and Hand") == ({"legs", "arms"}, False) and tf("Hands and Legs") == ({"legs", "arms"}, False)
    assert tf("Neck and Hand") == ({"arms"}, True) and tf("Back and Head") == ({"trunk"}, True) and tf("Back (Overbend)") == ({"trunk"}, False)
    assert tf("Standing") == (set(), True) and tf("Ankle") == (set(), True) and tf("Head") == (set(), True) and tf("Neck") == (set(), True)

if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"): f(); print("ok ", n)
    print("all phase3 tests passed")
