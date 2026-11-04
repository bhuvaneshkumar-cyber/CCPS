"""geo_v2_lightly_supervised tests: hand-built TEST fixtures (each qualification check must be able to reject a cell on its own). Run: python tests/test_geo_v2.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard import geo_v2 as G
from poseguard import phase3 as P
from poseguard.rules import Reference

ANG = "left_knee"                                   # family: legs

def cell(right_blocks, wrong_blocks, right_val, wrong_vals, fams=frozenset({"legs"}), per_block=10):
    """TEST fixture: right_blocks Right blocks with constant value; wrong_blocks Wrong blocks, block j has constant wrong_vals[j % len]."""
    rows = []
    for i in range(right_blocks): rows += [dict(asana="X", pclass=1, block=f"R{i}", label="Right", fams=frozenset(), **{n: (right_val if n == ANG else np.nan) for n in P.ANGLE_NAMES}) for _ in range(per_block)]
    for j in range(wrong_blocks): rows += [dict(asana="X", pclass=1, block=f"W{j}", label="Wrong", fams=fams, **{n: (wrong_vals[j % len(wrong_vals)] if n == ANG else np.nan) for n in P.ANGLE_NAMES}) for _ in range(per_block)]
    return pd.DataFrame(rows)

rng = lambda: np.random.default_rng(0)

def test_auc_hand_computed():
    assert abs(G.auc([1, 2], [2, 3]) - 0.875) < 1e-12                                # (3 wins + 1 tie*0.5) / 4
    assert G.auc([1, 1], [5, 5]) == 1.0 and G.auc([5, 5], [1, 1]) == 0.0 and G.auc([1, 2], [1, 2]) == 0.5

def test_clear_cell_qualifies():
    r = G.evaluate_cell(cell(8, 8, 100.0, [150.0]), ANG, rng()); assert r["q1"] and r["q2"] and r["q3"] and r["q4"] and r["qualifies"] and r["auc"] == 1.0

def test_q1_support_rejects_too_few_blocks_or_values():
    r = G.evaluate_cell(cell(3, 8, 100.0, [150.0]), ANG, rng()); assert not r["q1"] and not r["qualifies"]              # 3 Right blocks < 4
    r = G.evaluate_cell(cell(8, 8, 100.0, [150.0], per_block=7), ANG, rng()); assert not r["q1"]                       # 56 < 60 values per class
    r = G.evaluate_cell(cell(4, 4, 100.0, [150.0], per_block=15), ANG, rng()); assert r["q1"] and r["qualifies"]        # exactly enough: 4 blocks, 60 values

def test_q2_effect_size_rejects_small_effect_alone():
    # Wrong blocks: 3 of 8 above the Right value, 5 tied -> AUC = (3*1 + 5*0.5)/8 = 0.6875?? use 2 above / 6 tied -> (2 + 3)/8 = 0.625: |0.125| < 0.20
    r = G.evaluate_cell(cell(8, 8, 100.0, [150.0, 150.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]), ANG, rng())
    assert r["q1"] and r["q4"] and abs(r["auc"] - 0.625) < 1e-9 and not r["q2"] and not r["qualifies"]

def test_q3_stability_rejects_effect_carried_by_two_blocks():
    r = G.evaluate_cell(cell(8, 4, 100.0, [150.0, 150.0, 100.0, 100.0], per_block=15), ANG, rng())     # AUC 0.75 passes Q2, but resampling 4 Wrong blocks often draws 0-1 high ones
    assert r["q1"] and r["q2"] and r["q4"] and abs(r["auc"] - 0.75) < 1e-9 and not r["q3"] and not r["qualifies"]

def test_q4_family_fit_rejects_unrelated_angle_alone():
    r = G.evaluate_cell(cell(8, 8, 100.0, [150.0], fams=frozenset({"arms"})), ANG, rng())   # knee separates, but the labelled faults are arm faults
    assert r["q1"] and r["q2"] and not r["q4"] and not r["qualifies"]
    ok = G.evaluate_cell(cell(8, 8, 100.0, [150.0], fams=frozenset({"arms", "legs"})), ANG, rng()); assert ok["q4"] and ok["qualifies"]

def test_direction_is_symmetric():
    r = G.evaluate_cell(cell(8, 8, 150.0, [100.0]), ANG, rng()); assert r["auc"] == 0.0 and r["qualifies"]      # Wrong LOWER than Right also qualifies (|AUC-0.5|)

def test_permutation_preserves_block_structure():
    d = pd.concat([cell(6, 5, 100.0, [150.0]), cell(6, 5, 100.0, [150.0]).assign(asana="Y")], ignore_index=True); p = G.permute_labels(d, np.random.default_rng(1))
    for a in ("X", "Y"):
        assert p[p.asana == a].groupby("block").label.nunique().max() == 1                              # a block never mixes labels
        assert p[p.asana == a].groupby("block").label.first().value_counts().to_dict() == {"Right": 6, "Wrong": 5}   # counts of Right/Wrong blocks preserved
    assert (p.label != d.label).any()                                                                   # it actually permuted something

def test_permuted_null_selects_nothing_from_a_cell_with_no_structure():
    same = pd.concat([cell(8, 8, 100.0, [100.0]) ], ignore_index=True)                                  # Wrong values identical to Right: no effect anywhere
    assert int(G.select_cells(same, rng()).qualifies.sum()) == 0

def test_judge_v2_uses_only_selected_angles_and_abstains_without_any():
    refs = {("X", 1, "left_knee"): Reference(170, 180), ("X", 1, "left_elbow"): Reference(170, 180)}; sel = {("X", 1, "left_knee")}
    mk = lambda **v: pd.DataFrame([{"asana": "X", "pclass": 1, **{n: np.nan for n in P.ANGLE_NAMES}, **v}])
    assert G.judge_v2(mk(left_knee=175, left_elbow=100), refs, sel).status[0] == "correct"             # elbow is way out but NOT selected: ignored
    assert G.judge_v2(mk(left_knee=150, left_elbow=175), refs, sel).status[0] == "incorrect"
    assert G.judge_v2(mk(left_elbow=100), refs, sel).status[0] == "abstain"                            # selected angle not computable -> abstain, never a fault
    assert G.judge_v2(mk(left_knee=150), refs, set()).status[0] == "abstain"                            # class with no selected angle -> abstain

if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"): f(); print("ok ", n)
    print("all geo_v2 tests passed")
