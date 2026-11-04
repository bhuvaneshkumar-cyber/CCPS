"""Phase 5 unit tests. ALL keypoint sets and reference ranges below are hand-constructed TEST fixtures (explicitly not data, not results):
they encode a known deviation (e.g. 'right knee 20 deg beyond the reference') and assert what the engine says - above all the SIDE it names,
because a mirrored-side bug is invisible in aggregate metrics. Run: python tests/test_feedback.py   (or pytest)"""
import re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard import lr
from poseguard.angles import ANGLE_BY_NAME, compute_angles
from poseguard.feedback import TemporalFlag, generate, severity_of
from poseguard.rules import Reference, judge_all

TEST_REFS = {"left_knee": Reference(170, 180), "right_knee": Reference(170, 180), "left_elbow": Reference(170, 180), "right_elbow": Reference(170, 180)}  # TEST ONLY

def person():
    """Front-facing standing person as the camera sees them (verified convention: subject's LEFT is on the image's RIGHT), arms hanging straight."""
    kp = np.zeros((17, 3)); kp[:, 2] = 0.9
    for i, xy in {5: (600, 300), 6: (400, 300), 7: (640, 450), 8: (360, 450), 9: (680, 600), 10: (320, 600), 11: (570, 600), 12: (430, 600),
                  13: (570, 800), 14: (430, 800), 15: (570, 1000), 16: (430, 1000)}.items(): kp[i, :2] = xy
    return kp

def set_interior(kp, hip, knee, ank, deg):
    """Move the ankle (keeping leg length) so the interior angle at the knee is exactly `deg`."""
    L = np.linalg.norm(kp[ank, :2] - kp[knee, :2]); up = (kp[hip, :2] - kp[knee, :2]) / np.linalg.norm(kp[hip, :2] - kp[knee, :2]); th = np.radians(180 - deg)
    c, s = np.cos(th), np.sin(th); kp[ank, :2] = kp[knee, :2] + L * (np.array([[c, -s], [s, c]]) @ (-up))
    return kp

def run(kp, refs=TEST_REFS, temporal=None, top_k=3):
    return generate(judge_all(compute_angles(kp), refs, ANGLE_BY_NAME), temporal, top_k)

def with_lr(anat, mirrored, fn):
    old = (lr.BACKBONE_LEFT_IS_ANATOMICAL_LEFT, lr.INPUT_MIRRORED); lr.BACKBONE_LEFT_IS_ANATOMICAL_LEFT, lr.INPUT_MIRRORED = anat, mirrored
    try: return fn()
    finally: lr.BACKBONE_LEFT_IS_ANATOMICAL_LEFT, lr.INPUT_MIRRORED = old

def test_right_knee_20_degrees_beyond_reference_says_right_not_left():   # the case named in the spec
    kp = set_interior(person(), 12, 14, 16, 150)                          # right knee interior 150 vs TEST reference 170-180 -> 20 deg too bent
    assert abs(compute_angles(kp)["right_knee"] - 150) < 1e-9
    r = run(kp); assert len(r.items) == 1, r.items; f = r.items[0]
    assert "right knee" in f.text.lower() and "left" not in f.text.lower() and f.text.lower().startswith("straighten"), f.text
    assert f.side == "right" and f.angle == "right_knee" and f.severity == "moderate" and abs(f.deviation + 20) < 1e-9 and "20" in f.text

def test_left_knee_fault_says_left():
    r = run(set_interior(person(), 11, 13, 15, 150)); assert "left knee" in r.items[0].text.lower() and "right" not in r.items[0].text.lower()

def test_above_range_uses_the_opposite_advice_and_severity_tiers():
    refs = {"right_knee": Reference(100, 150)}                            # TEST: a pose that wants a bent knee; the fixture leg is straight (180) -> 30 above
    f = run(person(), refs).items[0]; assert f.text.lower().startswith("bend your right knee") and f.severity == "major" and abs(f.deviation - 30) < 1e-9
    assert [severity_of(d) for d in (0.1, 9.99, 10, 24.99, 25, -30)] == ["minor", "minor", "moderate", "moderate", "major", "major"]   # tier boundaries

def test_mirrored_input_flips_the_named_side():
    kp = set_interior(person(), 12, 14, 16, 150)                          # fault on the backbone's RIGHT knee
    assert "left knee" in with_lr(True, True, lambda: run(kp).items[0].text.lower())    # mirrored feed: the user's actual LEFT knee
    assert "right knee" in with_lr(True, False, lambda: run(kp).items[0].text.lower())

def test_all_four_lr_flag_combinations():
    exp = {(True, False): "right", (True, True): "left", (False, False): "left", (False, True): "right"}   # (backbone_left_is_anatomical, input_mirrored) -> user's side
    for (a, m), want in exp.items(): assert with_lr(a, m, lambda: lr.anatomical("right")) == want, (a, m)
    assert lr.anatomical(None) is None

def test_unavailable_is_never_a_fault():
    kp = set_interior(person(), 12, 14, 16, 150); kp[16, 2] = 0.3         # right ankle not confident -> right knee angle unavailable although the leg is badly bent
    r = run(kp); assert r.all_items == () and "right_knee" in r.unavailable and r.message == "Good form - hold it."
    assert 0 < r.coverage < 1 and abs(r.coverage - 3 / 4) < 1e-9          # 3 of 4 referenced angles were judgeable

def test_nothing_visible_gives_a_cant_see_message_not_good_form():
    kp = person(); kp[:, 2] = 0.1; r = run(kp); assert r.coverage == 0 and "can't see" in r.message.lower() and r.all_items == ()

def test_ordering_and_top_k():
    kp = set_interior(set_interior(person(), 12, 14, 16, 140), 11, 13, 15, 155)   # right knee 30 off (major), left knee 15 off (moderate; avoids the float-fragile tier boundary)
    r = run(kp, top_k=1); assert len(r.items) == 1 and len(r.all_items) == 2 and r.items[0].angle == "right_knee" and r.items[0].severity == "major"
    assert [f.angle for f in run(kp).items] == ["right_knee", "left_knee"]

def test_temporal_fusion_rules():
    kp = set_interior(person(), 12, 14, 16, 165)                          # 5 deg off -> geometric 'minor'
    base = run(kp); assert base.items[0].severity == "minor" and base.items[0].source == "geometric"
    esc = run(kp, temporal=[TemporalFlag("right_knee", 3.2)]).items[0]; assert esc.severity == "moderate" and esc.source == "geometric+temporal"
    assert run(kp, temporal=None) == base                                  # temporal=None IS the geometric-only ablation arm
    ok = run(person(), temporal=[TemporalFlag("left_knee", 4.0)]).items   # pose in range, trajectory drifts -> low-severity hold-steady note
    assert len(ok) == 1 and ok[0].severity == "minor" and ok[0].source == "temporal" and "left knee" in ok[0].text.lower() and "steady" in ok[0].text.lower()
    bad = person(); bad[16, 2] = 0.3   # temporal flag on an UNAVAILABLE angle is ignored
    assert run(bad, temporal=[TemporalFlag("right_knee", 9.0)]).all_items == ()
    top = run(set_interior(person(), 12, 14, 16, 140), temporal=[TemporalFlag("right_knee", 3.0)]).items[0]; assert top.severity == "major"   # cannot exceed the top tier

def test_side_logic_lives_only_in_lr_py():
    pkg = Path(__file__).resolve().parent.parent / "poseguard"; flags = ("INPUT_MIRRORED", "BACKBONE_LEFT_IS_ANATOMICAL_LEFT")
    for f in pkg.glob("*.py"):
        if f.name in ("lr.py",): continue
        assert not any(x in f.read_text(encoding="utf-8") for x in flags), f"{f.name} must not touch the L/R flags"
    src = (pkg / "feedback.py").read_text(encoding="utf-8")
    assert not re.search(r"""["'](left|right)["']""", src), "feedback.py must not hard-code a side literal"

if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"): f(); print("ok ", n)
    print("all feedback tests passed")
