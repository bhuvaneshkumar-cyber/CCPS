"""Phase 2 unit tests. Every expected value is HAND-COMPUTED (comment shows the arithmetic). Hand-built keypoints are explicit TEST fixtures,
not data of any kind used for training/evaluation. Run: python tests/test_angles.py   (or pytest)"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard.angles import ANGLE_BY_NAME, compute_angles, interior_angle, shoulder_line_tilt, trunk_tilt

def close(a, b, tol=1e-6): return a is not None and abs(a - b) < tol

def test_interior_angle_hand_computed():
    assert close(interior_angle((0, -1), (0, 0), (0, 1)), 180)                       # collinear, vertex in the middle
    assert close(interior_angle((0, 3), (0, 0), (4, 0)), 90)                         # 3-4-5 right triangle, vertex at the right angle
    assert close(interior_angle((0, 0), (0, 3), (4, 0)), np.degrees(np.arccos(0.6)))  # at a: cos = 9/(3*5) = 0.6 -> 53.1301 deg
    assert close(interior_angle((0, 3), (4, 0), (0, 0)), np.degrees(np.arctan2(3, 4)))  # at c: tan = 3/4 -> 36.8699 deg
    assert close(interior_angle((0, 0), (1, 0), (0.5, np.sqrt(3) / 2)), 60)          # equilateral
    assert close(interior_angle((1, 0), (0, 0), (-1, 1e-12)), 180, 1e-6)             # near-collinear must clip, not NaN
    assert interior_angle((5, 5), (5, 5), (1, 1)) is None                            # zero-length ray

def test_angle_is_symmetric_in_its_ends():
    assert close(interior_angle((0, 3), (0, 0), (4, 0)), interior_angle((4, 0), (0, 0), (0, 3)))

def test_trunk_tilt_hand_computed():
    assert close(trunk_tilt((-50, 0), (50, 0), (-50, 100), (50, 100)), 0)            # shoulders straight above hips (y grows DOWN): upright
    assert close(trunk_tilt((50, 0), (150, 0), (-50, 100), (50, 100)), 45)           # shoulder-mid (100,0), hip-mid (0,100): vector (100,-100) -> 45 deg
    assert close(trunk_tilt((50, 100), (150, 100), (-50, 100), (50, 100)), 90)       # torso horizontal
    assert close(trunk_tilt((-50, 200), (50, 200), (-50, 100), (50, 100)), 180)      # shoulders below hips: inverted

def test_shoulder_line_tilt_hand_computed():
    assert close(shoulder_line_tilt((0, 0), (100, 0)), 0)
    assert close(shoulder_line_tilt((0, 0), (100, 100 * np.tan(np.radians(10)))), 10)
    assert close(shoulder_line_tilt((100, 0), (0, 100 * np.tan(np.radians(10)))), 10)  # order of the two shoulders must not change the (unsigned) result
    assert close(shoulder_line_tilt((0, 0), (0, 100)), 90)

def _kp(**conf_over):
    kp = np.zeros((17, 3)); kp[:, 2] = 0.9
    for i, (x, y) in {5: (-50, 0), 6: (50, 0), 11: (-40, 100), 12: (40, 100), 13: (-40, 200), 14: (40, 200), 15: (-40, 300), 16: (40, 300), 7: (-50, 100), 8: (50, 100), 9: (-50, 200), 10: (50, 200)}.items(): kp[i, :2] = (x, y)
    for i, c in conf_over.items(): kp[int(i[1:]), 2] = c
    return kp

def test_side_and_index_mapping():
    assert ANGLE_BY_NAME["left_knee"].idx == (11, 13, 15) and ANGLE_BY_NAME["right_knee"].idx == (12, 14, 16)
    assert ANGLE_BY_NAME["left_elbow"].idx == (5, 7, 9) and ANGLE_BY_NAME["right_hip"].idx == (6, 12, 14)
    assert ANGLE_BY_NAME["left_knee"].side == "left" and ANGLE_BY_NAME["right_knee"].side == "right" and ANGLE_BY_NAME["trunk_tilt"].side is None
    assert "neck" not in " ".join(ANGLE_BY_NAME) and "head" not in " ".join(ANGLE_BY_NAME)   # head/neck are excluded on purpose

def test_straight_standing_fixture():
    a = compute_angles(_kp())
    assert close(a["left_knee"], 180) and close(a["right_knee"], 180) and close(a["left_elbow"], 180)
    assert close(a["trunk_tilt"], 0) and close(a["shoulder_line_tilt"], 0)
    # left hip: shoulder(-50,0) - hip(-40,100) - knee(-40,200): rays (-10,-100) and (0,100); cos = -10000/(sqrt(10100)*100) -> 180 - atan(10/100) = 174.2894 deg
    assert close(a["left_hip"], 180 - np.degrees(np.arctan(0.1)))

def test_confidence_gating_boundary_and_isolation():
    a = compute_angles(_kp(k15=0.49))                       # left ankle just below tau=0.5
    assert a["left_knee"] is None and a["right_knee"] is not None and a["left_hip"] is not None   # only angles using that keypoint become unavailable
    assert compute_angles(_kp(k15=0.5))["left_knee"] is not None                                   # exactly tau passes (>=)
    kp = _kp(); kp[13, 0] = np.nan; assert compute_angles(kp)["left_knee"] is None                 # non-finite coordinate is unavailable, not an error
    assert all(v is None for v in compute_angles(np.zeros((17, 3))).values())                       # nothing confident -> everything unavailable

def test_still_stretch_correction():
    # true leg: hip (0,0), knee (0,100), ankle (100,200): rays (0,-100) and (100,100): cos = -10000/(100*141.42) = -0.7071 -> 135 deg.
    k = 1.65; kp = np.zeros((17, 3)); kp[:, 2] = 0.9; kp[12], kp[14], kp[16] = (0, 0, .9), (0, 100, .9), (100 * k, 200, .9)   # x stretched by k as in the stills
    assert close(compute_angles(kp, stretch_k=k)["right_knee"], 135, 1e-9)         # corrected -> the true angle
    assert abs(compute_angles(kp)["right_knee"] - 135) > 5                         # uncorrected is visibly wrong (this is why the correction exists)

if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"): f(); print("ok ", n)
    print("all angle tests passed")
