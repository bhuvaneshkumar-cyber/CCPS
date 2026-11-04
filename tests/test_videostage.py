"""Video-stage helper tests: hand-computed segment arithmetic (TEST fixtures) + the real annotated marks of the two force-closed clips. Run: python tests/test_videostage.py"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from poseguard.videostage import bands_from_frames, crossfit_pairs, frame_table, held_segments

FPS = 30.0
def test_trim_and_merge_hand_computed():
    # 1: 100-400 (10 s) -> trim 45 frames each end -> (145, 355); the two zeros merge; 2: 500-900 (13.3 s) -> (545, 855)
    assert held_segments({100: 1, 400: 0, 430: 0, 500: 2, 900: 0}, 1000, FPS) == [(145, 355, 1), (545, 855, 2)]

def test_short_and_boundary_segments():
    assert held_segments({100: 1, 200: 0}, 1000, FPS) == []                              # 3.33 s < 4 s: dropped
    assert held_segments({0: 1, 120: 0}, 1000, FPS) == [(45, 75, 1)]                     # exactly 4.0 s is kept; 1 s remains after trimming
    assert held_segments({0: 1, 119: 0}, 1000, FPS) == []                                # 3.97 s: dropped

def test_repeated_press_merges_and_junk_mark_on_last_frame_drops():
    assert held_segments({100: 3, 150: 3, 500: 0}, 1000, FPS) == [(145, 455, 3)]         # same label pressed twice = one hold
    assert held_segments({100: 4, 800: 0, 1893: 5}, 1894, FPS) == [(145, 755, 4)]        # step 5 marked on the very last frame: 1 frame, dropped
    assert held_segments({100: 3, 300: 0, 600: 3, 900: 0}, 1000, FPS) == [(145, 255, 3), (645, 855, 3)]   # a genuine repeat (separated by 0) stays two holds

def test_string_keys_as_stored_in_json():
    assert held_segments({"100": 1, "400": 0}, 1000, FPS) == held_segments({100: 1, 400: 0}, 1000, FPS)

def test_real_force_closed_clips_have_no_degenerate_segments():
    a = json.load(open(Path(__file__).resolve().parent.parent / "audit" / "step_annotations_Bhuvanesh.json"))["videos"]
    for k, n in (("Videos/Ardhakati Chakrasana/Ardhakati Chakrasana Right Steps/Ardhakati Chakrasana Right Step Angle 2-converted.mp4", 2078),
                 ("Videos/Tadasana/Tadasana Wrong Steps/Tadasana Wrong Steps Angle 1-converted.mp4", 1894)):
        segs = held_segments(a[k]["marks"], n, FPS); assert segs and all(e - s >= 30 for s, e, _ in segs), (k, segs)

def test_frame_table_keeps_only_trimmed_held_frames():
    kp = np.zeros((6, 17, 3)); kp[:, :, 2] = 0.9; kp[:, :2, :2] = 1.0; fr = [50, 100, 200, 300, 350, 400]     # marks: step 1 held 50-400 -> trimmed (95, 355)
    t = frame_table(dict(asana="Tadasana", label="Right", angle=1), kp, fr, {50: 1, 400: 0}, 500, FPS, {("Tadasana", 1): 1})
    assert list(t.frame) == [100, 200, 300, 350] and set(t.step) == {1} and set(t.pclass) == {1} and t.hold.nunique() == 1

def test_crossfit_single_hold_gap_hand_computed():
    f = np.arange(0, 600, 6)                                                             # 100 sampled frames (20 s at 5 fps), midpoint 297, half-gap 22.5 frames
    (ref, test), = crossfit_pairs({0: f}, FPS)
    assert ref.max() < 297 - 22.5 and test.min() > 297 + 22.5 and len(ref) == 46 and len(test) == 46 and test.min() - ref.max() >= 45   # >= 1.5 s (45 frames) apart

def test_crossfit_two_holds_are_swapped():
    a, b = np.arange(0, 120, 6), np.arange(300, 420, 6); pairs = crossfit_pairs({0: a, 1: b}, FPS)
    assert len(pairs) == 2 and (pairs[0][0] == b).all() and (pairs[0][1] == a).all() and (pairs[1][0] == a).all() and (pairs[1][1] == b).all()

def test_video_bands_need_min_frames():
    from poseguard.phase3 import ANGLE_NAMES
    mk = lambda n: pd.DataFrame({a: np.linspace(0, 100, n) for a in ANGLE_NAMES})
    assert set(bands_from_frames(mk(10))) == set(ANGLE_NAMES) and bands_from_frames(mk(9)) == {}
    b = bands_from_frames(mk(101))["left_knee"]; assert (b.lo, b.hi) == (15 - 5, 85 + 5)   # P15 / P85 of 0..100 widened by 5

if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"): f(); print("ok ", n)
    print("all videostage tests passed")
