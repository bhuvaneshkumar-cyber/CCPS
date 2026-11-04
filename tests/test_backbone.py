"""Phase 1 tests. The model tests run the REAL pinned backbone on REAL Mendeley video frames. Two edge-case fixtures are explicitly test-only:
a blank image (asserts 'nobody found' is handled) and hand-built Person objects (asserts the person-selection rule). Run: python tests/test_backbone.py"""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard import config as C
from poseguard.backbone import Backbone, Person, pick_device, primary, sha256_of

VID = pd.read_csv(C.ROOT / "audit" / "inventory_videos.csv"); STORED = np.load(C.ROOT / "audit" / "kpts_video_640.npz", allow_pickle=True)
_bb = None

def bb():
    global _bb
    if _bb is None: _bb = Backbone(device="cpu")
    return _bb

def frame(vi, fi):
    cap = cv2.VideoCapture(str(C.media(VID.path[vi]))); cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read(); assert ok; return f

def test_weights_match_pinned_hash():
    assert sha256_of(C.BACKBONE_WEIGHTS) == C.BACKBONE_SHA256

def test_unpinned_weights_are_refused():
    old = C.BACKBONE_SHA256; C.BACKBONE_SHA256 = "0" * 64
    try: Backbone(device="cpu"); raise AssertionError("an unpinned backbone was accepted")
    except RuntimeError as e: assert "SHA-256" in str(e)
    finally: C.BACKBONE_SHA256 = old

def test_primary_selection_rule():   # TEST fixture: hand-built persons
    mk = lambda c: Person(np.zeros((17, 3)), np.zeros(4), c)
    assert primary([]) is None and primary([mk(.3), mk(.9), mk(.5)]).conf == .9

def test_real_frame_output_is_well_formed():
    vi = VID[(VID.asana == "Tadasana") & (VID.label == "Right") & (VID.angle == 1)].index[0]; f = frame(vi, 900); h, w = f.shape[:2]
    p = primary(bb()(f)); assert p is not None and p.kp.shape == (17, 3) and p.box.shape == (4,)
    assert (p.kp[:, 2] >= 0).all() and (p.kp[:, 2] <= 1).all() and 0 < p.conf <= 1
    good = p.kp[p.kp[:, 2] >= C.KPT_CONF_TAU]; assert len(good) >= 10 and (good[:, 0] >= 0).all() and (good[:, 0] <= w).all() and (good[:, 1] >= 0).all() and (good[:, 1] <= h).all()

def test_reproduces_phase0_extraction():
    """The wrapper must give the same keypoints as the audit extraction (same weights, same imgsz, same defaults) on the same real frames."""
    worst = 0.0; conf_worst = 0.0
    for vi in (56, 8, 30):                                   # Tadasana R a1, Ardhakati R a1, Kati R a3 - real videos
        for j in (5, 20, 35):
            p = primary(bb()(frame(vi, STORED["fidx"][vi][j]))); ref = STORED["kp"][vi][j]; assert p is not None and not np.isnan(ref).all()
            m = (ref[:, 2] >= C.KPT_CONF_TAU) & (p.kp[:, 2] >= C.KPT_CONF_TAU)
            worst = max(worst, float(np.abs(p.kp[m, :2] - ref[m, :2]).max())); conf_worst = max(conf_worst, float(np.abs(p.kp[:, 2] - ref[:, 2]).max()))
    print(f"   max keypoint diff vs Phase 0 extraction: {worst:.4f} px, max conf diff {conf_worst:.5f}")
    assert worst < 1.0 and conf_worst < 0.02

def test_blank_frame_means_nobody():   # TEST fixture: a blank image is not data, it checks the empty-result path
    assert primary(bb()(np.zeros((480, 640, 3), np.uint8))) is None

def test_video_iterator_indexing_and_stride():
    vi = VID[(VID.asana == "Tadasana") & (VID.label == "Right") & (VID.angle == 1)].index[0]
    got = [(i, f.shape, len(ps)) for i, f, ps in bb().video(str(C.media(VID.path[vi])), stride=10, max_frames=35)]
    assert [g[0] for g in got] == [0, 10, 20, 30] and len({g[1] for g in got}) == 1

def test_device_selection_is_explicit():
    import torch
    assert pick_device("cpu") == "cpu"
    if not torch.cuda.is_available(): assert pick_device("auto") == "cpu"   # this workstation has no NVIDIA GPU; on the 5080 box it must be cuda:0 after the matmul check
    else: assert pick_device("auto") == "cuda:0"

if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"): f(); print("ok ", n)
    print("all backbone tests passed")
