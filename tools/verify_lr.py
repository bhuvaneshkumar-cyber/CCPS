"""Empirical left/right verification (spec Phase 5). Two modes:
  --dataset            objective check on the real Mendeley videos: standing asanas (Tadasana, Ardhakati Chakrasana), angle 1 (subject FRONT-facing the
                       fixed camera) and angle 3 (subject facing AWAY). If 'left_*' is the subject's left, then facing the camera left_shoulder has
                       LARGER x than right_shoulder (image right), facing away SMALLER x. Uses audit/kpts_video_640.npz.
  --video P --raised S run the backbone on YOUR recording in which you raised one KNOWN arm S (left|right, your anatomical side) and report which
                       backbone-named side was detected raised -> tells you whether INPUT_MIRRORED must be True for that camera source.
usage: python tools/verify_lr.py --dataset      |      python tools/verify_lr.py --video clip.mp4 --raised left"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poseguard.config import KPT_CONF_TAU as TAU

def dataset():
    """View is classified PER FRAME, never from the 'Angle n' number (that number is not a reliable view label: e.g. Tadasana Wrong swaps 3 and 4).
    Criteria fixed before running: Tadasana + Ardhakati Chakrasana only (upright, no twist); torso span ratio >= 0.4 (excludes profile);
    facing CAMERA = mean conf(nose, both eyes) >= 0.6, facing AWAY = <= 0.2 (ambiguous middle dropped). Pass: fraction >= 0.95 with >= 100 frames per side."""
    vid = pd.read_csv(ROOT / "audit" / "inventory_videos.csv"); kp = np.load(ROOT / "audit" / "kpts_video_640.npz", allow_pickle=True)["kp"]
    v = vid[vid.asana.isin(["Tadasana", "Ardhakati Chakrasana"])]; k = kp[v.index].reshape(-1, 17, 3); vi = np.repeat(v.index, kp.shape[1])
    k = np.nan_to_num(k); m = k[:, [5, 6, 11, 12], 2].min(1) >= TAU; k, vi = k[m], vi[m]
    ratio = np.linalg.norm(k[:, 5, :2] - k[:, 6, :2], axis=1) / np.linalg.norm(k[:, [5, 6], :2].mean(1) - k[:, [11, 12], :2].mean(1), axis=1); face = k[:, 0:3, 2].mean(1)
    front, back = (ratio >= 0.4) & (face >= 0.6), (ratio >= 0.4) & (face <= 0.2)
    print(f"tau={TAU}; per-frame view classification (see docstring); test on shoulders (left idx 5, right idx 6)")
    ok_all = True
    for name, sel, want, cond in (("FACING CAMERA", front, "left_shoulder.x > right_shoulder.x", lambda a: a[:, 5, 0] > a[:, 6, 0]),
                                  ("FACING AWAY", back, "left_shoulder.x < right_shoulder.x", lambda a: a[:, 5, 0] < a[:, 6, 0])):
        frac = cond(k[sel]).mean(); n = int(sel.sum()); print(f"{name}: {n} frames from {len(np.unique(vi[sel]))} videos; fraction satisfying '{want}' = {frac:.3f}")
        bad = np.unique(vi[sel][~cond(k[sel])]); print("   videos with violations:", [vid.path[b].split(chr(92))[-1][:45] for b in bad] or "none"); ok_all &= (frac >= 0.95 and n >= 100)
    print("VERDICT:", "backbone 'left_*' is the subject's anatomical left (COCO convention) on non-mirrored input" if ok_all else "NOT CONFIRMED - do not rely on BACKBONE_LEFT_IS_ANATOMICAL_LEFT")

def video(path, raised, maxf=1200):
    import cv2
    from ultralytics import YOLO
    from poseguard.config import BACKBONE_WEIGHTS
    m = YOLO(str(BACKBONE_WEIGHTS)); cap = cv2.VideoCapture(path); votes = {"left": 0, "right": 0}; n = 0
    while True:
        ok, fr = cap.read()
        if not ok or n >= maxf: break
        n += 1
        if n % 5: continue
        r = m.predict(fr, imgsz=640, verbose=False)[0]
        if not len(r.boxes): continue
        k = r.keypoints.data[int(r.boxes.conf.argmax())].numpy()
        if k[[5, 6, 9, 10, 11, 12], 2].min() < TAU: continue
        torso = np.linalg.norm(k[[5, 6], :2].mean(0) - k[[11, 12], :2].mean(0)); up = lambda w, s: k[w, 1] < k[s, 1] - 0.3 * torso   # wrist clearly above shoulder (y grows down)
        l, rr = up(9, 5), up(10, 6)
        if l != rr: votes["left" if l else "right"] += 1
    print(f"frames with exactly one arm raised: {votes}")
    if sum(votes.values()) < 10: sys.exit("too few usable frames (need >=10): raise ONE arm clearly above the shoulder, full upper body in view")
    seen = max(votes, key=votes.get); print(f"you raised your {raised}; the backbone named the raised arm '{seen}_*'")
    print("OK: names match -> keep INPUT_MIRRORED = False for this source" if seen == raised else
          "MISMATCH: this source is mirrored (or the arm was misstated) -> set INPUT_MIRRORED = True in poseguard/lr.py for this camera source, re-run to confirm")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", action="store_true"); ap.add_argument("--video"); ap.add_argument("--raised", choices=["left", "right"]); ap.add_argument("--max-frames", type=int, default=1200); a = ap.parse_args()
    if a.dataset: dataset()
    elif a.video and a.raised: video(a.video, a.raised, a.max_frames)
    else: ap.print_help()
