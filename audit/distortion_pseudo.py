"""ESTIMATOR VALIDATION ONLY (not training/evaluation data, never reported as a system result).
Real video frames given a KNOWN geometric transform, then run through the real backbone, so distortion.py can be checked
against a known ground-truth k:
  squash: resize to 256x256 like the stills -> relative x-stretch k_true = H/W = 1280/720 = 1.778 (portrait videos only)
  null  : uniform resize to width 256 (aspect kept)           -> k_true = 1.0
Frames are taken 20 frames after the reference frames used by distortion.py (never the identical frame)."""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard.config import BACKBONE_WEIGHTS, DATA_DIR, ROOT

B = DATA_DIR / "raw" / "Yoga Postures Dataset"
vid = pd.read_csv(ROOT / "audit" / "inventory_videos.csv"); ref = np.load(ROOT / "audit" / "kpts_video_640.npz", allow_pickle=True)
model = YOLO(str(BACKBONE_WEIGHTS)); OFF = 20; KEEP = slice(0, None, 2)  # 20 of the 40 reference positions
out = {c: dict(kp=[], vi=[]) for c in ("squash", "null")}
for i, r in vid.iterrows():
    if r.asana in ("Sarvangasana", "Viparita Karani") or r.h < r.w: continue  # portrait only, spec-excluded asanas skipped
    cap = cv2.VideoCapture(str(B / r.path)); fr = []
    for f in ref["fidx"][i][KEEP]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f) + OFF); ok, im = cap.read(); assert ok; fr.append(im)
    for cond, tf, sz in (("squash", lambda im: cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA), 320),
                         ("null", lambda im: cv2.resize(im, (256, round(256 * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA), 448)):
        for x in model.predict([tf(im) for im in fr], imgsz=sz, verbose=False, device="cpu"):
            out[cond]["kp"].append(x.keypoints.data[int(x.boxes.conf.argmax())].numpy() if len(x.boxes) else np.full((17, 3), np.nan, np.float32))
            out[cond]["vi"].append(i)
    print(i, r.asana, flush=True)
np.savez(ROOT / "audit" / "kpts_pseudo.npz", **{f"{c}_{k}": np.array(v) for c, d in out.items() for k, v in d.items()})
print("saved")
