"""Phase 0 follow-up (distortion factor): backbone keypoints on all 80 real videos, 40 evenly spaced frames each
(native 720x1280 / 1280x720 pixel coordinates = undistorted reference). imgsz=640 (video is the high-quality source).
Also stores the frame index so later scripts can re-decode neighbouring frames."""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard.config import BACKBONE_WEIGHTS, DATA_DIR, ROOT

N = 40
B = DATA_DIR / "raw" / "Yoga Postures Dataset"
vid = pd.read_csv(ROOT / "audit" / "inventory_videos.csv")
model = YOLO(str(BACKBONE_WEIGHTS))
kp = np.full((len(vid), N, 17, 3), np.nan, np.float32); fidx = np.zeros((len(vid), N), np.int32)
for i, r in vid.iterrows():
    fidx[i] = np.linspace(30, r.frames - 60, N).astype(int)
    cap = cv2.VideoCapture(str(B / r.path)); frames = []
    for f in fidx[i]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, fr = cap.read(); assert ok, (r.path, f); frames.append(fr)
    for j, x in enumerate(model.predict(frames, imgsz=640, verbose=False, device="cpu")):
        if len(x.boxes): kp[i, j] = x.keypoints.data[int(x.boxes.conf.argmax())].numpy()
    print(i, r.asana, r.label, r.angle, flush=True)
np.savez(ROOT / "audit" / "kpts_video_640.npz", kp=kp, fidx=fidx, path=vid.path.values)
print("saved; frames with no detection:", int(np.isnan(kp[:, :, 0, 0]).sum()))
