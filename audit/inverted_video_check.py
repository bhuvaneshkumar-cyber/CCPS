"""Phase 0 follow-up: does the backbone also fail on the inverted-pose VIDEOS (fixed camera), or only on the stills?
20 evenly spaced frames per video for Sarvangasana + Viparita Karani (all 16 videos), Tadasana angle 1/2 Right as a standing reference.
Reports detection rate and joint-chain coverage at tau, and saves an overlay montage for visual sanity (confident-but-wrong skeletons are not visible in coverage numbers)."""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard.config import BACKBONE_WEIGHTS, DATA_DIR, KPT_CONF_TAU as TAU, ROOT

B = DATA_DIR / "raw" / "Yoga Postures Dataset"
vid = pd.read_csv(ROOT / "audit" / "inventory_videos.csv")
model = YOLO(str(BACKBONE_WEIGHTS))
CH = {"L_arm": [5, 7, 9], "R_arm": [6, 8, 10], "L_leg": [11, 13, 15], "R_leg": [12, 14, 16], "trunk": [5, 6, 11, 12]}
SK = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
rows, shots = [], {}
sel = vid[vid.asana.isin(["Sarvangasana", "Viparita Karani"]) | ((vid.asana == "Tadasana") & (vid.label == "Right") & vid.angle.isin([1, 2]))]
for _, r in sel.iterrows():
    cap = cv2.VideoCapture(str(B / r.path)); frames = []
    for f in np.linspace(30, r.frames - 30, 20).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, fr = cap.read(); frames.append(fr)
    res = model.predict(frames, imgsz=640, verbose=False, device="cpu")
    for j, x in enumerate(res):
        det = len(x.boxes) > 0
        k = x.keypoints.data[int(x.boxes.conf.argmax())].numpy() if det else np.zeros((17, 3))
        row = dict(asana=r.asana, label=r.label, angle=r.angle, det=det, **{c: bool(det and (k[i, 2] >= TAU).all()) for c, i in CH.items()})
        rows.append(row)
        if j == 10 and r.label == "Right" and r.asana != "Tadasana":  # one overlay per Right video, mid-clip
            im = frames[j].copy()
            for u, v in SK:
                if k[u, 2] >= TAU and k[v, 2] >= TAU: cv2.line(im, tuple(k[u, :2].astype(int)), tuple(k[v, :2].astype(int)), (255, 200, 0), 4)
            for px, py, c in k: cv2.circle(im, (int(px), int(py)), 6, (0, 255, 0) if c >= TAU else (0, 0, 255), -1)
            s = 480 / max(im.shape[:2]); im = cv2.resize(im, None, fx=s, fy=s)
            cv2.putText(im, f"{r.asana} ang{r.angle}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            shots[(r.asana, r.angle)] = cv2.copyMakeBorder(im, 0, 480 - im.shape[0], 0, 480 - im.shape[1], cv2.BORDER_CONSTANT)
d = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(d.groupby(["asana", "label"])[["det", *CH]].mean().round(2).to_string())
print(d.groupby(["asana", "angle"])[["det", *CH]].mean().round(2).to_string())
rowsimg = [np.hstack([shots[(a, g)] for g in (1, 2, 3, 4)]) for a in ("Sarvangasana", "Viparita Karani")]
cv2.imwrite(str(ROOT / "audit" / "inverted_video_overlay.jpg"), cv2.resize(np.vstack(rowsimg), None, fx=0.6, fy=0.6))
