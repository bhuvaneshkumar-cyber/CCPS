"""Phase 0: is the 256x256 still-image set anisotropically resized? Frontal-view proxy r = shoulder-span / shoulder-mid-to-hip-mid
distance (2D) for Tadasana. Videos (native 720x1280, fixed camera, angle 1 = front) vs stills (320px extraction, usable only).
Horizontal stretch in stills would inflate r for frontal-view images relative to the video."""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard.config import BACKBONE_WEIGHTS, DATA_DIR, KPT_CONF_TAU as TAU, ROOT

B = DATA_DIR / "raw" / "Yoga Postures Dataset"
def ratio(k):  # k: 17x3
    if (k[[5, 6, 11, 12], 2] < TAU).any(): return np.nan
    return np.linalg.norm(k[5, :2] - k[6, :2]) / np.linalg.norm(k[[5, 6], :2].mean(0) - k[[11, 12], :2].mean(0))

z = np.load(ROOT / "audit" / "kpts_320_all.npz", allow_pickle=True)
man = pd.read_csv(ROOT / "audit" / "manifest_dedup.csv")
m = pd.DataFrame({"path": z["path"]}).merge(man[["path", "asana", "usable"]], on="path")
sel = np.where((m.asana == "Tadasana") & m.usable)[0]
ri = np.array([ratio(z["kp"][i]) for i in sel]); ri = ri[~np.isnan(ri)]
print(f"stills Tadasana: n={len(ri)}  r pct[10,50,75,90,95]={np.percentile(ri, [10, 50, 75, 90, 95]).round(2)}")

model = YOLO(str(BACKBONE_WEIGHTS)); vid = pd.read_csv(ROOT / "audit" / "inventory_videos.csv")
for ang, name in ((1, "front"), (3, "back")):
    v = vid[(vid.asana == "Tadasana") & (vid.angle == ang)]
    rv = []
    for _, r in v.iterrows():
        cap = cv2.VideoCapture(str(B / r.path)); frames = []
        for f in np.linspace(0, r.frames - 2, 30).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, fr = cap.read(); frames.append(fr)
        for res in model.predict(frames, imgsz=640, verbose=False, device="cpu"):
            if len(res.boxes): rv.append(ratio(res.keypoints.data[int(res.boxes.conf.argmax())].numpy()))
    rv = np.array(rv); rv = rv[~np.isnan(rv)]
    print(f"video Tadasana angle{ang} ({name}), Right+Wrong: n={len(rv)}  r pct[10,50,75,90,95]={np.percentile(rv, [10, 50, 75, 90, 95]).round(2)}")
