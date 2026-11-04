"""Phase 0/1: run the pinned pretrained backbone on real dataset images, save per-image keypoints.
usage: python audit/extract_kpts.py <imgsz> <out.npz> [sample_n_per_asana_label]
Stores the highest-confidence person only (+ number of detected persons); no image is skipped silently:
images with no detection get NaN keypoints and n_person=0."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard.config import BACKBONE_WEIGHTS, DATA_DIR, ROOT

imgsz, out = int(sys.argv[1]), ROOT / "audit" / sys.argv[2]
d = pd.read_csv(ROOT / "audit" / "inventory_images.csv")
if len(sys.argv) > 3:  # stratified sample, fixed seed
    d = d.groupby(["asana", "label"], group_keys=False).apply(lambda g: g.sample(min(len(g), int(sys.argv[3])), random_state=0))
B = DATA_DIR / "raw" / "Yoga Postures Dataset"
m = YOLO(str(BACKBONE_WEIGHTS))
kp = np.full((len(d), 17, 3), np.nan, np.float32); bconf = np.zeros(len(d), np.float32)
box = np.full((len(d), 4), np.nan, np.float32); npers = np.zeros(len(d), np.int16)
paths = [str(B / p) for p in d.path]
for i in range(0, len(paths), 32):
    for j, r in enumerate(m.predict(paths[i:i + 32], imgsz=imgsz, verbose=False, device="cpu", batch=32)):
        k = i + j; npers[k] = len(r.boxes)
        if npers[k]:
            t = int(r.boxes.conf.argmax())
            kp[k] = r.keypoints.data[t].numpy(); bconf[k] = float(r.boxes.conf[t]); box[k] = r.boxes.xyxy[t].numpy()
    if (i // 32) % 20 == 0: print(i, len(paths), flush=True)
np.savez(out, path=d.path.values, kp=kp, bconf=bconf, box=box, npers=npers, imgsz=imgsz)
print("saved", out, "no-detection:", int((npers == 0).sum()))
