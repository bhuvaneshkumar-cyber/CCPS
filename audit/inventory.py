"""Phase 0: build the label inventory purely from the on-disk folder names + real file metadata.
Outputs audit/inventory_images.csv, audit/inventory_videos.csv, and prints summary tables."""
import re, sys
from pathlib import Path
import cv2, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poseguard.config import DATA_DIR, ROOT

BASE = DATA_DIR / "raw" / "Yoga Postures Dataset"
OUT = ROOT / "audit"

# Image folder: <Asana>/<Asana> {Right|Wrong} Steps/<Asana> {Right|Wrong} Step <k>[- <fault>]
rows = []
for p in sorted((BASE / "Images").rglob("*.jpg")):
    asana, grp, leaf = p.relative_to(BASE / "Images").parts[:3]
    m = re.search(r"(Right|Wrong) Step (\d+)\s*-?\s*(.*)$", leaf)
    assert m, leaf
    label, step, fault = m.group(1), int(m.group(2)), m.group(3).strip()
    ts = re.match(r"(\d{8})_(\d{6})", p.stem)  # camera timestamp YYYYMMDD_HHMMSS
    im = cv2.imread(str(p))
    h, w = im.shape[:2]
    rows.append(dict(path=str(p.relative_to(BASE)), asana=asana, label=label, step=step, fault=fault,
                     date=ts.group(1) if ts else "", time=ts.group(2) if ts else "", w=w, h=h))
img = pd.DataFrame(rows)
img.to_csv(OUT / "inventory_images.csv", index=False)

vrows = []
for p in sorted((BASE / "Videos").rglob("*.mp4")):
    asana, grp = p.relative_to(BASE / "Videos").parts[:2]
    a = re.search(r"Angle (\d)", p.name)
    cap = cv2.VideoCapture(str(p))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    vrows.append(dict(path=str(p.relative_to(BASE)), asana=asana, label="Right" if "Right" in grp else "Wrong",
                      angle=int(a.group(1)), frames=n, fps=round(fps, 2), dur_s=round(n / fps, 1),
                      w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    cap.release()
vid = pd.DataFrame(vrows)
vid.to_csv(OUT / "inventory_videos.csv", index=False)

pd.set_option("display.width", 200, "display.max_rows", 200)
print("IMAGES total", len(img), "| sizes:", img.groupby(["w", "h"]).size().to_dict())
print("\nper asana x label\n", img.pivot_table(index="asana", columns="label", values="path", aggfunc="count", margins=True))
print("\nper asana x step x label\n", img.pivot_table(index=["asana", "step"], columns="label", values="path", aggfunc="count", fill_value=0))
print("\nfault tags (Wrong)\n", img[img.label == "Wrong"].groupby(["asana", "fault"]).size())
print("\nimage capture dates\n", img.groupby("date").size())
print("\nVIDEOS", len(vid), "\n", vid.groupby(["w", "h", "fps"]).size(), "\n", vid.groupby("label")[["frames", "dur_s"]].agg(["sum", "min", "max"]))
print(vid.to_string())
