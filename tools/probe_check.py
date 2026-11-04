"""Reproducibility probes: run the backbone on the real frames recorded in PROBES.json (by tools/make_release.py on the SOURCE machine) and compare keypoints.
usage (from the extracted project root): python tools/probe_check.py [--gpu]     tolerance: 0.05 px on the same device type, 3 px with --gpu (GPU vs CPU numerics differ)"""
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poseguard import config as C
from poseguard.backbone import Backbone, primary

gpu = "--gpu" in sys.argv; tol = 3.0 if gpu else 0.05; bb = Backbone(device="auto" if gpu else "cpu"); worst, n = 0.0, 0
for p in json.load(open(ROOT / "PROBES.json")):
    if p["kind"] == "still": im = cv2.imread(str(C.media(p["path"])))
    else:
        cap = cv2.VideoCapture(str(C.media(p["path"]))); cap.set(cv2.CAP_PROP_POS_FRAMES, p["frame"]); ok, im = cap.read(); assert ok, p
    q = primary(bb(im)); ref = None if p["kp"] is None else np.array(p["kp"])
    if ref is None or q is None:
        assert ref is None and q is None, f"detection differs on {p['path']}"; continue
    m = (ref[:, 2] >= C.KPT_CONF_TAU) & (q.kp[:, 2] >= C.KPT_CONF_TAU); worst = max(worst, float(np.abs(q.kp[m, :2] - ref[m, :2]).max())); n += 1
print(f"{n} real frames; worst keypoint difference {worst:.4f} px (tolerance {tol})"); sys.exit(0 if worst <= tol else 1)
