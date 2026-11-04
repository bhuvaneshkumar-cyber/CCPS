"""Phase 1 verification on REAL data: (1) overlay sheets per asana (4 video angles + 1 Right still + 1 Wrong still) through poseguard.backbone,
(2) video-level detection and per-angle coverage over all 80 videos (from the stored 640-px extraction, which test_backbone proves identical to the wrapper's output),
(3) tau sensitivity of coverage, (4) CPU time per frame of the wrapper (NOT an RTX 5070 number).
usage: python scripts/phase1_verify.py"""
import sys, time
from pathlib import Path
import cv2, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poseguard import config as C
from poseguard.angles import ANGLES, compute_angles
from poseguard.backbone import Backbone, primary
from poseguard.overlay import draw_skeleton

OUT = ROOT / "phase1"; (OUT / "overlays").mkdir(parents=True, exist_ok=True)
BASE = C.DATA_DIR / "raw" / "Yoga Postures Dataset"; vid = pd.read_csv(ROOT / "audit" / "inventory_videos.csv"); man = pd.read_csv(ROOT / "audit" / "manifest_dedup.csv")
bb = Backbone(device="cpu"); S = 400

def tile(im, kp, label):
    h, w = im.shape[:2]; s = S / max(h, w); draw_skeleton(im, kp, scale=1.0, thickness=max(2, int(3 / s))) if kp is not None else None
    im = cv2.resize(im, (int(w * s), int(h * s))); c = np.zeros((S, S, 3), np.uint8); c[:im.shape[0], :im.shape[1]] = im
    cv2.putText(c, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1); return c

# (1) overlays
report = []
for asana in sorted(vid.asana.unique()):
    tiles = []
    for ang in (1, 2, 3, 4):
        r = vid[(vid.asana == asana) & (vid.label == "Right") & (vid.angle == ang)].iloc[0]
        cap = cv2.VideoCapture(str(BASE / r.path)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frames * 0.4)); ok, f = cap.read(); p = primary(bb(f))
        tiles.append(tile(f, None if p is None else p.kp, f"video R ang{ang}" + ("" if p else " NO DET"))); report.append((asana, f"video R ang{ang}", 0 if p is None else int((p.kp[:, 2] >= C.KPT_CONF_TAU).sum())))
    for lab in ("Right", "Wrong"):
        g = man[(man.asana == asana) & (man.label == lab)].sort_values("time"); r = g.iloc[len(g) // 2]
        f = cv2.imread(str(BASE / r.path)); p = primary(bb(f)); tiles.append(tile(f, None if p is None else p.kp, f"still {lab}" + ("" if p else " NO DET")))
        report.append((asana, f"still {lab}", 0 if p is None else int((p.kp[:, 2] >= C.KPT_CONF_TAU).sum())))
    cv2.imwrite(str(OUT / "overlays" / f"{asana.replace(' ', '_')}.jpg"), cv2.resize(np.hstack(tiles), None, fx=0.6, fy=0.6))
print("overlay sheets written:", len(list((OUT / 'overlays').glob('*.jpg'))))
print(pd.DataFrame(report, columns=["asana", "tile", "n_kpts>=tau"]).pivot(index="asana", columns="tile", values="n_kpts>=tau").to_string())

# (2)+(3) video-level detection and coverage from the stored extraction
kp = np.load(ROOT / "audit" / "kpts_video_640.npz", allow_pickle=True)["kp"]           # (80, 40, 17, 3)
det = ~np.isnan(kp[:, :, 0, 0]); print(f"\nVIDEO detection: {det.sum()} / {det.size} frames = {det.mean():.4f}")
rows = []
for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
    A = [[compute_angles(f, tau=tau) if not np.isnan(f[:, 0]).all() else {s.name: None for s in ANGLES} for f in kp[i]] for i in range(len(vid))]
    for i, r in vid.iterrows():
        for s in ANGLES: rows.append((tau, r.asana, r.label, s.name, np.mean([a[s.name] is not None for a in A[i]])))
d = pd.DataFrame(rows, columns=["tau", "asana", "label", "angle", "cov"])
pd.set_option("display.width", 250)
print("\nVIDEO angle coverage at tau=0.5 (mean over the 8 clips of each asana):")
print(d[d.tau == 0.5].pivot_table(index="asana", columns="angle", values="cov").round(2)[[s.name for s in ANGLES]].to_string())
print("\ntau sensitivity: mean coverage over all clips, by angle")
print(d.pivot_table(index="tau", columns="angle", values="cov").round(2)[[s.name for s in ANGLES]].to_string())

# (4) CPU time of the wrapper
r = vid.iloc[56]; cap = cv2.VideoCapture(str(BASE / r.path)); fr = [cap.read()[1] for _ in range(12)]; bb(fr[0]); t = time.perf_counter()
for f in fr[1:]: bb(f)
print(f"\nCPU wrapper time: {(time.perf_counter() - t) / 11 * 1000:.0f} ms/frame at imgsz {C.BACKBONE_IMGSZ} (CPU only; NOT an RTX 5070 measurement)")
