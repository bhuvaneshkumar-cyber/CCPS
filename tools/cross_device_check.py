"""Cross-device check (MIGRATION.md, first GPU task): do the GPU keypoints reproduce the CPU caches that every Phase 0-3 result was computed from?
Never overwrites a CPU cache. Compares on the SAME frames:
  (a) video: phase4/kpts_video_full_cuda.npz vs phase3/kpts_video_s6.npz (24 annotated clips, every 6th frame) and audit/kpts_video_640.npz (80 clips x 40 frames)
  (b) stills: all 8,259 stills of phase3/kpts_stills_640.npz re-run on the GPU -> phase4/kpts_stills_640_cuda.npz (skip with --no-stills)
Criterion (fixed before the first run): PASS iff, in every comparison, detect/no-detect agreement >= 0.99 AND >= 0.99 of frames detected on both devices have
worst confident-joint difference <= 3 px (the probe tolerance). Angle differences are reported, not gated. The result is reported whatever it is.
usage: python tools/cross_device_check.py [--no-stills]      -> phase4/cross_device_report.txt"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poseguard import config as C
from poseguard.temporal import angle_seq

TOL, MIN_FRAC = 3.0, 0.99; lines = []
def say(s=""): print(s, flush=True); lines.append(s)

def compare(name, ref, new):
    """ref/new (n,17,3) on the same frames. Returns PASS/FAIL and prints the numbers."""
    dr, dn = ~np.isnan(ref[:, 0, 0]), ~np.isnan(new[:, 0, 0]); both = dr & dn; agree = float((dr == dn).mean())
    r, q = ref[both], new[both]; m = (r[:, :, 2] >= C.KPT_CONF_TAU) & (q[:, :, 2] >= C.KPT_CONF_TAU)
    d = np.where(m, np.linalg.norm(r[:, :, :2] - q[:, :, :2], axis=2), np.nan); worst = np.nanmax(np.where(m.any(1)[:, None], d, 0), 1)
    with np.errstate(invalid="ignore"): da = np.abs(angle_seq(r) - angle_seq(q))
    frac = float((worst <= TOL).mean()) if len(worst) else np.nan; ok = agree >= MIN_FRAC and frac >= MIN_FRAC
    say(f"{name}: frames {len(ref)} | detection agreement {agree:.4f} (CPU-only {int((dr & ~dn).sum())}, GPU-only {int((~dr & dn).sum())}) | both detected {int(both.sum())}")
    say(f"   joint diff px: median {np.nanmedian(d):.3f}  p95 {np.nanpercentile(d, 95):.3f}  max {np.nanmax(d):.2f} | frames with worst joint <= {TOL} px: {frac:.4f}")
    say(f"   angle diff deg (both computable): median {np.nanmedian(da):.3f}  p95 {np.nanpercentile(da, 95):.3f}  max {np.nanmax(da):.1f}   -> {'PASS' if ok else 'FAIL'}")
    return ok

full = ROOT / "phase4" / "kpts_video_full_cuda.npz"
if not full.exists(): sys.exit(f"{full} missing: run scripts/extract_video_full.py on the GPU first")
G = np.load(full, allow_pickle=True); gi = {k: i for i, k in enumerate(G["paths"])}
def gpu_at(key, frames):
    fr = G[f"fr_{gi[key]}"]; pos = np.minimum(np.searchsorted(fr, frames), len(fr) - 1); assert (fr[pos] == frames).all(), f"GPU cache lacks frames of {key}"
    return G[f"kp_{gi[key]}"][pos]
oks = []
z = np.load(ROOT / "phase3" / "kpts_video_s6.npz", allow_pickle=True); ks = [k for k in z.files if k.startswith("kp_")]
oks.append(compare("video s6 (24 clips)", np.concatenate([z[k] for k in ks]), np.concatenate([gpu_at(z["paths"][int(k[3:])], z[f"fr_{k[3:]}"]) for k in ks])))
z = np.load(ROOT / "audit" / "kpts_video_640.npz", allow_pickle=True)
oks.append(compare("video 40/clip (80 clips)", z["kp"].reshape(-1, 17, 3), np.concatenate([gpu_at(p.replace("\\", "/"), z["fidx"][i]) for i, p in enumerate(z["path"])])))
if "--no-stills" not in sys.argv:
    import cv2
    from poseguard.backbone import Backbone, primary
    z = np.load(ROOT / "phase3" / "kpts_stills_640.npz", allow_pickle=True); bb = Backbone(device="auto"); assert str(bb.device).startswith("cuda"), "stills check needs the GPU"
    out = ROOT / "phase4" / "kpts_stills_640_cuda.npz"; kp = np.full(z["kp"].shape, np.nan, np.float32)
    for i, p in enumerate(z["path"]):
        q = primary(bb(cv2.imread(str(C.media(p)))))
        if q is not None: kp[i] = q.kp
    np.savez(out, path=z["path"], kp=kp, done=len(kp)); oks.append(compare(f"stills ({len(kp)})", z["kp"], kp))
say(f"\nOVERALL: {'PASS - CPU-derived Phase 0-3 results carry over to this device' if all(oks) else 'FAIL - report the differences; do not reuse CPU caches as if identical'}")
(ROOT / "phase4" / "cross_device_report.txt").write_text("\n".join(lines) + "\n"); sys.exit(0 if all(oks) else 1)
