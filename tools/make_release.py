"""Build a corruption-evident release bundle of the project for transfer to another machine (e.g. the RTX 5080 machine).
What it does: (1) refuses to run while a project job is still writing files, (2) hashes every project file (SHA-256) into MANIFEST.sha256/.json, (3) records the exact
environment (ENVIRONMENT.json) and reproducibility probes (PROBES.json: real frames + the keypoints THIS machine produced), (4) writes one zip and its own SHA-256.
Excluded (re-obtained and verified by tools/fetch_data.py): data/raw, data/*.zip, weights/*.pt (their hashes are pinned in poseguard/config.py), __pycache__, release/.
--manifest-only (git transport): writes MANIFEST.sha256/.json, ENVIRONMENT.json, PROBES.json into the project root instead of a zip, to be COMMITTED;
  on the clone, `python tools/verify_release.py --checkout .` checks every file against them (catches line-ending rewrites, LFS pointers, partial clones).
usage: python tools/make_release.py [--force] [--no-probes] [--manifest-only]"""
import argparse, hashlib, json, os, platform, subprocess, sys, time, zipfile
from importlib import metadata
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SKIP_DIRS = {"__pycache__", ".git", "release", ".venv", "venv", "smoke"}           # phase4/smoke = plumbing runs, never results
SKIP_SUFFIX = {".zip", ".onnx", ".pyc", ".swp", ".tmp"}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true"); ap.add_argument("--no-probes", action="store_true"); ap.add_argument("--manifest-only", action="store_true"); a = ap.parse_args()
GENERATED = {"MANIFEST.sha256", "MANIFEST.json", "ENVIRONMENT.json", "PROBES.json"}

import psutil
me = os.getpid()
busy = [p.info["pid"] for p in psutil.process_iter(["pid", "cmdline"]) if p.info["pid"] != me and p.info["cmdline"]
        and any("python" in (p.info["cmdline"][0] or "").lower() for _ in [0]) and any(("PoseGuard" in c) or c.replace("\\", "/").startswith(("scripts/", "tools/annotate")) for c in p.info["cmdline"][1:])]
if busy and not a.force: sys.exit(f"project processes still running (pids {busy}): a file being written now would be bundled half-written. Stop them or use --force.")

def keep(p):
    rel = p.relative_to(ROOT); parts = rel.parts
    if set(parts) & SKIP_DIRS or p.suffix in SKIP_SUFFIX or p.name.startswith("~") or ".tmp." in p.name or rel.as_posix() in GENERATED: return False
    if p.suffix == ".pt": return parts[0] != "weights"                               # backbone weights are re-fetched; trained Phase 4 models (phase4/*.pt) must travel
    if parts[0] == "data": return len(parts) == 2 and p.suffix == ".json"          # data/ holds third-party datasets (Mendeley, MPI-INF-3DHP, ...): never bundled; only the tiny metadata JSONs travel
    return True

files = sorted(p for p in ROOT.rglob("*") if p.is_file() and keep(p)); total = sum(p.stat().st_size for p in files); print(f"{len(files)} files, {total / 1e6:.1f} MB")
if total > 1e9 and not a.force: sys.exit("bundle > 1 GB: something large slipped in; check the skip rules or use --force")
man = [dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p), bytes=p.stat().st_size) for p in files]
names = ("torch", "torchvision", "ultralytics", "opencv-python", "numpy", "pandas", "scipy", "scikit-learn", "onnx", "onnxruntime", "psutil")
def ver(n):
    try: return metadata.version(n)
    except metadata.PackageNotFoundError: return None
import cv2
env = dict(python=sys.version.split()[0], platform=platform.platform(), libs={n: ver(n) for n in names}, cv2_loaded=cv2.__version__,   # two OpenCV wheels may be installed; this is the one `import cv2` gets
           pip_freeze=subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines(), built=time.strftime("%Y-%m-%d %H:%M:%S"))

probes = []
if not a.no_probes:
    import cv2, pandas as pd
    from poseguard import config as C
    from poseguard.backbone import Backbone, primary
    bb = Backbone(device="cpu"); vid = pd.read_csv(ROOT / "audit" / "inventory_videos.csv"); mi = pd.read_csv(ROOT / "audit" / "manifest_dedup.csv")
    for asana in ("Tadasana", "Marjariasana", "Parvatasana", "Kati Chakrasana"):
        rel = mi[mi.usable & (mi.asana == asana)].path.iloc[7]; pr = primary(bb(cv2.imread(str(C.media(rel))))); probes.append(dict(kind="still", path=rel, frame=None, kp=None if pr is None else pr.kp.tolist()))
    for i, f in ((56, 900), (8, 600), (30, 300), (72, 1000)):
        cap = cv2.VideoCapture(str(C.media(vid.path.iloc[i]))); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read(); pr = primary(bb(im))
        probes.append(dict(kind="video", path=vid.path.iloc[i], frame=f, kp=None if pr is None else pr.kp.tolist()))
    print(f"{len(probes)} reproducibility probes recorded")

docs = {"MANIFEST.sha256": "\n".join(f"{m['sha256']}  {m['path']}" for m in man) + "\n", "MANIFEST.json": json.dumps(man, indent=1), "ENVIRONMENT.json": json.dumps(env, indent=1), "PROBES.json": json.dumps(probes)}
if a.manifest_only:
    for n, t in docs.items(): (ROOT / n).write_bytes(t.encode())
    print(f"manifest written for {len(man)} files: commit {', '.join(docs)} together with them"); sys.exit(0)
out = ROOT / "release"; out.mkdir(exist_ok=True); zp = out / f"poseguard_{time.strftime('%Y%m%d_%H%M%S')}.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files: z.write(p, p.relative_to(ROOT).as_posix())
    for n, t in docs.items(): z.writestr(n, t)
(out / f"{zp.name}.sha256").write_text(f"{sha(zp)}  {zp.name}\n"); print("bundle:", zp, f"({zp.stat().st_size / 1e6:.1f} MB)\nzip sha256:", sha(zp))
