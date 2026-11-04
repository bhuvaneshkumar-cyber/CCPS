"""Re-obtain the dataset and the backbone weights on a new machine and VERIFY them against the pinned SHA-256 values (never copy the 3.2 GB dataset by hand).
usage: python tools/fetch_data.py [--verify-only]"""
import argparse, hashlib, sys, urllib.request, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from poseguard import config as C

DATA_URL = "https://data.mendeley.com/public-files/datasets/jc4mmnvcdk/files/65279f16-910d-4910-b797-4db64fce9dc2/file_downloaded"
W_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-pose.pt"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
def get(url, dest, want):
    if dest.exists() and sha(dest) == want: print("ok (already verified):", dest.name); return
    if a.verify_only: sys.exit(f"MISSING or CORRUPT: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True); print("downloading", url); urllib.request.urlretrieve(url, dest)
    if sha(dest) != want: dest.unlink(); sys.exit(f"CHECKSUM MISMATCH after download: {dest.name} (deleted)")
    print("downloaded and verified:", dest.name)
ap = argparse.ArgumentParser(); ap.add_argument("--verify-only", action="store_true"); a = ap.parse_args()
get(W_URL, C.BACKBONE_WEIGHTS, C.BACKBONE_SHA256)
z = C.DATA_DIR / "yoga.zip"; get(DATA_URL, z, C.DATASET_ZIP_SHA256)
raw = C.DATA_DIR / "raw" / "Yoga Postures Dataset"
if not raw.exists():
    if a.verify_only: sys.exit("dataset not extracted")
    zipfile.ZipFile(z).extractall(C.DATA_DIR / "raw")
n_jpg, n_mp4 = len(list(raw.rglob("*.jpg"))), len(list(raw.rglob("*.mp4")))
print(f"images {n_jpg} (expect 11344), videos {n_mp4} (expect 80)"); sys.exit(0 if (n_jpg, n_mp4) == (11344, 80) else "dataset file counts wrong")
