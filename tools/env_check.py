"""Environment check for a target machine (run FIRST after the transfer; required before any GPU result is trusted).
usage: python tools/env_check.py [--require-cuda] [--pins ENVIRONMENT.json]
Checks: Python/platform, library versions (ultralytics and torch versions change keypoints, so they are compared against the pins), CUDA availability, GPU name and compute
capability, and a small GPU matmul that fails loudly on a missing-kernel wheel (RTX 5080 = Blackwell sm_120 needs a CUDA >= 12.8 build of torch)."""
import argparse, json, platform, sys
from importlib import metadata
from pathlib import Path

LIBS = ["torch", "torchvision", "ultralytics", "opencv-python", "numpy", "pandas", "scipy", "scikit-learn", "onnx", "onnxruntime", "psutil"]
ap = argparse.ArgumentParser(); ap.add_argument("--require-cuda", action="store_true"); ap.add_argument("--pins"); a = ap.parse_args(); bad = []
print(f"python {sys.version.split()[0]} | {platform.platform()}")
vers = {}
for n in LIBS:
    try: vers[n] = metadata.version(n)
    except metadata.PackageNotFoundError: vers[n] = None
print("versions:", vers)
if a.pins and Path(a.pins).exists():
    try: allp = json.load(open(a.pins)); pins = allp.get("libs", {})
    except (ValueError, OSError) as e: allp, pins = {}, {}; bad.append(f"pins file {a.pins} unreadable ({e})")
    import cv2; print("cv2 loaded:", cv2.__version__, "| pinned:", allp.get("cv2_loaded"))
    if allp.get("cv2_loaded") and cv2.__version__ != allp["cv2_loaded"]:   # the video decoder decides which frames the backbone sees
        bad.append(f"cv2 {cv2.__version__} loaded, keypoint caches were decoded with {allp['cv2_loaded']} (see the OpenCV note in requirements.txt)")
    for n in ("ultralytics", "torch"):
        if pins.get(n) and vers.get(n) and pins[n].split("+")[0] != vers[n].split("+")[0]: bad.append(f"{n} {vers[n]} differs from pinned {pins[n]} (keypoints may differ)")
    for n in ("numpy", "pandas", "scipy", "scikit-learn", "opencv-python"):
        if pins.get(n) and vers.get(n) and pins[n].split(".")[:2] != vers[n].split(".")[:2]: print(f"  note: {n} {vers[n]} vs pinned {pins[n]} (minor version differs)")
import torch
print("torch", torch.__version__, "| cuda build:", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0), "| arch list:", torch.cuda.get_arch_list())
    try:
        x = torch.ones(256, 256, device="cuda"); ok = float((x @ x).sum().item()) == 256 ** 3; print("gpu matmul check:", "OK" if ok else "WRONG RESULT")
        if not ok: bad.append("GPU matmul returned a wrong result")
    except RuntimeError as e: bad.append(f"GPU matmul failed ({e}); on an RTX 5080 install a CUDA >= 12.8 torch build")
elif a.require_cuda: bad.append("CUDA required but not available")
print("RESULT:", "FAIL" if bad else "OK"); [print("  -", b) for b in bad]; sys.exit(1 if bad else 0)
