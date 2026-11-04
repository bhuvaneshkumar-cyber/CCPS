"""Phase 1: the pretrained pose backbone (YOLO26m-Pose), frozen - no fine-tuning, no training code anywhere.

* Weights integrity: the file's SHA-256 must equal config.BACKBONE_SHA256, otherwise construction fails (so results always name the model actually used).
* Fixed input resolution (config.BACKBONE_IMGSZ), batch 1 per call: no dynamic shapes, so latency stays comparable and the ONNX export shape is fixed.
* Output: every detected person as COCO-17 (x, y, conf) in INPUT-FRAME pixels, sides as named by the backbone (see poseguard.lr for the anatomical meaning).
* Person selection policy: `primary()` = highest box confidence. This is a single-user system; tracking several people is out of scope.
* Device: 'auto' = CUDA if available (after a small matmul health check that FAILS LOUDLY on a missing-kernel wheel, e.g. Blackwell/sm_120 needing CUDA >= 12.8), else CPU with a printed notice."""
import hashlib
from dataclasses import dataclass
import numpy as np
from poseguard import config as C

@dataclass(frozen=True)
class Person:
    kp: np.ndarray    # (17, 3): x, y, confidence
    box: np.ndarray   # (4,): x1, y1, x2, y2
    conf: float       # person-box confidence

def primary(persons):
    """The person the pipeline follows: highest box confidence; None if nobody was detected."""
    return max(persons, key=lambda p: p.conf) if persons else None

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def pick_device(device="auto"):
    if device != "auto": return device
    import torch
    if not torch.cuda.is_available():
        print("[poseguard] CUDA not available -> running on CPU (latency numbers from this run are NOT RTX 5080 numbers)"); return "cpu"
    try:
        a = torch.ones(64, 64, device="cuda"); ok = float((a @ a).sum().item()) == 64 ** 3   # every entry of ones@ones is 64 -> sum = 64*64*64
    except RuntimeError as e:
        raise RuntimeError("CUDA is available but a small GPU matmul failed. On an RTX 5080 (Blackwell, sm_120) this means the torch wheel lacks its kernels: "
                           "install a CUDA >= 12.8 build of torch. Not falling back to CPU silently.") from e
    if not ok: raise RuntimeError("GPU matmul health check returned a wrong result")
    return "cuda:0"

class Backbone:
    def __init__(self, device="auto", imgsz=C.BACKBONE_IMGSZ, verify_hash=True):
        if verify_hash and sha256_of(C.BACKBONE_WEIGHTS) != C.BACKBONE_SHA256:
            raise RuntimeError(f"{C.BACKBONE_WEIGHTS} does not match the pinned SHA-256; refusing to run an unpinned backbone")
        from ultralytics import YOLO
        self.model, self.imgsz, self.device = YOLO(str(C.BACKBONE_WEIGHTS)), imgsz, pick_device(device)
        self.name = C.BACKBONE_NAME

    def __call__(self, frame):
        """BGR frame (H,W,3 uint8) -> list[Person] (possibly empty)."""
        r = self.model.predict(frame, imgsz=self.imgsz, conf=C.BACKBONE_BOX_CONF, device=self.device, verbose=False)[0]
        return [Person(r.keypoints.data[i].cpu().numpy(), r.boxes.xyxy[i].cpu().numpy(), float(r.boxes.conf[i])) for i in range(len(r.boxes))]

    def video(self, source, stride=1, max_frames=None):
        """Yield (frame_index, frame, persons) from a video path or a camera index. stride>1 skips frames (the skipped ones are still decoded)."""
        import cv2
        cap = cv2.VideoCapture(source); i = 0
        try:
            while max_frames is None or i < max_frames:
                ok, frame = cap.read()
                if not ok: break
                if i % stride == 0: yield i, frame, self(frame)
                i += 1
        finally: cap.release()
