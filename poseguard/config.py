"""Single source of truth for pinned choices. Every phase and the paper must cite THESE values."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Backbone (Phase 0 decision) -------------------------------------------------
# YOLO26-Pose weights verified to exist in official release
# https://github.com/ultralytics/assets/releases/tag/v8.4.0 (yolo26{n,s,m,l,x}-pose.pt) -> no YOLO11 fallback needed.
# Size 'm' chosen as accuracy/latency middle; revisit ONLY with latency data from the RTX 5080 (Phase 6).
BACKBONE_NAME = "yolo26m-pose"
BACKBONE_WEIGHTS = ROOT / "weights" / "yolo26m-pose.pt"
BACKBONE_SHA256 = "2fbf16367022256a226035695c5c389384c6706e8bb8ab8fcd0e7976f05443c4"
KPT_CONF_TAU = 0.5  # confidence-gating threshold; FIXED (Phase 1 decision: not tuned - no ground-truth keypoints exist to tune it against)
# Inference settings (Phase 1). imgsz is FIXED (never dynamic) so latency numbers stay comparable and the ONNX export shape is fixed (Phase 6).
# 640 = the backbone's native resolution. The Phase 0 still-image statistics used 320 as a documented CPU-time proxy; video and Phase 1+ use 640.
BACKBONE_IMGSZ = 640
BACKBONE_BOX_CONF = 0.25  # person-box confidence threshold (ultralytics default, now explicit)

# --- Still-image distortion correction (Phase 0 follow-up, see audit/PHASE0_REPORT.md section 13) ----------
# Mendeley stills are stretched horizontally. Correct still-derived keypoints BEFORE any angle: x_corrected = x / STILL_STRETCH_K.
# Effective (backbone-output-space) factor, NOT the geometric one (geometric ~1.77; the backbone partly regresses stretched bodies toward
# normal proportions). PROVISIONAL: only 3 standing asanas were measurable; final acceptance test is agreement of still-derived vs
# video-derived reference angle ranges per (asana, step) in Phase 3. Not an exact fix: a bone-dependent residual of ~+-0.1 remains.
STILL_STRETCH_K = 1.65
STILL_STRETCH_K_RANGE = (1.5, 1.85)  # plausible range used for sensitivity analysis in Phase 3
# Applies to STILLS only. Video is native, undistorted, and must NOT be corrected.

# --- Method names (owner's rule: never conflate these in code, tables or the paper) ------------------
# v1: reference bands from correct examples only, every angle judged. ZERO label supervision beyond 'these are the correct examples'.
GEO_V1 = "geo_v1_zero_supervision"
# v2: same bands, but WHICH angles are judged per (asana, posture class) is selected using dev Wrong labels (phase3/PROTOCOL_V2.md). Lightly supervised.
# It must never be called 'the geometric rule engine' and never replace v1's result in any report.
GEO_V2 = "geo_v2_lightly_supervised"

# --- Dataset pin -----------------------------------------------------------------
DATASET_TITLE = "Yoga for all: A Comprehensive Collection of Yoga Images and Videos dataset"
DATASET_DOI = "10.17632/jc4mmnvcdk.1"
DATASET_VERSION = 1
DATASET_LICENSE = "CC BY 4.0"
DATASET_ZIP_SHA256 = "57851569d79efc04112d7d69c60dfb1e2f8cb0a4a8759af69734b1fd94fbe8c1"
DATA_DIR = ROOT / "data"
MEDIA_DIR = DATA_DIR / "raw" / "Yoga Postures Dataset"

def media(rel):
    """Dataset-relative path as stored in the CSVs/caches (Windows '\' separators) -> absolute path that works on Windows AND Linux."""
    return MEDIA_DIR / str(rel).replace("\\", "/")
