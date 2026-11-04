"""THE single place for left/right conventions. No other module may hard-code, flip, or re-derive a side.

Two independent facts, both recorded here:
1. BACKBONE_LEFT_IS_ANATOMICAL_LEFT - does a keypoint the backbone names 'left_*' belong to the subject's anatomical left?
   VERIFIED True for YOLO26m-Pose on a non-mirrored image: front-facing subject => left_shoulder is on the image's RIGHT
   (evidence: tools/verify_lr.py --dataset -> audit/verify_lr_report.txt).
2. INPUT_MIRRORED - are frames flipped (selfie preview / cv2.flip) BEFORE the backbone sees them? Depends on the camera source, so it must be
   confirmed per source with a recording of one known raised arm: `tools/verify_lr.py --video clip.mp4 --raised left|right` (required in Phase 6).
   False for the Mendeley videos as far as the backbone-side check can tell; not verifiable for a live webcam until it is run.

Everything downstream works with COCO sides ('left'/'right' as named by the backbone) and calls anatomical() exactly once, when text is produced."""

BACKBONE_LEFT_IS_ANATOMICAL_LEFT = True
INPUT_MIRRORED = False
COCO_LEFT = (1, 3, 5, 7, 9, 11, 13, 15)   # COCO-17 odd indices are 'left_*'
COCO_RIGHT = (2, 4, 6, 8, 10, 12, 14, 16)

def coco_side(idx):
    """'left'/'right' as NAMED by the backbone for COCO-17 index idx (None for the nose)."""
    return "left" if idx in COCO_LEFT else "right" if idx in COCO_RIGHT else None

def anatomical(side):
    """Backbone-named side -> the USER's anatomical side, given the two facts above."""
    if side is None: return None
    flip = BACKBONE_LEFT_IS_ANATOMICAL_LEFT == INPUT_MIRRORED   # (anatomical, not mirrored) -> keep; (anatomical, mirrored) -> flip; (not anatomical, not mirrored) -> flip
    return {"left": "right", "right": "left"}[side] if flip else side
