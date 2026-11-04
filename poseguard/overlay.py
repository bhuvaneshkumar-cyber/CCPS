"""Skeleton drawing (Phase 1 verification now; the Phase 6 overlay builds on it). Colours by BACKBONE-named side:
left_* blue, right_* red (BGR); keypoints below tau are drawn small and grey (they would be gated out of every angle); limbs only between confident joints."""
import cv2
from poseguard import lr
from poseguard.config import KPT_CONF_TAU

LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]

def draw_skeleton(frame, kp, tau=KPT_CONF_TAU, scale=1.0, thickness=2):
    """Draw on `frame` in place and return it. kp is (17,3); NaN keypoints are skipped."""
    for a, b in LIMBS:
        if kp[a, 2] >= tau and kp[b, 2] >= tau:
            cv2.line(frame, (int(kp[a, 0] * scale), int(kp[a, 1] * scale)), (int(kp[b, 0] * scale), int(kp[b, 1] * scale)), (255, 200, 0), thickness)
    for i, (x, y, c) in enumerate(kp):
        if c != c: continue                                   # NaN
        if c >= tau: cv2.circle(frame, (int(x * scale), int(y * scale)), thickness + 2, (255, 90, 0) if lr.coco_side(i) == "left" else (0, 0, 255) if lr.coco_side(i) == "right" else (0, 255, 0), -1)
        else: cv2.circle(frame, (int(x * scale), int(y * scale)), 2, (150, 150, 150), -1)
    return frame
