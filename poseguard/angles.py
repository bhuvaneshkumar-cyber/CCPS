"""Phase 2: joint angles from COCO-17 keypoints, with confidence gating. Pure functions, no I/O.

CONVENTIONS (unit-tested against hand-computed values in tests/test_angles.py):
  * All angles are 2D IMAGE-PLANE angles in degrees. They depend on camera viewpoint (limitation to state in the paper).
  * 'interior' angles are measured AT the middle keypoint of the triple, in [0, 180]; 180 = straight, small = folded.
      knee     hip - knee - ankle          how bent the leg is
      hip      shoulder - hip - knee       torso-thigh fold (180 = standing straight / open hip)
      elbow    shoulder - elbow - wrist    how bent the arm is
      shoulder elbow - shoulder - hip      arm elevation relative to the trunk (0 = arm along the body, ~90 = arm out to the side, 180 = overhead)
  * trunk_tilt: angle in [0, 180] between the hip-mid -> shoulder-mid vector and image-up (0 = upright, 90 = horizontal, 180 = inverted).
  * shoulder_line_tilt: angle in [0, 90] between the shoulder line and the image horizontal (0 = level).
  * Sides are the BACKBONE-named sides ('left'/'right' from COCO indices). Convert to the user's anatomical side ONLY via poseguard.lr.anatomical().
  * Head/neck angles are deliberately not defined: Phase 0 measured 5-60 % keypoint coverage for them, too low to score.
  * CONFIDENCE GATING: an angle is computed only if every constituent keypoint has conf >= tau (and finite coordinates); otherwise None
    ('unavailable'). None must never be turned into 'incorrect' downstream.
  * Stills carry a horizontal stretch (Phase 0 sec.13): pass stretch_k=STILL_STRETCH_K for still-derived keypoints, leave 1.0 for video."""
from typing import NamedTuple, Optional
import numpy as np
from poseguard.config import KPT_CONF_TAU
from poseguard.lr import coco_side

class AngleSpec(NamedTuple):
    joint: str; side: Optional[str]; kind: str; idx: tuple
    @property
    def name(self): return f"{self.side}_{self.joint}" if self.side else self.joint

_TRIPLES = {"knee": ((11, 13, 15), (12, 14, 16)), "hip": ((5, 11, 13), (6, 12, 14)), "elbow": ((5, 7, 9), (6, 8, 10)), "shoulder": ((7, 5, 11), (8, 6, 12))}
ANGLES = [AngleSpec(j, coco_side(l[1]), "interior", l) for j, (l, r) in _TRIPLES.items() for l in (l, r)] + [
    AngleSpec("trunk_tilt", None, "trunk_tilt", (5, 6, 11, 12)), AngleSpec("shoulder_line_tilt", None, "shoulder_line", (5, 6))]
ANGLE_BY_NAME = {a.name: a for a in ANGLES}

def interior_angle(a, b, c):
    """Angle at b between rays b->a and b->c, degrees in [0, 180]; None if a ray has zero length."""
    u, v = np.asarray(a, float) - b, np.asarray(c, float) - b; nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return None if nu == 0 or nv == 0 else float(np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1.0, 1.0))))

def trunk_tilt(sho_l, sho_r, hip_l, hip_r):
    t = (np.asarray(sho_l, float) + sho_r) / 2 - (np.asarray(hip_l, float) + hip_r) / 2; n = np.linalg.norm(t)
    return None if n == 0 else float(np.degrees(np.arccos(np.clip(t @ np.array([0.0, -1.0]) / n, -1.0, 1.0))))  # image y grows DOWN, so 'up' = (0,-1)

def shoulder_line_tilt(sho_l, sho_r):
    d = np.asarray(sho_r, float) - sho_l; return None if not d.any() else float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]))))

def compute_angles(kp, tau=KPT_CONF_TAU, stretch_k=1.0):
    """kp: (17,3) array (x, y, conf) -> {angle name: degrees or None}. x is divided by stretch_k first (still correction), then gating."""
    kp = np.array(kp, float); kp[:, 0] /= stretch_k; out = {}
    for s in ANGLES:
        pts, conf = kp[list(s.idx), :2], kp[list(s.idx), 2]
        if not (np.isfinite(pts).all() and np.isfinite(conf).all() and (conf >= tau).all()): out[s.name] = None; continue
        out[s.name] = (interior_angle(*pts) if s.kind == "interior" else trunk_tilt(*pts) if s.kind == "trunk_tilt" else shoulder_line_tilt(*pts))
    return out
