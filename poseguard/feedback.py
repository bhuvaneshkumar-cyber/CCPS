"""Phase 5: fusion + human-readable corrective feedback. Pure functions, no I/O.

Input : per-angle Verdicts from the geometric rule engine (PRIMARY signal, always used) and, optionally, TemporalFlags from the BiLSTM model (SECONDARY).
        temporal=None reproduces the geometric-only system exactly (the ablation arm).
Output: FeedbackReport - the top_k most important corrections (worded for the USER's anatomical sides), plus which angles were unavailable and the coverage.

Rules (all PROVISIONAL design choices, to be revisited after the Phase 3/4 evaluations - none is fitted to data):
  * 'unavailable' angles are skipped, never reported as faults (they only lower `coverage`).
  * geometric 'incorrect' -> a correction; severity tier from |deviation| (TIERS below); a temporal flag on the same angle raises the tier by one.
  * temporal flag on an angle the geometric engine calls 'correct' -> a low-severity 'hold steady' note (the trajectory drifts although the pose is in range).
  * temporal flag on an unavailable angle is ignored (nothing to anchor it to).
Left/right: this module NEVER decides a side. Verdict sides are backbone-named; poseguard.lr.anatomical() is called exactly once per message."""
from dataclasses import dataclass
from typing import Optional
from poseguard import lr

SEVERITY = ("minor", "moderate", "major")
TIERS = (10.0, 25.0)  # |deviation| in degrees: < 10 minor, < 25 moderate, else major.  PROVISIONAL - recalibrate against the reference spread in Phase 3.

# joint: (advice when the angle is BELOW the reference band, advice when ABOVE it). Angle meanings are defined in poseguard/angles.py.
PHRASES = {
    "knee": ("straighten your {side} knee", "bend your {side} knee slightly"),
    "hip": ("open up your {side} hip", "fold a little more at your {side} hip"),
    "elbow": ("straighten your {side} arm", "bend your {side} elbow a little"),
    "shoulder": ("raise your {side} arm higher", "lower your {side} arm"),
    "trunk_tilt": ("lean your torso a little more", "bring your torso more upright"),
    "shoulder_line_tilt": ("tilt your shoulders slightly", "level your shoulders"),
}
NOUN = {"knee": "knee", "hip": "hip", "elbow": "elbow", "shoulder": "arm", "trunk_tilt": "torso", "shoulder_line_tilt": "shoulders"}

@dataclass(frozen=True)
class TemporalFlag:
    angle: str; z: float           # forecast error of this angle in units of the frozen validation sigma (Phase 4 defines the flag threshold k)

@dataclass(frozen=True)
class Feedback:
    text: str; severity: str; angle: str; joint: str; side: Optional[str]   # side = the user's ANATOMICAL side
    deviation: Optional[float]; source: str                                 # source: geometric | geometric+temporal | temporal

@dataclass(frozen=True)
class FeedbackReport:
    items: tuple; all_items: tuple; unavailable: tuple; coverage: float; message: str   # items = top_k shown; all_items = every correction found

def severity_of(deviation):
    a = abs(deviation); return SEVERITY[0] if a < TIERS[0] else SEVERITY[1] if a < TIERS[1] else SEVERITY[2]

def generate(verdicts, temporal=None, top_k=3):
    flags = {t.angle: t for t in (temporal or [])}; items, unavailable = [], []
    for v in verdicts:
        if v.status == "unavailable": unavailable.append(v.angle); continue
        side, flag = lr.anatomical(v.side), flags.get(v.angle)
        if v.status == "incorrect":
            sev, src = severity_of(v.deviation), "geometric"
            if flag: sev, src = SEVERITY[min(len(SEVERITY) - 1, SEVERITY.index(sev) + 1)], "geometric+temporal"
            advice = PHRASES[v.joint][0 if v.deviation < 0 else 1].format(side=side)
            items.append(Feedback(f"{advice[0].upper() + advice[1:]} - about {abs(v.deviation):.0f}° off.", sev, v.angle, v.joint, side, v.deviation, src))
        elif flag:
            who = f"{side} {NOUN[v.joint]}" if side else NOUN[v.joint]
            items.append(Feedback(f"Hold your {who} steady - it is drifting from the usual movement.", SEVERITY[0], v.angle, v.joint, side, None, "temporal"))
    key = lambda f: (SEVERITY.index(f.severity), abs(f.deviation) if f.deviation is not None else 0.0)
    items.sort(key=key, reverse=True)
    judged = [v for v in verdicts if v.status != "unavailable"]; cov = len(judged) / len(verdicts) if verdicts else 0.0
    msg = " ".join(f.text for f in items[:top_k]) if items else ("Good form - hold it." if judged else "I can't see you clearly enough to check this pose.")
    return FeedbackReport(tuple(items[:top_k]), tuple(items), tuple(unavailable), cov, msg)
