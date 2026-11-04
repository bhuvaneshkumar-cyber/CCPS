"""Phase 3 (comparison logic only): judge one measured angle against a reference range. Pure functions, no I/O.
The reference ranges themselves are NOT defined here: they must be derived statistically from labelled-correct real data (Phase 3, per (asana, step)).
Nothing in this module ships default/placeholder ranges; callers (and tests, labelled TEST) pass them in explicitly."""
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Reference:
    lo: float; hi: float                       # accepted band in degrees, inclusive
    def __post_init__(self):
        if not self.lo <= self.hi: raise ValueError(f"Reference lo {self.lo} > hi {self.hi}")

@dataclass(frozen=True)
class Verdict:
    angle: str; joint: str; side: Optional[str]      # side = backbone-named side (see poseguard.lr)
    status: str                                       # 'correct' | 'incorrect' | 'unavailable'  (unavailable is NEVER a fault)
    value: Optional[float]; ref: Optional[Reference]
    deviation: Optional[float]                        # signed degrees outside the band: value-hi (>0) or value-lo (<0); 0 inside; None if unavailable

def judge(spec, value, ref):
    if value is None: return Verdict(spec.name, spec.joint, spec.side, "unavailable", None, ref, None)
    dev = value - ref.hi if value > ref.hi else value - ref.lo if value < ref.lo else 0.0
    return Verdict(spec.name, spec.joint, spec.side, "correct" if dev == 0.0 else "incorrect", value, ref, dev)

def judge_all(angles, refs, specs):
    """angles: {name: value|None}; refs: {name: Reference}. Angles without a reference are not judged (absent from the result)."""
    return [judge(specs[n], v, refs[n]) for n, v in angles.items() if n in refs]
