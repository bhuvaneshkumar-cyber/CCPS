"""Phase 3 video stage helpers (see phase3/PROTOCOL_VIDEO.md). Pure functions except frame_table()'s inputs."""
import numpy as np, pandas as pd
from poseguard.angles import compute_angles
from poseguard.phase3 import ANGLE_NAMES

TRIM_S, MIN_S = 1.5, 4.0

def held_segments(marks, n_frames, fps, trim_s=TRIM_S, min_s=MIN_S):
    """Annotated marks {frame: label} -> [(start, end_exclusive, step)] of TRIMMED held segments (label > 0). Consecutive equal labels are merged
    (stray repeated presses); segments shorter than min_s before trimming are dropped; trim_s is cut from both ends."""
    ks = sorted(int(k) for k in marks); segs = [[k, ks[i + 1] if i + 1 < len(ks) else n_frames, int(marks[k] if k in marks else marks[str(k)])] for i, k in enumerate(ks)]
    merged = []
    for s, e, l in segs:
        if merged and merged[-1][2] == l: merged[-1][1] = e
        else: merged.append([s, e, l])
    out = []
    for s, e, l in merged:
        if l > 0 and (e - s) / fps >= min_s and (e - s) / fps - 2 * trim_s > 0: out.append((int(round(s + trim_s * fps)), int(round(e - trim_s * fps)), l))
    return out

def frame_table(clip, kp, fr, marks, n_frames, fps, pclass):
    """One row per kept (held, trimmed) sampled frame: clip meta + step/posture class/hold id + the 10 angles (video: no stretch correction)."""
    rows = []
    for h, (s, e, step) in enumerate(held_segments(marks, n_frames, fps)):
        for k, f in zip(kp, fr):
            if s <= f < e: rows.append(dict(**clip, step=step, pclass=pclass[(clip["asana"], step)], hold=h, frame=int(f), **dict(zip(ANGLE_NAMES, [np.nan if v is None else v for v in compute_angles(k).values()]))))
    return pd.DataFrame(rows)


MIN_REF_FRAMES, GAP_S = 10, 1.5

def crossfit_pairs(hold_frames, fps, gap_s=GAP_S):
    """hold_frames: {hold_id: sorted array of sampled frame indices} for ONE posture class of ONE Right clip -> list of (reference_frames, test_frames).
    Two or more holds: each hold is tested with the frames of all the others. One hold: first part = reference, last part = test, separated by a gap_s gap around the midpoint."""
    ids = sorted(hold_frames)
    if len(ids) >= 2: return [(np.concatenate([hold_frames[j] for j in ids if j != i]), np.asarray(hold_frames[i])) for i in ids]
    f = np.asarray(hold_frames[ids[0]]); mid = (f.min() + f.max()) / 2; half = gap_s * fps / 2
    return [(f[f < mid - half], f[f > mid + half])]

def bands_from_frames(df, p=15.0, delta=5.0, min_n=MIN_REF_FRAMES):
    """{angle: Reference} from the rows of df (one asana/class/context); angles with < min_n values are omitted (not judged)."""
    from poseguard.phase3 import band
    return {n: band(v.to_numpy(), p, delta) for n in ANGLE_NAMES for v in [df[n].dropna()] if len(v) >= min_n}
