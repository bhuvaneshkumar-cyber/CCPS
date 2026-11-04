"""Phase 4: temporal forecaster, method name temporal_bilstm_att_v1 (phase4/PROTOCOL_PHASE4.md). Library only; scripts do the I/O.
Trained, early-stopped and calibrated on RIGHT video only (no Wrong labels). Its forecast error in units of the Right-validation sigma (z) is the
deviation signal; persistence (last observed value) is the baseline it must beat. All constants below are fixed by the protocol, never tuned.

Leakage rules (unit-tested in tests/test_temporal.py): the target step lies H steps AFTER the window and is never inside it; a window and its target lie
inside one segment; NaN (unavailable) angles enter as 0 plus an availability flag and never contribute to loss, sigma or score."""
import copy
import numpy as np, torch, torch.nn as nn
from poseguard.angles import compute_angles
from poseguard.feedback import TemporalFlag
from poseguard.phase3 import ANGLE_NAMES

METHOD = "temporal_bilstm_att_v1"
ASANAS = ["Anantasana", "Ardhakati Chakrasana", "Bhujangasana", "Kati Chakrasana", "Marjariasana", "Parvatasana", "Sarvangasana", "Tadasana", "Vajrasana", "Viparita Karani"]
HEADLINE = [a for a in ASANAS if a not in ("Sarvangasana", "Viparita Karani")]     # the two inverted asanas are supplementary only (Phase 0 decision 4)
SRC_FPS, FPS = 30, 10                     # source ~30 fps -> 10 fps (every 3rd frame)
W, H, HIDDEN = 20, 5, 64                  # 2.0 s window, 0.5 s horizon
LR, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 256, 200, 15
SEEDS = (0, 1, 2, 3, 4)                   # every arm is a 5-seed ensemble (mean forecast): no lucky/unlucky single seed can carry the result
MIN_JUDGED, FLAG_K = 3, 3.0
Z_CAP, MIN_SIGMA_N = 10.0, 20             # per-angle z capped so one glitchy joint cannot dominate; per-(asana, angle) sigma needs >= 20 judged val values, else pooled
UNIT_S = 1.0                              # decision unit: mean sample score over 1 s of consecutive samples (protocol sec. 4)
NA, NIN = len(ANGLE_NAMES), 2 * len(ANGLE_NAMES) + len(ASANAS)

# ---- data ------------------------------------------------------------------------------------------------------
def angle_seq(kp):
    """(n,17,3) keypoints (NaN rows = nobody detected) -> (n, 10) angles in degrees, NaN = unavailable."""
    return np.array([[np.nan if v is None else v for v in compute_angles(k).values()] for k in kp], float).reshape(len(kp), NA)

def load_clips(npz_path, inventory, labels=("Right", "Wrong")):
    """Keypoint cache (kp_<i>, fr_<i>, paths) -> [dict(key, asana, label, angle, ang (n,10) at FPS)]. Clips whose label is not in `labels` are skipped
    BEFORE any angle is computed (the training script passes labels=('Right',))."""
    z = np.load(npz_path, allow_pickle=True); inv = inventory.assign(key=inventory.path.str.replace("\\", "/")).set_index("key"); out = []
    for i, k in enumerate(z["paths"]):
        r = inv.loc[k]
        if r.label not in labels or f"kp_{i}" not in z.files: continue
        keep = z[f"fr_{i}"] % (SRC_FPS // FPS) == 0       # frame-index based; a stride-6 cache (5 fps) keeps everything (smoke runs only)
        out.append(dict(key=k, asana=r.asana, label=r.label, angle=int(r.angle), ang=angle_seq(z[f"kp_{i}"][keep])))
    return out

def chrono_bounds(n):
    """CHRONO arm segments of one clip of n steps (protocol sec. 3). Wrong clips are scored on the same 'test' segment."""
    a, b = int(0.7 * n), int(0.8 * n); return dict(train=(0, a), val=(a + 2 * FPS, b), test=(b + 5 * FPS, n))

def loao_bounds(n):
    """LOAO arm: training clips split 90/10 (2 s guard); held-out clips are scored in full."""
    a = int(0.9 * n); return dict(train=(0, a), val=(a + 2 * FPS, n), test=(0, n))

def windows(ang, asana, start, end):
    """Every window whose W inputs AND target lie in [start, end). Returns None if there is none, else dict:
    X (n,W,NIN) float32 model input, y (n,NA) target /180 (NaN = unavailable), seen (n,NA) angle observed in the window, last (n,NA) last observed value /180, t (n,) target step."""
    n = end - start - W - H + 1
    if n <= 0: return None
    t0 = np.arange(start, start + n); x = ang[t0[:, None] + np.arange(W)] / 180.0; m = ~np.isnan(x)
    oh = np.zeros(len(ASANAS), np.float32); oh[ASANAS.index(asana)] = 1
    X = np.concatenate([np.nan_to_num(x), m, np.broadcast_to(oh, (n, W, len(ASANAS)))], -1).astype(np.float32)
    pos = np.where(m, np.arange(W)[None, :, None], -1).max(1); seen = pos >= 0
    last = np.where(seen, x[np.arange(n)[:, None], np.clip(pos, 0, None), np.arange(NA)[None, :]], np.nan)
    t = t0 + W - 1 + H; return dict(X=X, y=ang[t] / 180.0, seen=seen, last=last, t=t, a=np.full(n, ASANAS.index(asana)))

def collect(clips, bounds, part):
    """Stack windows of segment `part` over clips (bounds: n -> {part: (start, end)}). Adds clip (index into clips) per window. None if empty."""
    ws = []
    for ci, c in enumerate(clips):
        s, e = bounds(len(c["ang"]))[part]; w = windows(c["ang"], c["asana"], s, e)
        if w is not None: ws.append(dict(w, clip=np.full(len(w["t"]), ci)))
    return {k: np.concatenate([w[k] for w in ws]) for k in ws[0]} if ws else None

# ---- model -----------------------------------------------------------------------------------------------------
class Forecaster(nn.Module):
    """BiLSTM over the past window (bidirectional WITHIN the window, causal w.r.t. the target) + additive attention pooling + linear head."""
    def __init__(self, hidden=HIDDEN):
        super().__init__(); self.lstm = nn.LSTM(NIN, hidden, batch_first=True, bidirectional=True); self.att = nn.Linear(2 * hidden, 1); self.head = nn.Linear(2 * hidden, NA)
    def forward(self, x):
        h, _ = self.lstm(x); a = torch.softmax(self.att(h).squeeze(-1), 1); return self.head((a.unsqueeze(-1) * h).sum(1))

def masked_mse(pred, y):
    m = ~torch.isnan(y); return ((pred - torch.nan_to_num(y)) ** 2 * m).sum() / m.sum().clamp(min=1)

def predict(model, X, device="cpu", bs=4096):
    """One model or a list of models (ensemble: mean forecast)."""
    if isinstance(model, (list, tuple)): return np.mean([predict(m, X, device, bs) for m in model], 0)
    model.eval()
    with torch.no_grad(): return np.concatenate([model(torch.from_numpy(X[i:i + bs]).to(device)).cpu().numpy() for i in range(0, len(X), bs)]) if len(X) else np.zeros((0, NA))

def fit(train, val, device="cpu", max_epochs=MAX_EPOCHS, log=print, seed=0):
    """Adam + early stopping on Right-validation masked MSE (protocol sec. 2). Returns (model at best val epoch, info)."""
    torch.manual_seed(seed); torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model = Forecaster().to(device); opt = torch.optim.Adam(model.parameters(), LR); g = torch.Generator().manual_seed(seed)
    Xt, Yt = torch.from_numpy(train["X"]).to(device), torch.from_numpy(train["y"].astype(np.float32)).to(device)
    Xv, Yv = torch.from_numpy(val["X"]).to(device), torch.from_numpy(val["y"].astype(np.float32)).to(device)
    best, state, best_ep, bad = np.inf, None, -1, 0
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(len(Xt), generator=g).to(device)
        for i in range(0, len(Xt), BATCH):
            b = perm[i:i + BATCH]; loss = masked_mse(model(Xt[b]), Yt[b]); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = float(masked_mse(model(Xv), Yv))
        if v < best: best, state, best_ep, bad = v, copy.deepcopy(model.state_dict()), ep, 0
        else: bad += 1
        if ep % 10 == 0 or bad == 0: log(f"  epoch {ep:3d}  train {loss.item():.5f}  val {v:.5f}{'  *' if bad == 0 else ''}")
        if bad >= PATIENCE: break
    model.load_state_dict(state); return model, dict(best_val_mse=best, best_epoch=best_ep, epochs_run=ep + 1, n_train=len(Xt), n_val=len(Xv))

# ---- scoring ---------------------------------------------------------------------------------------------------
def errors(pred, w):
    """|forecast - observed| in degrees and the judged mask (target available AND angle seen in the window: identical for model and persistence)."""
    judged = w["seen"] & ~np.isnan(w["y"]); return np.where(judged, np.abs(pred - w["y"]) * 180.0, np.nan), judged

def _robust(e): return float(np.median(e) / 0.6745) if len(e) else np.nan      # median |error| -> sigma of a normal; one detection glitch cannot inflate it (RMS can)

def sigma(err, judged, a):
    """(len(ASANAS), NA) robust sigma of validation residuals in degrees: per (asana, angle) when >= MIN_SIGMA_N judged values, else the angle's pooled value.
    Per asana because the same joint is still in one asana and moving in another; a pooled sigma would make one asana's joints dominate its score."""
    pooled = np.array([_robust(err[judged[:, j], j]) for j in range(NA)]); out = np.tile(pooled, (len(ASANAS), 1))
    for k in np.unique(a):
        for j in range(NA):
            e = err[(a == k) & judged[:, j], j]
            if len(e) >= MIN_SIGMA_N: out[k, j] = _robust(e)
    return out

def zscores(err, sig, a):
    """err (n,NA), sig from sigma(), a (n,) asana index -> z capped at Z_CAP (NaN stays NaN = not judged)."""
    with np.errstate(divide="ignore", invalid="ignore"): return np.minimum(err / sig[a], Z_CAP)

def unit_scores(d):
    """Sample rows (arm, key, label, asana, angle, t, s_model, s_persist) -> one row per UNIT_S-second decision unit (mean of its non-abstained samples;
    NaN if all abstained). Consecutive windows overlap by 95 %, so the unit, not the window, is what is counted and scored."""
    u = d.assign(unit=d.t // int(round(UNIT_S * FPS)))
    return u.groupby(["arm", "key", "label", "asana", "angle", "unit"], as_index=False)[["s_model", "s_persist"]].mean().assign(t=lambda x: x.unit * int(round(UNIT_S * FPS)))

def score(z, judged):
    """Sample score = mean z over judged angles; NaN (abstain) if fewer than MIN_JUDGED angles are judged."""
    n = (judged & ~np.isnan(z)).sum(1)
    with np.errstate(invalid="ignore"): s = np.nansum(np.where(judged, z, 0), 1) / n
    return np.where(n >= MIN_JUDGED, s, np.nan)

def flags(z_row):
    """One sample's z vector -> TemporalFlags for poseguard.feedback.generate (Phase 5). Threshold FLAG_K is provisional."""
    return [TemporalFlag(n, float(v)) for n, v in zip(ANGLE_NAMES, z_row) if v == v and v >= FLAG_K]
