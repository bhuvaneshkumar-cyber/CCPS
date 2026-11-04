"""Phase 4 temporal-model tests: leakage rules, masking, persistence baseline, scoring (hand-built TEST sequences, no real data). Run: python tests/test_temporal.py"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from poseguard import temporal as T

def seq(n, nan_at=()):
    """TEST sequence: angle j at step i = 10*j + i (so a value identifies its step); NaN at the given (step, angle) cells."""
    a = 10.0 * np.arange(T.NA)[None, :] + np.arange(n)[:, None]
    for i, j in nan_at: a[i, j] = np.nan
    return a

def test_target_is_after_window_and_inside_segment():
    w = T.windows(seq(100), "Tadasana", 10, 60)
    n = 60 - 10 - T.W - T.H + 1; assert len(w["t"]) == n
    first_input = np.round(w["X"][:, 0, 0] * 180).astype(int); last_input = np.round(w["X"][:, -1, 0] * 180).astype(int)
    assert (first_input >= 10).all() and (w["t"] < 60).all()                       # nothing from outside [10, 60)
    assert (w["t"] - last_input == T.H).all()                                      # target exactly H steps after the last input, never inside the window
    assert T.windows(seq(100), "Tadasana", 0, T.W + T.H - 1) is None               # too short for one window

def test_unavailable_angles_are_flagged_not_imputed():
    a = seq(40, nan_at=[(i, 2) for i in range(0, 40)] + [(T.W - 1, 3)])          # angle 2 never seen; angle 3 missing at the window's last step
    w = T.windows(a, "Tadasana", 0, 40); x0 = w["X"][0]
    assert (x0[:, 2] == 0).all() and (x0[:, T.NA + 2] == 0).all()                 # value 0 AND availability flag 0
    assert not w["seen"][0, 2] and np.isnan(w["last"][0, 2])
    assert w["seen"][0, 3] and np.isclose(w["last"][0, 3] * 180, 30 + T.W - 2)     # persistence = last OBSERVED value, not the NaN
    assert x0[:, 2 * T.NA + T.ASANAS.index("Tadasana")].all() and x0[:, 2 * T.NA:].sum() == T.W   # one-hot asana

def test_masked_loss_ignores_missing_targets():
    y = torch.tensor([[1.0, float("nan")], [0.0, 2.0]]); p = torch.tensor([[2.0, 100.0], [0.0, 0.0]])
    assert np.isclose(float(T.masked_mse(p, y)), (1 + 0 + 4) / 3)

def test_persistence_errors_and_scores_hand_computed():
    w = T.windows(seq(30), "Tadasana", 0, 30)
    err, judged = T.errors(w["last"], w); assert judged.all() and np.allclose(err, T.H)   # value grows 1 deg/step -> persistence is off by exactly H deg
    sig = T.sigma(err, judged, w["a"]); assert np.allclose(sig, T.H / 0.6745)                # robust sigma = median |err| / 0.6745
    z = T.zscores(err, sig, w["a"]); assert np.allclose(T.score(z, judged), 0.6745)
    jm = judged.copy(); jm[:, 2:] = False; assert np.isnan(T.score(z, jm)).all()          # < MIN_JUDGED judged angles -> abstain

def test_sigma_per_asana_robust_with_pooled_fallback():
    n = 40; err = np.ones((n, T.NA)); err[:, 0] = 2.0; judged = np.ones((n, T.NA), bool)
    a = np.r_[np.full(30, 0), np.full(10, 1)]; err[a == 1, 1] = 9.0                       # asana 1 has only 10 values (< MIN_SIGMA_N) -> pooled
    err[0, 2] = 1000.0                                                                    # one glitch: the median ignores it, an RMS would not
    s = T.sigma(err, judged, a)
    assert np.isclose(s[0, 0], 2 / 0.6745) and np.isclose(s[0, 2], 1 / 0.6745)
    assert np.isclose(s[1, 1], np.median(err[:, 1]) / 0.6745) and np.isclose(s[5, 0], 2 / 0.6745)   # fallback = pooled, also for asanas never seen

def test_z_is_capped_and_nan_stays_unjudged():
    sig = np.ones((len(T.ASANAS), T.NA)); err = np.full((1, T.NA), 1e6); err[0, 0] = np.nan
    z = T.zscores(err, sig, np.array([0])); assert np.isnan(z[0, 0]) and (z[0, 1:] == T.Z_CAP).all()

def test_unit_scores_average_one_second_and_skip_abstained():
    import pandas as pd
    t = np.arange(25); s = np.where(t == 3, np.nan, t.astype(float))
    d = pd.DataFrame(dict(arm="x", key="k", label="Right", asana="Tadasana", angle=1, t=t, s_model=s, s_persist=s))
    u = T.unit_scores(d); assert list(u.t) == [0, 10, 20]
    assert np.isclose(u.s_model[0], np.mean([0, 1, 2, 4, 5, 6, 7, 8, 9])) and np.isclose(u.s_model[2], 22.0)

def test_ensemble_predict_is_the_mean():
    torch.manual_seed(0); ms = [T.Forecaster(), T.Forecaster()]; X = np.random.default_rng(0).random((7, T.W, T.NIN), np.float32)
    assert np.allclose(T.predict(ms, X), (T.predict(ms[0], X) + T.predict(ms[1], X)) / 2)

def test_splits_are_ordered_and_guarded():
    b = T.chrono_bounds(1000); assert b["train"] == (0, 700) and b["val"] == (720, 800) and b["test"] == (850, 1000)
    l = T.loao_bounds(1000); assert l["train"] == (0, 900) and l["val"] == (920, 1000) and l["test"] == (0, 1000)

def test_flags_threshold():
    z = np.full(T.NA, np.nan); z[0], z[1] = T.FLAG_K, T.FLAG_K - 0.01
    f = T.flags(z); assert [x.angle for x in f] == [T.ANGLE_NAMES[0]]

def test_model_learns_a_learnable_sequence():
    # sanity: on a smooth periodic TEST signal the forecaster must beat persistence after a short fit (catches a broken training loop / wiring)
    t = np.arange(3000); a = 90 + 40 * np.sin(2 * np.pi * t / 25)[:, None] + 5 * np.arange(T.NA)[None, :]
    tr, va = T.windows(a, "Tadasana", 0, 2700), T.windows(a, "Tadasana", 2700, 3000)
    m, info = T.fit(tr, va, max_epochs=40, log=lambda s: None)
    e_m, j = T.errors(T.predict(m, va["X"]), va); e_p, _ = T.errors(va["last"], va)
    assert np.nanmean(e_m) < 0.5 * np.nanmean(e_p), (np.nanmean(e_m), np.nanmean(e_p))

if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"): f(); print("ok ", n)
    print("all temporal tests passed")
