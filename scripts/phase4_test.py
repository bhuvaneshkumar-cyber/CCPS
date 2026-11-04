"""Phase 4: the ONE look (phase4/PROTOCOL_PHASE4.md sec. 5-6). Refuses unless the protocol is frozen and unchanged and poseguard/temporal.py, the keypoint cache and
the model files match the hashes recorded at training time. Computes everything first, then records the look (phase4/TEST_TOUCHED_<device>.json), then prints
and saves phase4/phase4_test_report_<device>.txt + test_rows_<device>.csv. Both arms are always reported side by side.
usage: python scripts/phase4_test.py [--device auto|cuda:0|cpu] [--second-look] [--smoke]
--smoke: reads phase4/smoke/; real Wrong clips are NOT loaded, each Right clip's time-reversed copy plays 'Wrong' (a null) - NOT RESULTS, no marker."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poseguard import phase3 as P
from poseguard import temporal as T
from poseguard.backbone import pick_device
from poseguard.geo_v2 import auc

sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest(); PH = ROOT / "phase4"; PROT = PH / "PROTOCOL_PHASE4.md"
ap = argparse.ArgumentParser(); ap.add_argument("--device", default="auto"); ap.add_argument("--second-look", action="store_true"); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
device = pick_device(a.device); tag = "cuda" if device.startswith("cuda") else "cpu"; src = PH / "smoke" if a.smoke else PH
cfg = json.load(open(src / f"frozen_temporal_{tag}.json")); kpts = ROOT / cfg["kpts"]; touched = PH / f"TEST_TOUCHED_{tag}.json"
if a.smoke: print("*** SMOKE RUN: Wrong = time-reversed Right copies (null), stride-6 cache - NOT RESULTS ***")
else:
    assert "STATUS: DRAFT" not in PROT.read_text(encoding="utf-8") and cfg["protocol_sha256"] == sha(PROT), "protocol is a draft or changed after training - refusing to score"
    assert cfg["temporal_py_sha256"] == sha(ROOT / "poseguard" / "temporal.py") and cfg["kpts_sha256"] == sha(kpts), "temporal.py or the keypoint cache changed after training - refusing to score"
    assert all(m["sha256"] == sha(src / m["file"]) for s in cfg["arms"].values() for m in s["models"]), "a model file changed after training - refusing to score"
    if touched.exists() and not a.second_look: sys.exit("the Phase 4 test was already scored once on this device; a second look needs --second-look and is reported as one")

clips = T.load_clips(kpts, pd.read_csv(ROOT / "audit" / "inventory_videos.csv"), labels=("Right",) if a.smoke else ("Right", "Wrong"))
if a.smoke: clips += [dict(c, label="Wrong", ang=c["ang"][::-1].copy()) for c in clips]

def score_arm(arm, cl, bounds):
    w = T.collect(cl, bounds, "test"); s = cfg["arms"][arm]; ms = []
    for f in s["models"]: m = T.Forecaster().to(device); m.load_state_dict(torch.load(src / f["file"], map_location=device)); ms.append(m)
    em, j = T.errors(T.predict(ms, w["X"], device), w); ep, _ = T.errors(w["last"], w)
    meta = pd.DataFrame([dict(key=c["key"], asana=c["asana"], label=c["label"], angle=c["angle"]) for c in cl])
    d = meta.iloc[w["clip"]].reset_index(drop=True).assign(arm=arm, t=w["t"], s_model=T.score(T.zscores(em, np.array(s["sigma_model"], float), w["a"]), j), s_persist=T.score(T.zscores(ep, np.array(s["sigma_persist"], float), w["a"]), j))
    d = T.unit_scores(d); d["block"] = d.key + "|" + (d.t // (5 * T.FPS)).astype(str); return d

D = pd.concat([score_arm("chrono", clips, T.chrono_bounds)] + [score_arm(f"loao{k}", [c for c in clips if c["angle"] == k], T.loao_bounds) for k in (1, 2, 3, 4)], ignore_index=True)
D["design"] = np.where(D.arm == "chrono", "CHRONO", "LOAO")

def per_asana(d, col):
    out = {}
    for x, g in d.groupby("asana"):
        r, wr = g[g.label == "Right"][col].dropna(), g[g.label == "Wrong"][col].dropna(); out[x] = auc(r, wr) if len(r) and len(wr) else np.nan
    return out
def macro(d, col, asanas=T.HEADLINE):
    p = per_asana(d, col); v = [p[x] for x in asanas if x in p]; return float(np.nanmean(v)) if len(v) and not np.all(np.isnan(v)) else np.nan

res, lines = {}, []
for design, d in D.groupby("design"):
    sc = d[d.s_model.notna()].reset_index(drop=True); hd = sc[sc.asana.isin(T.HEADLINE)].reset_index(drop=True)
    r = dict(model=P.boot_stat(hd, lambda x: macro(x, "s_model")), persist=P.boot_stat(hd, lambda x: macro(x, "s_persist")), diff=P.boot_stat(hd, lambda x: macro(x, "s_model") - macro(x, "s_persist")))
    tab = pd.DataFrame({x: dict(n_right=int((g.label == "Right").sum()), n_wrong=int((g.label == "Wrong").sum()), blocks=g.block.nunique(), auc_model=per_asana(g, "s_model").get(x), auc_persist=per_asana(g, "s_persist").get(x),
                                abstain=float(d[d.asana == x].s_model.isna().mean()), headline=x in T.HEADLINE) for x, g in sc.groupby("asana")}).T
    sup = {x: P.boot_stat(sc[sc.asana == x].reset_index(drop=True), lambda y, x=x: per_asana(y, "s_model").get(x, np.nan)) for x in T.ASANAS if x not in T.HEADLINE and (sc.asana == x).any()}
    ok_a, ok_b = r["model"][1][0] > 0.55, r["diff"][1][0] > 0
    res[design] = (r, tab, sup, f"(a) macro AUC lower bound > 0.55: {ok_a} | (b) model beats persistence (diff lower bound > 0): {ok_b} -> " + ("SEPARATES" if ok_a and ok_b else "separates, but adds nothing over persistence" if ok_a else "NEGATIVE"))

# ---- only now: record the look, then print ----
if not a.smoke:
    looks = json.load(open(touched)) if touched.exists() else []
    looks.append(dict(at=time.strftime("%Y-%m-%d %H:%M:%S"), second_look=a.second_look, device=tag, protocol_sha256=cfg["protocol_sha256"], method=T.METHOD)); json.dump(looks, open(touched, "w"), indent=1)
def say(s=""): print(s); lines.append(s)
pd.set_option("display.width", 220)
say(f"method {T.METHOD} | device {tag} ({cfg['gpu']}) | second look: {a.second_look} | smoke: {a.smoke}\nscore = mean z over judged angles (sigma from Right val); AUC = P(Wrong score > Right score); headline macro over {len(T.HEADLINE)} asanas; unit = {T.UNIT_S:g} s (mean of its windows), n_right/n_wrong count units; CI resamples (clip, 5 s) blocks")
for design in ("LOAO", "CHRONO"):
    r, tab, sup, verdict = res[design]; f = lambda k: f"{r[k][0]:.3f} [{r[k][1][0]:.3f}, {r[k][1][1]:.3f}]"
    say(f"\n=== {design}{' (Right test frames from the training recordings: optimistic)' if design == 'CHRONO' else ' (both sides unseen recordings and view)'} ===")
    say(f"macro AUC model {f('model')} | persistence {f('persist')} | model - persistence {f('diff')}"); say(tab.round(3).to_string())
    for x, (v, ci) in sup.items(): say(f"supplementary {x}: AUC model {v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    say("criterion: " + verdict)
(PH if not a.smoke else src).joinpath(f"phase4_test_report_{tag}.txt").write_text("\n".join(lines) + "\n"); D.to_csv((PH if not a.smoke else src) / f"test_rows_{tag}.csv", index=False)
