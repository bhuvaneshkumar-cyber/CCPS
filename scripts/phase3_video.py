"""Phase 3 video stage for geo_v1_zero_supervision (phase3/PROTOCOL_VIDEO.md + clarifications). v2 was stopped at its dev gate and is NOT run here.
Everything is computed first; only then is the single look recorded and printed (a crash therefore neither burns the look nor leaks partial results).
usage: python scripts/phase3_video.py [--second-look]"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poseguard import config as C
from poseguard import phase3 as P
from poseguard.videostage import bands_from_frames, crossfit_pairs, frame_table

PH = ROOT / "phase3"; second = "--second-look" in sys.argv; sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
if (PH / "VIDEO_TOUCHED.json").exists() and not second: sys.exit("the video stage was already scored once; a second look needs --second-look and is reported as one")
z = np.load(PH / "kpts_video_s6.npz", allow_pickle=True); q = list(z["paths"])
if not all(f"kp_{i}" in z.files for i in range(len(q))) or len(q) != 24: sys.exit("video keypoint extraction is not complete")
vid = pd.read_csv(ROOT / "audit" / "inventory_videos.csv"); vid["key"] = vid.path.str.replace("\\", "/"); ann = json.load(open(ROOT / "audit" / "step_annotations_Bhuvanesh.json"))["videos"]
man = pd.read_csv(ROOT / "audit" / "manifest_dedup.csv"); pc = P.posture_classes(man[~man.ambiguous])
VT = pd.concat([frame_table(dict(asana=(r := vid[vid.key == k].iloc[0]).asana, label=r.label, angle=int(r.angle), clip=i), z[f"kp_{i}"], z[f"fr_{i}"], ann[k]["marks"], int(r.frames), float(r.fps), pc)
                for i, k in enumerate(q)], ignore_index=True); FPS = 30.01 / 1.0
ASANAS = ["Tadasana", "Ardhakati Chakrasana", "Kati Chakrasana"]

def refs_from(df, asana):   # {(asana, class, angle): Reference} from the rows of df, per class
    out = {}
    for c, g in df.groupby("pclass"):
        for n, b in bands_from_frames(g).items(): out[(asana, c, n)] = b
    return out

def collect(asana, a, mode):
    R = VT[(VT.asana == asana) & (VT.label == "Right") & (VT.angle == a)]; W = VT[(VT.asana == asana) & (VT.label == "Wrong") & (VT.angle == a)]; rows = []
    if mode == "V2":
        for c, g in R.groupby("pclass"):
            for ref_f, test_f in crossfit_pairs({h: np.sort(x.frame.to_numpy()) for h, x in g.groupby("hold")}, FPS):
                ref, test = g[g.frame.isin(ref_f) & g.hold.isin(g.hold)], g[g.frame.isin(test_f)]
                if len(test) == 0: continue
                rows.append(test.join(P.judge_frame(test, refs_from(ref, asana))).assign(y=0))
        wref = refs_from(R, asana)
    else:
        wref = refs_from(VT[(VT.asana == asana) & (VT.label == "Right") & (VT.angle != a)], asana)
        rows.append(R.join(P.judge_frame(R, wref)).assign(y=0))
    if len(W): rows.append(W.join(P.judge_frame(W, wref)).assign(y=1))
    d = pd.concat(rows); d["mode"] = mode; d["block"] = d["clip"].astype(str) + "|" + d.hold.astype(str); return d.reset_index(drop=True)

res = {m: pd.concat([collect(a, ang, m) for a in ASANAS for ang in (1, 2, 3, 4)], ignore_index=True) for m in ("V2", "V3")}
def ba_by_asana(d):
    s = d[d.status != "abstain"]; return {a: P.prf(g.y, g.status == "incorrect")["bal_acc"] for a, g in s.groupby("asana")}
def macro_ba(d): return float(np.nanmean(list(ba_by_asana(d).values()))) if len(d) else np.nan
summary = {}
for m, d in res.items():
    sc = d[d.status != "abstain"]; per = {}
    for a in ASANAS:
        g = sc[sc.asana == a]; r = P.prf(g.y, g.status == "incorrect") if len(g) else {}
        per[a] = dict(n_scored=len(g), n_right=int((g.y == 0).sum()), n_wrong=int((g.y == 1).sum()), holds=g.block.nunique(), flag_right=1 - r.get("specificity", np.nan), flag_wrong=r.get("recall", np.nan), bal_acc=r.get("bal_acc", np.nan), abstain=float((d[d.asana == a].status == "abstain").mean()))
    mp, ci = P.boot_stat(sc.reset_index(drop=True), macro_ba, nb=1000); summary[m] = dict(per=pd.DataFrame(per).T, macro=mp, ci=ci, abstain=float((d.status == "abstain").mean()))

# V1 acceptance test (Right clips): stills dev Right (K-corrected) vs pooled Right video frames
S = P.load_stills(); sr = S[(S.split == "dev") & (S.label == "Right")]; vr = VT[VT.label == "Right"]; acc = []
for (a, c), g in vr.groupby(["asana", "pclass"]):
    for n in P.ANGLE_NAMES:
        v, s = g[n].dropna(), sr[(sr.asana == a) & (sr.pclass == c)][n].dropna()
        if len(v) >= 20 and len(s) >= 40:
            bv, bs = np.percentile(v, [15, 85]), np.percentile(s, [15, 85]); inter = max(0.0, min(bv[1], bs[1]) - max(bv[0], bs[0])); union = max(bv[1], bs[1]) - min(bv[0], bs[0])
            acc.append(dict(asana=a, pclass=c, angle=n, n_video=len(v), n_still=len(s), median_diff=float(np.median(s) - np.median(v)), band_overlap=inter / union if union > 0 else np.nan))
ACC = pd.DataFrame(acc)

# ---- only now: record the look and print ----
looks = json.load(open(PH / "VIDEO_TOUCHED.json")) if (PH / "VIDEO_TOUCHED.json").exists() else []
looks.append(dict(at=time.strftime("%Y-%m-%d %H:%M:%S"), second_look=second, protocol_video_sha256=sha(PH / "PROTOCOL_VIDEO.md"), method=C.GEO_V1)); json.dump(looks, open(PH / "VIDEO_TOUCHED.json", "w"), indent=1)
pd.set_option("display.width", 200)
print(f"method: {C.GEO_V1} (v2 not run: stopped at its dev gate)   second look: {second}")
print(f"video frame table: {len(VT)} held+trimmed sampled frames | Right {int((VT.label == 'Right').sum())}, Wrong {int((VT.label == 'Wrong').sum())} | holds {VT.groupby(['clip', 'hold']).ngroups}")
for m in ("V2", "V3"):
    s = summary[m]; print(f"\n=== {m} ({'same-view, cross-fitted' if m == 'V2' else 'cross-view'}) === macro balanced accuracy over 3 asanas = {s['macro']:.3f}  hold-level bootstrap 95% CI [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}] | abstain {s['abstain']:.3f}")
    print(s["per"].round(3).to_string())
print(f"\n=== V1 acceptance test (still K=1.65 vs video, Right): {len(ACC)} cells; |median diff| > 5 deg in {int((ACC.median_diff.abs() > 5).sum())} ({float((ACC.median_diff.abs() > 5).mean()):.2f}); median |diff| {ACC.median_diff.abs().median():.1f} deg; median band overlap {ACC.band_overlap.median():.2f}")
print(ACC.groupby("angle").median_diff.agg(lambda x: x.abs().median()).round(1).rename("median |diff| deg by angle").to_string())
ACC.to_csv(PH / "video_v1_acceptance_cells.csv", index=False)
for m in res: res[m].drop(columns=[c for c in ()], errors="ignore").to_csv(PH / f"video_rows_{m}.csv", index=False)
print("\nsaved phase3/video_v1_acceptance_cells.csv, video_rows_V2.csv, video_rows_V3.csv")
