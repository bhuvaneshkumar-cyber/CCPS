"""Keyboard-driven step-boundary annotator for the 80 Mendeley Yoga videos (tkinter window, cv2 only decodes).

PROTOCOL (what a mark means; same for every annotator):
  * Press the step number (1..N) on the FIRST frame where the performer has ARRIVED at that step's posture (main movement has stopped).
  * Press 0 on the first frame where they START MOVING OUT of it (0 = transition / other / idle). Movement between postures is labelled 0.
  * A mark holds until the next mark. Frames before the first mark are unlabeled and are not exported.
  * Label by SEQUENCE ORDER: a clip goes through the steps in order 1..N. Some steps are the same posture as an earlier step (the dataset's stills
    for Tadasana step 5 = step 1, step 4 = step 2; Bhujangasana step 3 = step 1: the performer returns to the start). Mark the return as its own
    step number (e.g. 5), not as 1 - the thumbnails for such steps will look identical on purpose.
  * If a step is genuinely repeated in the clip, mark every occurrence. Reference thumbnails (right panel) come from the labelled STILLS of the same
    asana and Right/Wrong label; stills are horizontally stretched and shot from other viewpoints - use them for posture identity only.
  * Do not look at any rule-engine or model output while annotating (needed for an independent second annotator).
Each annotator writes their own file (audit/step_annotations_<name>.json) so a second annotator can be compared for agreement later.

KEYS  space play/pause | a,d -/+1 frame | q,e -/+1 s | z,c -/+10 s | [ ] prev/next mark | click timeline = seek | + - speed
      0..N mark | x delete mark at/before cursor | u undo | Enter mark video DONE and go next | n,p next/prev video | Esc save+quit
usage: python tools/annotate_steps.py --annotator NAME [--start IDX]      # GUI
       python tools/annotate_steps.py --annotator NAME --check            # headless validation + CSV export
       python tools/annotate_steps.py --selftest"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "audit"; BASE = ROOT / "data" / "raw" / "Yoga Postures Dataset"

class Marks:
    """frame -> label (0 = transition/other, k = step k). Pure logic, unit-tested by --selftest."""
    def __init__(self, n_frames, marks=None):
        self.n, self.m, self.hist = n_frames, {int(k): int(v) for k, v in (marks or {}).items()}, []
    def set(self, f, lab): self.hist.append((f, self.m.get(f))); self.m[f] = lab
    def delete_at_or_before(self, f):
        ks = [k for k in sorted(self.m) if k <= f]
        if ks: self.hist.append((ks[-1], self.m.pop(ks[-1]))); return ks[-1]
    def undo(self):
        if not self.hist: return
        f, old = self.hist.pop()
        if old is None: self.m.pop(f, None)
        else: self.m[f] = old
    def segments(self):
        """[(start, end_exclusive, label)] - only labelled stretches."""
        ks = sorted(self.m); return [(k, ks[i + 1] if i + 1 < len(ks) else self.n, self.m[k]) for i, k in enumerate(ks)]
    def label_at(self, f):
        ks = [k for k in sorted(self.m) if k <= f]; return self.m[ks[-1]] if ks else -1

def load_meta():
    vid = pd.read_csv(AUDIT / "inventory_videos.csv"); img = pd.read_csv(AUDIT / "manifest_dedup.csv")
    return vid, img, img.groupby("asana").step.max().to_dict()

def queue(f):
    return [l.strip() for l in Path(f).read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')] if f else []

def ann_path(name): return AUDIT / f"step_annotations_{name}.json"
def load_ann(name):
    p = ann_path(name); return json.loads(p.read_text()) if p.exists() else {"annotator": name, "videos": {}}
def save_ann(a):
    p = ann_path(a["annotator"]); tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(a, indent=1)); os.replace(tmp, p)

def problems(n_frames, marks, asana, label, nsteps):
    """Why a video's marks cannot be accepted as a real step annotation (shared by the GUI's Enter key and --check)."""
    mk = Marks(int(n_frames), marks); seg = mk.segments(); steps = [s[2] for s in seg if s[2] > 0]; w = []
    if len(mk.m) < 2: w.append(f"ONLY {len(mk.m)} MARK(S): a real clip needs at least an arrival and a departure mark")
    need = set(range(1, nsteps[asana] + 1)) - set(steps)
    if label == "Right" and need: w.append(f"MISSING STEPS {sorted(need)}")
    if steps != sorted(steps): w.append(f"NON-MONOTONE {steps}")
    if any(s > nsteps[asana] for s in steps): w.append("STEP > N")
    w += [f"SHORT SEGMENT ({s1 - s0} frames) label {lab} @ {s0}" for s0, s1, lab in seg if s1 - s0 < 15]
    return w

def check(name, vid, nsteps):
    """Validate + export CSV. Returns list of warnings."""
    a, rows, warn = load_ann(name), [], []
    for _, r in vid.iterrows():
        v = a["videos"].get(r.path.replace("\\", "/"))
        if not v or not v.get("done"): warn.append(f"NOT DONE  {r.path}"); continue
        warn += [f"{p}{'  [FORCED DONE]' if v.get('forced') else ''}  {r.path}" for p in problems(r.frames, v["marks"], r.asana, r.label, nsteps)]
        for s0, s1, lab in Marks(r.frames, v["marks"]).segments():
            rows.append(dict(video=r.path.replace("\\", "/"), asana=r.asana, label=r.label, angle=r.angle, start_frame=s0, end_frame=s1, step=lab,
                             start_s=round(s0 / r.fps, 2), end_s=round(s1 / r.fps, 2), annotator=name))
    pd.DataFrame(rows).to_csv(AUDIT / f"step_segments_{name}.csv", index=False)
    return warn

# ------------------------------------------------------------------ GUI
def run_gui(name, start, hook=None, only=None):  # hook(root, press_key, state): scripted-key test harness only
    import tkinter as tk, cv2
    import time
    try: import ctypes; ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp + true pixel sizes on scaled Windows displays
    except Exception: pass
    vid, img, nsteps = load_meta(); ann = load_ann(name)
    if only: vid = vid[vid.path.str.replace(chr(92), '/').isin(only)].reset_index(drop=True)   # annotate only the queued videos, in queue order
    root = tk.Tk(); root.title("PoseGuard step annotator - " + name)
    VW, VH, TW, TH = 560, min(700, root.winfo_screenheight() - 230), 300, 64  # video box adapts to the screen height
    COL = {-1: "#000000", 0: "#666666", 1: "#e6194b", 2: "#3cb44b", 3: "#4363d8", 4: "#f58231", 5: "#911eb4"}
    HEX2BGR = lambda h: tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))
    cv, side = tk.Canvas(root, width=VW, height=VH, bg="black", highlightthickness=0), tk.Canvas(root, width=TW, height=VH, bg="#202020", highlightthickness=0)
    tl = tk.Canvas(root, width=VW + TW, height=TH, bg="#101010", highlightthickness=0)
    st = tk.Label(root, text="", font=("Consolas", 10), justify="left", anchor="w", bg="#101010", fg="#dddddd")
    cv.grid(row=0, column=0); side.grid(row=0, column=1); tl.grid(row=1, column=0, columnspan=2); st.grid(row=2, column=0, columnspan=2, sticky="we")
    S = dict(i=start, cap=None, pos=0, mk=None, play=False, speed=1.0, msg="", photo=None, thumbs=[], t=0.0, confirm=False)

    def still_thumbs(asana, label):
        d = img[(img.asana == asana) & (img.label == label) & ~img.ambiguous]; out = []  # not `usable`: dedup drops steps that copy an earlier step
        sz = int(min(122, (VH - 16) / nsteps[asana] - 8))  # 2 thumbnails per step, one row per step
        for k in range(1, nsteps[asana] + 1):
            g = d[d.step == k].sort_values("time")
            for q in (0.25, 0.75):
                if len(g):
                    t = cv2.resize(cv2.imread(str(BASE / g.path.iloc[int(q * (len(g) - 1))])), (sz, sz)); cv2.rectangle(t, (0, 0), (62, 20), (0, 0, 0), -1)
                    cv2.putText(t, f"STEP {k}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HEX2BGR(COL[k]), 1); out.append((k, t))
        return out

    def photo(im_bgr):
        return tk.PhotoImage(data=cv2.imencode(".ppm", im_bgr)[1].tobytes())

    def open_video(i):
        i = max(0, min(len(vid) - 1, i)); S["i"] = i; r = vid.iloc[i]; key = r.path.replace("\\", "/")
        if S["cap"]: S["cap"].release()
        S["cap"] = cv2.VideoCapture(str(BASE / r.path)); e = ann["videos"].setdefault(key, dict(marks={}, done=False, fps=float(r.fps), n_frames=int(r.frames)))
        S["mk"] = Marks(int(r.frames), e["marks"]); S["play"] = False; seek(0)
        th = still_thumbs(r.asana, r.label); side.delete("all"); S["thumbs"] = []
        for j, (k, t) in enumerate(th):
            p = photo(t); S["thumbs"].append(p); sz = t.shape[0]; side.create_image(6 + (j % 2) * (sz + 6), 8 + (j // 2) * (sz + 8), anchor="nw", image=p)

    def persist():
        r = vid.iloc[S["i"]]; ann["videos"][r.path.replace("\\", "/")]["marks"] = {str(k): v for k, v in S["mk"].m.items()}; save_ann(ann)

    def render():
        r = vid.iloc[S["i"]]; S["cap"].set(cv2.CAP_PROP_POS_FRAMES, S["pos"]); ok, fr = S["cap"].read()
        if ok: draw(fr)
        timeline()

    def draw(fr):
        r = vid.iloc[S["i"]]; h, w = fr.shape[:2]; s = min(VW / w, VH / h); im = cv2.resize(fr, (int(w * s), int(h * s)))
        lab = S["mk"].label_at(S["pos"]); txt = {-1: "unlabeled", 0: "TRANSITION/OTHER"}.get(lab, f"STEP {lab}")
        cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1); cv2.putText(im, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HEX2BGR(COL[max(lab, -1)] if lab > 0 else "#ffffff"), 2)
        S["photo"] = photo(im); cv.delete("all"); cv.create_image(VW // 2, VH // 2, image=S["photo"])
        done = ann["videos"][r.path.replace("\\", "/")]["done"]
        st.config(text=f"[{S['i'] + 1}/{len(vid)}] {r.asana} | {r.label} | angle {r.angle} | steps 1..{nsteps[r.asana]} | {'DONE' if done else 'todo'}    "
                       f"frame {S['pos']}/{int(r.frames)}  t={S['pos'] / r.fps:6.1f}s  speed x{S['speed']}  marks={len(S['mk'].m)}\n"
                       f"1..{nsteps[r.asana]} = performer just ARRIVED at that step's posture | 0 = they start MOVING OUT / idle / no matching step | Enter = clip finished\n{S['msg']}")

    def timeline():
        tl.delete("all"); W, n = VW + TW, S["mk"].n
        for s0, s1, lab in S["mk"].segments():
            tl.create_rectangle(s0 / n * W, 8, s1 / n * W, 44, fill=COL.get(lab, "#888"), outline=""); tl.create_text((s0 + s1) / 2 / n * W, 26, text=str(lab), fill="white")
        r = vid.iloc[S["i"]]
        for t in range(0, int(n / r.fps) + 1, 10): tl.create_line(t * r.fps / n * W, 44, t * r.fps / n * W, 52 if t % 30 else 58, fill="#888")
        x = S["pos"] / n * W; tl.create_line(x, 2, x, 62, fill="#ffee00", width=2)

    def seek(f):
        S["pos"] = max(0, min(int(S["mk"].n) - 1, int(f))); render()

    def tick():
        now = time.perf_counter()
        if S["play"]:
            fps = float(vid.iloc[S["i"]].fps); adv = min(8, max(1, round((now - S["t"]) * fps * S["speed"])))  # wall-clock accurate: skip frames if decode is slow (clamped after stalls)
            for _ in range(adv - 1): S["cap"].grab()
            ok, fr = S["cap"].read(); S["pos"] = min(S["mk"].n - 1, S["pos"] + adv)
            if ok:
                draw(fr); timeline()
            else: S["play"] = False
        S["t"] = now; root.after(15, tick)

    def next_mark(d):
        ks = sorted(S["mk"].m); c = [k for k in ks if (k > S["pos"] if d > 0 else k < S["pos"])]
        if c: seek(c[0] if d > 0 else c[-1])

    def key(e):
        k, n = e.keysym, nsteps[vid.iloc[S["i"]].asana]; S["msg"] = ""
        if k != "Return": S["confirm"] = False
        if k == "space": S["play"] = not S["play"]; S["t"] = time.perf_counter()
        elif k in ("a", "Left"): seek(S["pos"] - 1)
        elif k in ("d", "Right"): seek(S["pos"] + 1)
        elif k == "q": seek(S["pos"] - 30)
        elif k == "e": seek(S["pos"] + 30)
        elif k == "z": seek(S["pos"] - 300)
        elif k == "c": seek(S["pos"] + 300)
        elif k == "bracketleft": next_mark(-1)
        elif k == "bracketright": next_mark(1)
        elif k in ("plus", "equal"): S["speed"] = min(4.0, S["speed"] * 2)
        elif k == "minus": S["speed"] = max(0.5, S["speed"] / 2)
        elif k.isdigit() and len(k) == 1:
            if int(k) > n: S["msg"] = f"step {k} does not exist for this asana (1..{n})"
            else: S["mk"].set(S["pos"], int(k)); S["msg"] = f"marked {'transition/other' if k == '0' else 'STEP ' + k} from frame {S['pos']}"; persist()
        elif k == "x": f = S["mk"].delete_at_or_before(S["pos"]); S["msg"] = f"deleted mark at frame {f}"; persist()
        elif k == "u": S["mk"].undo(); S["msg"] = "undo"; persist()
        elif k == "Return":
            r = vid.iloc[S["i"]]; pr = problems(r.frames, S["mk"].m, r.asana, r.label, nsteps)
            if any(p.startswith("ONLY") for p in pr):  # <2 marks can never be a valid annotation: hard block, no force
                S["msg"] = "CANNOT FINISH: fewer than 2 marks. Mark when the performer ARRIVES at a step (its number) and when they LEAVE it (0). See the legend above."; render(); return
            if pr and not S["confirm"]:  # refuse to silently accept an empty/incomplete annotation; a second Enter forces it and records forced=True
                S["msg"] = "NOT MARKED DONE: " + "; ".join(pr[:3]) + "   -> fix it, or press Enter again to force"; S["confirm"] = True; render(); return
            persist(); e = ann["videos"][r.path.replace("\\", "/")]; e["done"], e["forced"] = True, bool(pr); save_ann(ann); open_video(S["i"] + 1); return
        elif k == "n": persist(); open_video(S["i"] + 1); return
        elif k == "p": persist(); open_video(S["i"] - 1); return
        elif k == "Escape": persist(); root.destroy(); return
        render()

    tl.bind("<Button-1>", lambda e: seek(e.x / (VW + TW) * S["mk"].n)); tl.bind("<B1-Motion>", lambda e: seek(e.x / (VW + TW) * S["mk"].n))
    root.bind("<Key>", key); root.protocol("WM_DELETE_WINDOW", lambda: (persist(), root.destroy()))
    open_video(start); tick()
    if hook:
        class E:  # minimal stand-in for a Tk key event
            def __init__(self, k): self.keysym = k
        root.after(600, lambda: hook(root, lambda k: key(E(k)), S))
    root.mainloop()

# ------------------------------------------------------------------ self-test (hand-constructed fixtures, pure logic only)
def selftest():
    m = Marks(1000)
    m.set(100, 1); m.set(300, 0); m.set(350, 2); m.set(700, 0)
    assert m.segments() == [(100, 300, 1), (300, 350, 0), (350, 700, 2), (700, 1000, 0)], m.segments()
    assert [m.label_at(f) for f in (0, 99, 100, 299, 300, 350, 699, 700, 999)] == [-1, -1, 1, 1, 0, 2, 2, 0, 0]
    assert m.delete_at_or_before(720) == 700 and m.segments()[-1] == (350, 1000, 2)   # deletes nearest mark at/before cursor
    m.undo(); assert m.m[700] == 0                                                     # undo restores it
    m.set(350, 3); m.undo(); assert m.m[350] == 2                                      # undo of an overwrite restores the old label
    m.set(555, 1); m.undo(); assert 555 not in m.m                                     # undo of a new mark removes it
    vid = pd.DataFrame([dict(path="Videos/T/x.mp4", asana="Tadasana", label="Right", angle=1, frames=1000, fps=30.0)])
    a = {"annotator": "selftest", "videos": {"Videos/T/x.mp4": {"marks": {"100": 1, "300": 0, "350": 2, "700": 0}, "done": True}}}
    global load_ann, AUDIT; import tempfile; old = (load_ann, AUDIT); load_ann = lambda n: a; AUDIT = Path(tempfile.mkdtemp())
    try:
        w = check("selftest", vid, {"Tadasana": 5}); assert any("MISSING STEPS [3, 4, 5]" in x for x in w), w   # Right video must contain all 5 steps
        assert not any("NON-MONOTONE" in x for x in w)
        a["videos"]["Videos/T/x.mp4"]["marks"] = {"100": 2, "300": 1, "310": 0}; w = check("selftest", vid, {"Tadasana": 5})
        assert any("NON-MONOTONE" in x for x in w) and any("SHORT SEGMENT (10 frames)" in x for x in w), w
        csv = pd.read_csv(AUDIT / "step_segments_selftest.csv"); assert list(csv.step) == [2, 1, 0] and csv.start_s.iloc[1] == 10.0
    finally: load_ann, AUDIT = old
    ns = {"Tadasana": 5}
    p = problems(2127, {0: 1}, "Tadasana", "Right", ns)                                 # the real failure seen: one mark at frame 0 for the whole clip
    assert any("ONLY 1 MARK" in x for x in p) and any("MISSING STEPS [2, 3, 4, 5]" in x for x in p), p
    good = {300: 1, 400: 0, 450: 2, 600: 0, 650: 3, 800: 0, 850: 4, 1000: 0, 1050: 5, 1200: 0}   # hand-built complete Tadasana Right annotation
    assert problems(2127, good, "Tadasana", "Right", ns) == [], problems(2127, good, "Tadasana", "Right", ns)
    assert problems(2127, {300: 1, 400: 0}, "Tadasana", "Wrong", ns) == []             # Wrong clips need not contain every step
    print("selftest OK")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--annotator"); ap.add_argument("--start", type=int, default=None); ap.add_argument("--queue", help="text file, one video path per line (forward slashes); only these are shown/checked"); ap.add_argument("--check", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: selftest(); sys.exit()
    if not a.annotator or a.annotator.strip().upper() in {"YOURNAME", "NAME", "ANNOTATOR"}:
        sys.exit("--annotator must be your real name (annotator identity is recorded with every annotation; the usage example's placeholder is not a name)")
    if a.check:
        vid, _, ns = load_meta(); q = queue(a.queue); vid = vid[vid.path.str.replace(chr(92), '/').isin(q)] if q else vid; w = check(a.annotator, vid, ns); print(*w, sep=chr(10)); av = load_ann(a.annotator)['videos']; print(f"{len(w)} warnings; {sum(av.get(x.replace(chr(92), '/'), {}).get('done', False) and len(av[x.replace(chr(92), '/')]['marks']) >= 2 for x in vid.path)}/{len(vid)} videos truly done (done AND >= 2 marks)")
    else:
        vid, _, _ = load_meta(); q = queue(a.queue); vid = vid[vid.path.str.replace(chr(92), '/').isin(q)].reset_index(drop=True) if q else vid; ann = load_ann(a.annotator); done = {k for k, v in ann["videos"].items() if v.get("done") and len(v["marks"]) >= 2}
        s = a.start if a.start is not None else next((i for i, p in enumerate(vid.path) if p.replace(chr(92), "/") not in done), 0); run_gui(a.annotator, s, only=q or None)
