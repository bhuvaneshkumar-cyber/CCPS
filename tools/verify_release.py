"""Verify a transferred bundle on the TARGET machine, end to end. Nothing is trusted until every step passes.
usage: python tools/verify_release.py BUNDLE.zip [--extract-to DIR] [--skip-tests] [--skip-data] [--gpu]
       python tools/verify_release.py --checkout DIR [--gpu]      (git clone: steps 1-2 are replaced by git itself; the committed manifest is checked from step 3)
Steps: (1) zip SHA-256 vs the .sha256 file that travelled with it; (2) zip CRC + extract; (3) every file's size + SHA-256 vs the manifest, and no unexpected files;
(4) weights + dataset re-fetched/verified against the pinned hashes; (5) environment check; (6) every test suite; (7) reproducibility probes on real frames."""
import argparse, hashlib, json, subprocess, sys, zipfile
from pathlib import Path

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

ap = argparse.ArgumentParser(); ap.add_argument("bundle", nargs="?"); ap.add_argument("--checkout"); ap.add_argument("--extract-to", default="poseguard_verified"); ap.add_argument("--skip-tests", action="store_true")
ap.add_argument("--skip-data", action="store_true"); ap.add_argument("--gpu", action="store_true"); a = ap.parse_args()
if bool(a.bundle) == bool(a.checkout): ap.error("give either BUNDLE.zip or --checkout DIR")
dest = Path(a.checkout or a.extract_to).resolve(); fails = []

def step(n, ok, msg):
    print(f"[{n}] {'PASS' if ok else 'FAIL'} {msg}")
    if not ok: fails.append(f"{n}: {msg}")

if a.bundle:
  zp = Path(a.bundle); side = zp.with_name(zp.name + ".sha256"); want = side.read_text().split()[0] if side.exists() else None
  step("1 zip hash", want is not None and sha(zp) == want, zp.name if want else f"{side.name} not found next to the bundle - cannot prove the zip arrived intact")
  if want is None or sha(zp) != want: sys.exit("STOP: bundle hash mismatch or missing - re-copy the bundle; nothing was extracted")
  with zipfile.ZipFile(zp) as z:
      bad = z.testzip(); step("2 zip CRC", bad is None, "all members readable" if bad is None else f"corrupt member {bad}"); z.extractall(dest)
man = json.load(open(dest / "MANIFEST.json"))
wrong = [m["path"] for m in man if not (dest / m["path"]).exists() or (dest / m["path"]).stat().st_size != m["bytes"] or sha(dest / m["path"]) != m["sha256"]]
step("3 manifest", not wrong, f"{len(man)} files verified" if not wrong else f"{len(wrong)} files missing/changed: {wrong[:5]}")
known = {m["path"] for m in man} | {"MANIFEST.sha256", "MANIFEST.json", "ENVIRONMENT.json", "PROBES.json"}
extra = [p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file() and p.relative_to(dest).as_posix() not in known and not set(p.relative_to(dest).parts) & {"__pycache__", ".git", ".venv", "venv", "release", "smoke"} and p.parts[len(dest.parts):][:2] != ("data", "raw") and p.suffix not in (".zip", ".pt")]
step("3b no extras", not extra, "no unexpected files" if not extra else f"unexpected: {extra[:5]}")

def run(cmd): return subprocess.run([sys.executable, *cmd], cwd=dest, capture_output=True, text=True)
tail = lambda r, n=1: "\n    ".join((r.stdout + r.stderr).strip().splitlines()[-n:])
if not a.skip_data: r = run(["tools/fetch_data.py"]); step("4 data+weights", r.returncode == 0, tail(r))
r = run(["tools/env_check.py", "--pins", "ENVIRONMENT.json"] + (["--require-cuda"] if a.gpu else [])); step("5 environment", r.returncode == 0, "\n    " + tail(r, 5))
if not a.skip_tests:
    for t in sorted((dest / "tests").glob("test_*.py")): r = run([f"tests/{t.name}"]); step(f"6 {t.name}", r.returncode == 0, tail(r) if r.returncode == 0 else (r.stdout + r.stderr).strip()[-300:])
if not a.skip_data and json.load(open(dest / "PROBES.json")): r = run(["tools/probe_check.py"] + (["--gpu"] if a.gpu else [])); step("7 probes", r.returncode == 0, tail(r))
print("\nOVERALL:", "ALL CHECKS PASSED - safe to use this copy" if not fails else "FAILED - do not use this copy:\n  " + "\n  ".join(fails)); sys.exit(1 if fails else 0)
