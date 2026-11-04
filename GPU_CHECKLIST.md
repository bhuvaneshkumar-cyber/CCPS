# GPU checklist (RTX 5080 machine)

Run the steps top to bottom, from the repository root, in one PowerShell window with the venv active. Each step says what "done" looks like. **If a step fails, stop and report it. Do not work around it**: no editing frozen files, no re-running a look.
The commands are Windows PowerShell. On Linux, use `python3 -m venv .venv; source .venv/bin/activate`, and `rm -rf` instead of `Remove-Item -Recurse -Force`; everything else is the same.

## A. Set up and verify (about 30 min, most of it the 3.2 GB download)
- [ ] **A1. Check the GPU driver.** Run `nvidia-smi`. Done when it shows the RTX 5080 and "CUDA Version" is **12.8 or higher**. If it is lower, update the NVIDIA driver first.
- [ ] **A2. Clone the repo and confirm the commit.**
  ```powershell
  git clone <YOUR_GITHUB_URL> PoseGuard; cd PoseGuard
  git log -1 --format="%H %s"
  ```
  Done when the hash matches the one you pushed from the old machine.
- [ ] **A3. Create the Python environment.** Python **3.14** is required (`py -3.14 --version`).
  ```powershell
  py -3.14 -m venv .venv; .venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements.txt
  pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93
  ```
- [ ] **A4. Verify the clone, the data, the environment, all tests and the real-frame probes.** This step downloads the dataset and weights and checks their hashes. Offline alternative: first copy `yoga.zip` into `data\` and `yolo26m-pose.pt` into `weights\` from USB.
  ```powershell
  python tools/verify_release.py --checkout . --gpu
  ```
  Done when it ends with **`ALL CHECKS PASSED`**. It checks every file's SHA-256 against `MANIFEST.json`, so line-ending or partial-clone damage is caught here. It also checks CUDA, a GPU matmul, the pinned torch, ultralytics and cv2 versions, all 7 test suites, and 8 real-frame probes against the CPU keypoints (3 px tolerance).
- [ ] **A5. Run the GPU plumbing check.** Training here uses Right clips only; the "Wrong" side is a time-reversed null. The numbers are not results.
  ```powershell
  python scripts/phase4_train.py --smoke
  python scripts/phase4_test.py --smoke
  Remove-Item -Recurse -Force phase4\smoke
  ```
  Done when both scripts finish without error and the first lines say `device cuda:0`.

## B. Dense keypoints and cross-device check (about 1-2 h, runs unattended)
- [ ] **B1. Extract keypoints for all 80 videos at every frame (152,039 frames).** The script resumes if interrupted: just run it again.
  ```powershell
  python scripts/extract_video_full.py
  ```
  Done when it prints `done: ...kpts_video_full_cuda.npz`, about 31 MB. Note any `WARNING clip` lines: they are reported, not fixed.
- [ ] **B2. Cross-device check.** It also re-runs the 8,259 stills on the GPU.
  ```powershell
  python tools/cross_device_check.py
  ```
  This writes `phase4/cross_device_report.txt`. PASS means the CPU-derived Phase 0-3 results carry over. FAIL is a result too: keep it and report it, and continue only after deciding with me how to word it.
- [ ] **B3. Commit both outputs** (keypoints first, before any model exists):
  ```powershell
  git add phase4/kpts_video_full_cuda.npz phase4/kpts_stills_640_cuda.npz phase4/cross_device_report.txt
  git commit -m "Phase 4 input: dense GPU keypoints + cross-device check (RTX 5080)"; git push
  ```

## C. Freeze the protocol (owner, about 15 min). Nothing trains until this is done
- [ ] **C1.** Open `phase4/PROTOCOL_PHASE4.md` and resolve **DECISION 1** (headline arm; recommended: LOAO) and **DECISION 2** (success criterion). Write your choices into the file.
- [ ] **C2.** Replace the line `STATUS: DRAFT - ...` with `STATUS: FROZEN by <your name> <date>`.
- [ ] **C3. Commit and push before training.** The pushed commit time is public proof that the rules came before the results.
  ```powershell
  git add phase4/PROTOCOL_PHASE4.md; git commit -m "Phase 4 protocol frozen"; git push
  ```

## D. Train (Right clips only; about 20-60 min)
- [ ] **D1.** Train 5 arms: CHRONO plus LOAO folds 1-4, each a 5-seed ensemble.
  ```powershell
  python scripts/phase4_train.py
  ```
  Done when it prints `frozen: phase4\frozen_temporal_cuda.json. No Wrong clip was read.` It refuses to run if the protocol is still a draft or a freeze already exists.
- [ ] **D2.** Commit the frozen models before the look:
  ```powershell
  git add phase4/frozen_temporal_cuda.json phase4/model_*_cuda.pt; git commit -m "Phase 4 models frozen (dev only)"; git push
  ```

## E. The single look (about 5-15 min)
- [ ] **E1.** Score the frozen models once.
  ```powershell
  python scripts/phase4_test.py
  ```
  It refuses if the protocol, `temporal.py`, the keypoints or any model changed since D1, and it records the look in `phase4/TEST_TOUCHED_cuda.json`. **Run it once.** A second run needs `--second-look` and is reported as one.
- [ ] **E2.** Commit the results:
  ```powershell
  git add phase4/phase4_test_report_cuda.txt phase4/test_rows_cuda.csv phase4/TEST_TOUCHED_cuda.json
  git commit -m "Phase 4 single look (RTX 5080)"; git push
  ```
- [ ] **E3.** Bring `phase4/cross_device_report.txt` and `phase4/phase4_test_report_cuda.txt` back for the write-up. Whatever the verdict is (SEPARATES, adds nothing over persistence, or NEGATIVE), it is reported as written. Per the timebox rule, a negative result is not followed by architecture changes.

## Later (not scripted yet; each needs its rule written down first)
- [ ] Phase 0 sec. 19 mark-lag check on the dense keypoints (24 annotated clips).
- [ ] Phase 6 live latency benchmark (backbone + forecaster, fixed imgsz 640) and the raised-arm mirroring check (`tools/verify_lr.py --video ... --raised left|right`) on the actual camera.

## Rules on this machine
- From the clone onward, this machine is the only place work happens. The old laptop is a read-only archive: don't edit both.
- Every GPU output carries `cuda` in its name and sits next to, never over, a CPU file.
- Never edit a frozen file: `phase3/PROTOCOL*.md`, `frozen_config.json`, `TEST_TOUCHED.json`, `VIDEO_TOUCHED.json`, `v2/STOPPED.json`, and after C2 also `phase4/PROTOCOL_PHASE4.md`, `frozen_temporal_cuda.json`, `model_*_cuda.pt`, `TEST_TOUCHED_cuda.json`.
