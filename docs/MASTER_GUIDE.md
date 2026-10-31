# Master Project Document & Learning Guide: lung nodule detection v1.0

> **Goal of this file:** after reading it you can **explain every part**, **defend every choice**, and **rebuild the idea** from memory.
> Sources: `README.md`, `docs/DECISIONS.md`, `docs/REALITY_CHECK.md`, and the code in `src/lungdet/`. A number here is either copied from those files or computed in front of you. `NOT MEASURED` means exactly that.
> Style: short points, **bold keywords**, "because" arrows. No quizzes; the viva Q&A in §12 is for rehearsal.

---

## Part A · The project

### 1. The story in 6 lines
- **Task:** in a full chest CT, find every **lung nodule** (≥ 3 mm, marked by ≥ 3 of 4 radiologists). Not cancer, only "is there a nodule here?"
- **Data:** **LIDC-IDRI**, 897 usable scans, 1,192 nodules. Split **by patient**: TRAIN 628 · VAL 134 · **TEST 135 scans / 225 nodules**.
- **Method (FINDERS):** classical **blob detector** proposes ~2,643 spots per scan → **3D CNN** (5 seeds) scores each spot → **occlusion maps** show why.
- **Result:** **TEST CPM 0.823** [0.767–0.868]. 83.6% of nodules found at **1 false alarm per scan**.
- **Honest bits:** the fancy HUSH tricks only **tied** the plain CNN on VAL; small nodules (3–6 mm) are the weak spot (63%); one dataset only.
- **Why it is a good CV-engineer project:** every step has a **measured reason**, and the failures are written down.

🖼 **One mental image: minesweeping.**
- **Metal detector** (blob detector) beeps at everything: ~20,860 peaks → keep 2,643.
- **Sniffer dog** (3D CNN) sniffs each beep: mine or junk?
- **Camera** (occlusion) photographs *why* the dog said "mine".

### 2. Vocabulary you must own (8 words)
| Word | Meaning | Picture |
|---|---|---|
| **CT / HU** | 3D X-ray volume; each voxel in **Hounsfield units** (air −1000, water 0, bone > 300) | a stack of grey photos |
| **Nodule** | round-ish spot 3–30+ mm in the lung | a pea in a sponge |
| **Vessel** | tube-shaped lung structure; **looks like a dot in one slice** | a straw cut across |
| **Candidate** | a spot the blob detector proposes | a beep |
| **FROC** | sensitivity vs **false alarms per scan** curve | "how many peas found per wrong alarm" |
| **CPM** | mean sensitivity at 1/8, 1/4, 1/2, 1, 2, 4, 8 FP/scan | one number for the whole curve |
| **Bootstrap CI** | resample *scans* 1,000× → spread of the score | re-draw the test set 1,000 times |
| **Ablation** | blank one input and see what breaks | pull one wire, watch the lamp |

---

## Part B · The pipeline layer by layer (FINDERS)

Each layer: **what → why (because) → how it was checked → file**.

### F · Fetch  (`scripts/01_fetch.py`, `02_ground_truth.py`, `src/lungdet/fetch/`)
- **What:** download scans with `idc-index`; read radiologist marks with `pylidc`.
- **Because** the IDC bucket is public → no login, parallel, resumable. **128 GB in 1 h 59 min = 18 MB/s.**
- **Ground-truth rule (as LUNA16):** slice ≤ 2.5 mm; nodule = **≥ 3 mm and ≥ 3 of 4 readers**.
  - **Ignore zones:** marks by 1–2 readers, < 3 mm, or "non-nodule" are neither hit nor false alarm → because they are *unclear*, punishing either way is unfair.
- **Split by patient (70/15/15, seed 42)** → because the same patient in two splits **leaks**.
- **Gate (scan #1):** 512×512×133, 2.5 mm slices, HU 1st percentile **−2048** → that is scanner **padding**, not air → Phase 1 clips it.

### I · Isolate  (`scripts/03_isolate.py`, `src/lungdet/isolate/`)
- **What (NET + TRACE):** clip HU to [−1000, 1000] → resample to **1 mm cubes** → lung mask → crop → `.npz`.
- **Because** a "5 mm" nodule must be 5 voxels in every scan (one ruler), and nodules only live inside lungs.
- **Lung mask TRACE:** threshold −320 HU → remove border air **per slice** → keep 2 biggest blobs → **3D closing r = 15 mm** → fill holes.
- **Worked example (why r = 15):** on TRAIN sweep set (151 nodules inside mask)
  - r = 10 → 91.4% · **r = 15 → 98.0%** · r = 20 → 98.7%
  - 10→15 gains **+6.6 pts**; 15→20 gains **+0.7 pts (≈ 1 nodule)** but adds **+4.7% lung volume** (more false candidates) → knee at 15.
- **Bug that taught us:** removing border air in **3D** → one thin leak deleted a whole lung (31 scans < 2 L, 52 nodules). Fix: **per slice** → a leak costs one slice.
- **Final:** 99.6% of GT nodules inside the mask; median HU at nodule centres **11 HU** (so coordinates are right).
- **Accepted flaw:** 5 scans (0619, 0697, 0403, 0523, 0655) have masks including body air. Because a too-big mask adds false candidates (the CNN can reject) but does not lose nodules.

### N · Nominate  (`scripts/04_nominate.py`, `nominate/blobs.py`)
- **What:** **Laplacian-of-Gaussian (LoG)** at σ = 1, 1.5, 2.5, 4, 6, 9, 13 mm. **Diameter ≈ 3.5 σ** → 3.5 to **45.5 mm** (largest GT nodule 46.8 mm).
- **Because** a ball answers a blob filter more than a tube (vessel) or sheet (wall): at matched size ball **0.86×** contrast vs tube **0.73×** (`test_ball_beats_vessel`).
- **Extra rules:** outside-mask set to −1000 HU; peak HU > −700; greedy NMS inside the blob radius.
- **The measured surprise:** first run TRAIN recall **92.8%**. My guess (bundling) was **wrong**: 56 of 59 misses were just **below the 3,000 cap** (faint ~8 mm blobs lose to bright vessels).
  - Fix: **per-size top-1200** (each size gets its own quota) → recall **96.0% at 2,643/scan** vs global top-N needing **5,000** (−47%).
- **Ceiling:** TEST 218/225 = **0.969** → no later stage can beat it.
- **R0 floor:** rank by blob strength only → TEST CPM **0.058**. The CNN is not decoration.

### D · Discard  (`scripts/05_build_patches.py`, `06_train_cnn.py`, `discard/`)
- **What:** a 3D CNN looks at a **32 mm cube** around each candidate → probability "real nodule".
- **Network (`discard/cnn.py:Small3DNet`):** 4 conv blocks 32→64→128→256 channels, cubes 32³→16³→8³→4³, global average, dropout 0.3, 1 logit. ≈ 1.5 M weights.
- **Because** vessel vs nodule is a **3D-shape** question; one 2D slice cannot tell a cut straw from a pea.
- **Training sample per scan:** all positives + GT centre ×5 (jitter ±2 mm) + 150 random negatives + 50 hardest. **Balanced batches** (25% positive) because positives are < 5%.
  - Full set would be 2,643 × 628 ≈ **1.66 M** cubes, so we sample.
- **Train 10 epochs, 5 seeds**, ensemble the 5 → because one seed moves ±0.01 by chance.
- **HUSH (`discard/hessian.py:hessian_channels`):** the input has **3 channels = [HU, blobness, vesselness]** from Hessian eigenvalues (max over σ = 1, 2, 4 mm).
  - 🖼 You hand the dog a **ball-shaped smell** and a **tube-shaped smell** along with the picture.
  - **H** = Hessian channels (shipped) · **U** = masked-cube pretraining · **S** = size head. U and S did not earn their place.

### E · Explain  (`scripts/11_explain.py`, `explain/occlusion.py`)
- **What:** slide an **8 mm block of normal lung (−850 HU)** over the cube, stride 4 mm; where the score **drops**, the model "looked".
- **Scored, not just pretty:** **hit** (peak inside nodule radius) and **concentration**. Trained: **100% / 0.91** (40 nodules); random weights: **0% / 0.00**.
- **Because** in project v2 layer-4 Grad-CAM scored **worse than an untrained network**.
- **Caveat:** random maps are almost flat → a weak test; the 40 nodules were the *most confident* (easy) ones.

### R · Rate  (`scripts/10b_test_report.py`, `rate/`)
- **Hit rule (`rate/matching.py:match`):** detection centre inside the nodule radius; 2 detections on one nodule = **1 hit**.
- **FROC / CPM (`rate/froc.py`)**, **bootstrap over scans** (`rate/bootstrap.py`) → because candidates of one scan are not independent.
- **TEST opened once:** `TEST_OPENED` marker; a second run refuses unless `FORCE=1`.

### S · Ship  (`predict.py`, `app/app.py`, `ship/`)
- **One pipeline** (`ship/pipeline.py:run_scan`) re-uses the Phase 1–3 functions → because a copy would drift from what TEST measured.
- **Operating point:** threshold = the **(⌊rate × n_scans⌋ + 1)-th strongest false candidate on VAL**.
  - Worked example: VAL has 134 scans, rate = 1 → take the **135th** strongest false candidate; keep `prob >` that → at most 134 false alarms ≈ **1 per scan**.
- **Merge** nearby detections within max(3 mm, √3 σ). ⚠ **Not measured on TEST.**
- **Device:** CUDA → MPS → CPU (`common/device.py`, `FORCE_CPU=1`).
- **Real run:** scan 0405 on a Mac CPU = **43 min** (2,522 candidates × 5 models × ~0.2 s).

---

## Part C · The numbers, worked by hand

### 3. FROC and CPM
- **Sensitivity** = nodules found ÷ all nodules. At 1 FP/scan: **188 / 225 = 0.836**.
- **CPM** = average of 7 sensitivities:
  `(0.702 + 0.733 + 0.787 + 0.836 + 0.867 + 0.902 + 0.933) ÷ 7 = 5.760 ÷ 7 = 0.823`
- **Why CPM, not accuracy?** Because ~97% of candidates are negatives; accuracy would be ~97% for "say no to everything".

### 4. The funnel (counts per scan)
| Stage | Count | Because |
|---|:-:|---|
| LoG peaks after bundling | ~20,860 | all local maxima |
| Kept candidates | **2,643** | per-size top-1200 → TRAIN recall 96.0% |
| Operating point 1 FP/scan | ≈ **1** | VAL-picked threshold |

### 5. Results table (TEST, 135 scans, 225 nodules)
| Model | CPM | 95% CI | @ 1/8 FP | @ 1 FP | @ 4 FP |
|---|:-:|:-:|:-:|:-:|:-:|
| R0 no learning | 0.058 | 0.031–0.094 | 0.044 | 0.053 | 0.062 |
| R2 plain 3D CNN | 0.797 | 0.725–0.847 | 0.591 | 0.831 | 0.884 |
| **R5 HUSH (shipped)** | **0.823** | 0.767–0.868 | 0.702 | 0.836 | 0.902 |

- **R5 − R2 = +0.026 [+0.001, +0.078]** → lower bound barely above 0 → **weak**.
- **Where R5 wins:** 1/8 FP (0.702 vs 0.591). At 1 FP they are the same (0.836 vs 0.831).
- **TEST > VAL (0.823 vs 0.779)?** Probably a higher candidate ceiling on TEST (0.969 vs ~0.90); `NOT MEASURED` as the cause.

### 6. The staircase on VAL (one change per rung)
| Rung | Change | VAL CPM | Verdict |
|---|---|:-:|---|
| R0 | blob strength only | 0.088 | floor |
| R1 | XGBoost, ~20 hand features | 0.275 | far below CNN |
| **R2** | **3D CNN** | **0.778** | big jump (+0.50) |
| R3 | focal + hard neg + rot90 + TTA together | 0.717 | **worse** |
| R4 | fuse R2 + R1 | 0.780 | tie |
| R5 | + H channels | 0.779 | tie, but explainable |
| R6 | + U pretraining | 0.776 | tie |
| R7 | + S size head | 0.772 | tie |

- **Rule that kept us honest:** a rung counts only if its 5-seed mean beats the rung below by **more than the larger seed SD** (≈ 0.01).
- **Ablation (R5, 1 seed, VAL):** all channels 0.776 · no vesselness 0.695 · **no blobness 0.663** · **HU only 0.397** → the net *uses* the shape channels (caveat: a blanked channel is out-of-distribution).
- **Length check:** 30 vs 10 epochs +0.009 / +0.011 < SD → keep 10. TRAIN AUC 1.000 vs VAL 0.988 → mild overfit, no underfit.

---

## Part D · What went wrong (and what each failure taught)

| # | Failure | Lesson (because →) |
|--:|---|---|
| 1 | 3D border-air removal deleted whole lungs (31 scans) | one leak path ⇒ global damage → work **per slice** |
| 2 | Closing r = 10 missed 16 big wall nodules | pick settings from a **sweep on TRAIN**, not by gut |
| 3 | "Padding is tissue-like" hypothesis | **FALSE** (−3024 or none) → test the hypothesis before building on it |
| 4 | Candidate cap 3,000 → recall 92.8% | my bundling suspicion was wrong; **measure the cause** |
| 5 | R3 bundled 4 tricks → worse | **one change per run**, or you can't say which hurt |
| 6 | 3–6 mm nodules: 63% found, 3/27 never proposed | small + faint is the open problem |
| 7 | 30-epoch run OOM on the 40 GB cap | keep VAL cubes in RAM; make batch size an argument |
| 8 | v2 Grad-CAM worse than random | **always add a sanity check** to explanations |

---

## Part E · Limits (say these before the examiner does)
- **One dataset** (LIDC-IDRI). No other scanner or hospital.
- **Labels = radiologist consensus**, not biopsy; says nothing about cancer.
- **TEST is small** → CI ≈ ±0.05 CPM.
- **Slice ≤ 2.5 mm only.**
- **`predict.py` adds threshold + merge** that TEST did not measure.
- **Probability is a rank score**, not calibrated (`NOT MEASURED`).
- **VAL used for choices** → VAL slightly optimistic; headline uses TEST only.
- **Built in ~2 days**, no large hyper-parameter search.
- **Code licence not chosen yet** (add a `LICENSE` before reuse).

---

## Part F · Code map: read in this order

| # | File | Read it for | Size hint |
|--:|---|---|---|
| 1 | `configs/default.yaml` | every knob in one place | start here |
| 2 | `isolate/geometry.py` → `lungmask.py` | resample rule; TRACE | small |
| 3 | `nominate/blobs.py` (`log_peaks`, `bundle`) | the metal detector | small |
| 4 | `discard/patches.py` → `cnn.py` | cube cutting; `Small3DNet` | core |
| 5 | `discard/hessian.py` | blobness / vesselness from eigenvalues | core of HUSH |
| 6 | `rate/matching.py` → `froc.py` → `bootstrap.py` | how a number is made | core |
| 7 | `explain/occlusion.py` | `occlusion_map`, `score_map`, `random_like` | short |
| 8 | `ship/pipeline.py` | `run_scan` = the whole chain | glue |
| 9 | `tests/` | one test per layer on a fake chest | proof |

- Whole library ≈ **1,150 lines**; you can read it in an afternoon.
- **Script numbers:** `00` gate · `01–02` data · `03*` isolate · `04*` nominate · `05–09` discard/HUSH · `10*` TEST · `11` explain · `12–14` ship (full table in `scripts/README.md`).

### Rules behind the code: **PRISM**
- **P**atient split · **R**esample 1 mm · **I**nputs from pixels only (no radiologist ratings) · **S**eeds ×5 + CI, test once · **M**aps scored.

---

## Part G · Learning path (do these, in order)

1. **Run the tests:** `python -m pytest tests/ -q` → 41 pass in ~16 s on a CPU. Read one test per layer.
2. **Hand-draw the funnel** (§4) and the FINDERS table from memory → compare with §B.
3. **Re-derive CPM** (§3) with the 7 numbers → get 0.823.
4. **Read `Small3DNet`** and compute the shapes: 32³ → 16³ → 8³ → 4³ with 256 channels.
5. **Read `hessian_channels`**: note what ball vs tube eigenvalues look like (ball: 3 big negative; tube: 2 big, 1 near 0).
6. **Run the phantom** through `predict.py` (planted ball must be candidate rank 1).
7. **Write `docs/phases/PHASE_N.md` from memory**, then diff against this guide (your routine).
8. **Hand-rewrite one core function** of your choice (suggestion: `rate/froc.py:cpm_from_scores`).

### Concepts to study next (each tied to this project)
- **Convolution & receptive field** → why 4 pooling levels see a 32 mm cube.
- **Class imbalance** → balanced batches, hard negatives, why focal loss did *not* help here.
- **Bootstrap & paired differences** → why +0.026 [+0.001, +0.078] is "weak".
- **Data leakage** → patient split, no LUNA16 pretrained weights (they saw our test scans).
- **Saliency sanity checks** (Adebayo 2018) → why random-weights comparison matters.

---

## Part H · Viva rehearsal (short honest answers)

1. **Is 0.823 real?** One TEST set, opened once, CI 0.767–0.868. It means "on LIDC-IDRI, these 135 scans".
2. **Is HUSH better than a plain CNN?** Weakly: +0.026 [+0.001, +0.078] on TEST, a tie on VAL. Shipped for explainability and the small edge.
3. **Did you tune on TEST?** No. Every choice was on TRAIN or VAL; `TEST_OPENED` marker enforces one look.
4. **Why a blob detector first?** Cuts ~20,860 peaks to 2,643 without training; price = **0.969 ceiling**.
5. **Why not LUNA16 pretrained weights?** LUNA16 comes from LIDC → they have seen our test scans.
6. **Do the maps prove it looks at the nodule?** They show the score drops when the nodule is removed (100%/0.91) and not for random weights; weak test, easy cases.
7. **Does the demo give exactly 1 FP/scan?** Threshold picked on VAL; real rate on new scans `NOT MEASURED`.
8. **What next?** External test set · layer-wise randomisation for maps · fix the 5 bad masks · study 3–6 mm misses · report 3-of-4 vs 4-of-4 agreement.

---

## Part I · Where everything lives
| Need | Go to |
|---|---|
| Headline results, quickstart | `README.md` |
| Every choice + rejected alternative + number | `docs/DECISIONS.md` |
| Failures, CIs, tough questions | `docs/REALITY_CHECK.md` |
| Where am I? | `docs/PROJECT_MAP.md` |
| Which script does what | `scripts/README.md` |
| Figures for the PPT | `docs/figures/` |
| Release notes | `docs/RELEASE_v1.0.md` |
