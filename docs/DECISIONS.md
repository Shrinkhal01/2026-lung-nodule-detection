# Decisions log (Claude's working log; your own notes go in `phases/PHASE_N.md`)

## Phase 0: setup

| Used | Why | Rejected alternative | Number that justified it |
|---|---|---|---|
| **idc-index** download from the public IDC bucket | no login, parallel (s5cmd), resumable | NBIA Data Retriever (GUI + login) | speed test 4.94 MB/s; full run **128 GB in 1 h 59 min = 18.0 MB/s average** (peak ~25.9) |
| **/scratch** for data, env, caches | `/home` is shared and 100% full | `/home`; streaming through RAM only | `/home` 0.4 GiB free vs `/scratch` 2,683 GiB free |
| **Reuse the old venv** (copied to scratch) | it already has torch 2.14.1 + CUDA 13 | fresh torch install | torch ≈ 2.5 GB at pip's ~0.1–0.5 MB/s = hours |
| **pip `-c keep.txt`** (pins torch + numpy) | new packages must not silently replace torch | plain `pip install` | NOT MEASURED (prevention) |
| **`setuptools<81`** | pylidc imports `pkg_resources`, which newer setuptools removed | patching pylidc | import failed: `No module named 'pkg_resources'` |
| **pylidc shims** (`np.int`, `SafeConfigParser`) | old pylidc code on NumPy 2 / Python 3.13 | pinning old NumPy (breaks torch) | pylidc issue #69 (`np.int`) |
| **Public repo** + https clone | the server needs no GitHub key | private repo + deploy key | clone asked for a username while private |
| **Mac: project `.venv`, test kit only** | tests run on fake scans; no CT data on the Mac | full server stack on the Mac | Mac: 8 GB RAM, 22 GiB free |
| **`xgboost-cpu`** instead of `xgboost` | XGBoost scores ~20 numbers per candidate → CPU takes seconds | GPU `xgboost` wheel (+ NCCL libraries) | 5.8 MB vs ~500 MB at PyPI's **~0.14–0.19 MB/s** on this server |

### What the gate showed on LIDC-IDRI-0001 (real numbers, `scripts/00_gate_one_scan.py`)
- Volume **512 × 512 × 133**, slice **2.5 mm**, pixel **0.703 mm**, loaded in **0.8 s**.
- HU percentiles 1 / 50 / 99 = **−2048 / −932 / 420**.
  - **−2048** is not air. It is the scanner's **padding** outside the round field of view.
  - → Phase 1 must **clip to [−1000, …]** before the lung mask, otherwise "darker than air" pixels look like lung.
  - The gate's air check (1st percentile ≤ −900) passed **because of the padding** → it was too weak. Phase 1 checks air inside the body instead.
- **1 nodule**, marked by **4 of 4** readers, mean diameter **32.8 mm**.
  - It is **above the 30 mm cap** in `nominate.size_range_mm`. With that cap, this ground-truth nodule could never be found.
  - → **Open for Phase 2:** raise the cap (e.g. 40 mm), checked against the diameter distribution of all ground-truth nodules.

## Pipeline as per the code at the end of Phase 0 (answer key; commit after `470b626`)
1. **Environment check**: `scripts/00_check_env.py:main` → `check_packages`, `check_gpu`, `check_disk`.
2. **Settings**: `configs/default.yaml` → `src/lungdet/common/config.py:load_config` (expands `$SCRATCH`).
3. **Download**: started by an **inline idc-index command in tmux** (not a repo script) → `$SCRATCH/lidc_raw/lidc_idri/<PatientID>/<StudyUID>/CT_<SeriesUID>/*.dcm`.
4. **Download check / resume**: `scripts/01_fetch.py:main` → `src/lungdet/fetch/inventory.py:scan_series_folders` + `series_status` (done / partial / missing).
5. **pylidc on today's Python**: `src/lungdet/fetch/pylidc_compat.py:apply_shims` (np.int, SafeConfigParser) → `import_pylidc`; `write_pylidcrc` points pylidc at the DICOM root.
6. **Gate (one scan)**: `scripts/00_gate_one_scan.py:main` → `pl.query(pl.Scan)` → `scan.to_volume` → `scan.cluster_annotations` → GT rule (≥ 3 readers, ≥ 3 mm).
7. **Tests (Mac, CPU)**: `tests/test_fetch_inventory.py`, `tests/test_config_and_compat.py` (5 tests).

## Phase 1: fetch + isolate (PLANET)

| Used | Why | Rejected alternative | Number that justified it |
|---|---|---|---|
| **LUNA16 rule**: slice ≤ 2.5 mm, nodule = ≥ 3 readers and ≥ 3 mm | a published, comparable definition | every scan / every mark | target ≈ 888 scans / 1,186 nodules → see `results/phase1/gt_report.md` |
| **Centre = mean of the readers' centroids**, diameter = mean of their diameters | each reader outlines independently | one reader's mark | NOT MEASURED |
| **Split by patient, 70/15/15, seed 42** | the same patient in two splits = leakage | split by scan or nodule | gate: 0 shared patients |
| **One coordinate rule**: voxel v ↔ v × spacing mm (first voxel = 0 mm) | volume and nodule centres can't drift apart | `scipy.ndimage.zoom` (stretches the grid slightly) | test: bright voxel (20, 20, 8) at 0.7/0.7/2.5 mm lands at (14, 14, 20) |
| **Clip HU to [−1000, 1000]** before resampling | removes the −2048 padding found in Phase 0 | no clipping | scan #1: 1st percentile −2048 |
| **Trilinear resampling to 1 mm** (`affine_transform`, order 1) | smooth, fast | nearest neighbour (blocky), cubic (overshoots) | NOT MEASURED |
| **Lung mask TRACE**: threshold −320 → drop border air → 2 biggest blobs → 3D closing r = 10 mm → fill holes per slice | classical, explainable, no training | a deep segmentation model | phantom test: wall nodule lost **without** closing, kept **with** it; real gate: ≥ 98% GT inside |
| **Closing via 2 distance transforms** | an exact ball of any radius, fast in 3D | `binary_closing` with a 21³ ball (slow) | NOT MEASURED |
| **Crop to the lung box + 10 mm**, `.npz` compressed | smaller cache, faster loading later | full 1 mm volume | see the cache size in `isolate_report.md` |
| **`spawn` worker processes** | each worker opens its own pylidc database | `fork` (shares one SQLite handle) | prevention, NOT MEASURED |

### First full run of `03_isolate.py` (897 scans, 315 s): 3 gates FAILED, here is why
- **Broken masks: 31 scans < 2 L** (some ~0.01 L), holding **52 GT nodules**, **45** of them outside the mask.
  - Cause: border air was removed **in 3D**. If the lung touches outside air through **one thin path anywhere**, the **whole lung** is deleted.
  - Fix: remove border air **slice by slice** → a leak only costs that slice. Test: `test_one_leak_no_longer_deletes_a_lung`.
- **16 GT nodules missed in normal scans**, all **large (14–38 mm)**.
  - Cause: closing r = 10 mm only fills wall "bites" narrower than about 20 mm.
  - Fix: choose r from a sweep on **TRAIN scans only** (`scripts/03b_mask_sweep.py`, radii 10/15/20).
- **42 scans at 8–10 L, 4 above 10 L**, with lung-like median HU (−858 to −937).
  - Read as **large or hyperinflated lungs**, not leaks → the volume gate is widened from **2–8 L to 2–10 L**.
  - NOT MEASURED: confirm on 3 pictures after the fix.
- **The coordinate gate passed on all 1,192 nodules:** median HU at the GT centre **10 HU**, so the nodule positions are right.

### Lung-mask sweep (`scripts/03b_mask_sweep.py`, TRAIN only: 36 problem + 60 random scans, 151 GT nodules)
| Mask | GT inside | Scans < 2 L | Scans > 10 L | Median lung L (random scans) |
|---|:-:|:-:|:-:|:-:|
| old: 3D border, r = 10 | 68.9% | 23 | 3 | 5.12 |
| per-slice border, r = 10 | 91.4% | 5 | 3 | 5.12 |
| **per-slice border, r = 15 ← chosen** | **98.0%** | 5 | 3 | 5.28 |
| per-slice border, r = 20 | 98.7% | 4 | 3 | 5.53 |
- **Per-slice border removal** alone: broken masks **23 → 5**, nodules inside **68.9% → 91.4%**.
- **r = 15 is the knee:**
  - 10 → 15 gains **+6.6 points**
  - 15 → 20 gains only **+0.7 points** (about **1 nodule of 151**) but adds **+4.7% lung volume** (more chest wall and mediastinum → more false candidates in Phase 2)
- Chosen on **train scans only**, because the test set must not shape any setting.
- **Still open:** 5 scans < 2 L and 3 scans > 10 L in this set → a different failure, listed in the full-run report.

### Full rerun with r = 15 (897 scans, 306 s, 64 workers)
- GT nodules inside the mask: **99.6%** (5 of 1,192 outside: 14.4, 23.6, 24.0, 30.0, 31.3 mm, all wall-attached) → **PASS**.
- Coordinates: median HU at GT centres **11 HU** → **PASS**.
- Sanity ranges: lung 2–10 L in **98.7%**, median HU −950..−700 in **98.9%** (target 99%) → **20 outlier scans**, listed in `results/phase1/qc_outliers.md`.

### The 20 outliers, checked by eye (`scripts/03c_qc_outliers.py`)
- **15 look like real anatomy:**
  - very large, over-inflated lungs: 10.0–11.3 L, HU down to −961
  - small, dense lungs: 1.5–2.8 L, HU −560 to −760
  - their masks stay 19–98 mm away from the image edge
- **5 masks include body or outside air:** LIDC-IDRI-**0619, 0697, 0403, 0523, 0655**.
- **Hypothesis "tissue-like scanner padding" → FALSE.** `scripts/03d_padding_check.py` shows a padding of **−3024** (air-like) or none, and the fix changed **nothing** (6.49 → 6.49 L, etc.).
  - `padding_to_air` stays in as a tested guard. It only acts on non-air padding, which none of these scans have.
- **Decision: accept and name the 5**, not fix them now, because:
  - the mask is only the **search region**
  - a mask that is **too big** adds false candidates that Phase 3 can reject; it does **not** lose nodules
  - the number that matters, **nodules inside = 99.6%**, already passes
  - Phase 2 reports their candidate counts separately; if they dominate the false alarms, we come back to them

## Phase 2: nominate (BLOB)
| Used | Why | Rejected alternative | Number that justified it |
|---|---|---|---|
| **Multi-scale LoG blob detector** | balls respond more than tubes (vessels) or sheets (walls); classical and explainable | threshold + connected blobs (vessel trees merge with nodules) | theory at matched size: ball **0.86 × contrast**, tube **0.73 ×** (`test_ball_beats_vessel`) |
| **σ = 1, 1.5, 2.5, 4, 6, 9, 13 mm** | diameter ≈ 3.5 σ → 3.5–45 mm | one size | Phase 1: largest GT **46.8 mm**, 48 GT > 30 mm |
| **Outside the mask set to −1000 HU** before LoG | the chest wall can't make blobs; wall-attached nodules stand out | LoG on the raw scan | NOT MEASURED |
| **Peak HU > −700** | air can't be a nodule | no check | NOT MEASURED; size table will show ground-glass losses |
| **Bundle = greedy NMS within the blob radius (≥ 3 mm)** | one blob → one candidate | keep all peaks | NOT MEASURED (see median peaks vs kept) |
| **Threshold = fewest candidates with TRAIN recall ≥ 95%** | recall first; nothing later can recover a miss | fixed threshold | see `nominate_report.md` |
| **R0 = rank by blob response only, FROC on VAL** | the "no learning" floor of the staircase | none | see `nominate_report.md` |

### First full run: TRAIN recall stuck at 92.8% → diagnosis (`scripts/04b_miss_diagnosis.py`)
| Cause of the 59 missed TRAIN nodules | Nodules | Median diameter | Median HU at centre |
|---|:-:|:-:|:-:|
| **ranked below the 3,000 cap** | **56** | 7.8 mm | −520 (faint / part-solid) |
| erased by bundling | 2 | 9.3 mm | −28 |
| below the −700 HU rule | 1 | 8.2 mm | −743 |
- Scale-aware bundling recovers **0** within the cap → bundling was **not** the problem. My suspicion was wrong; the measurement decided.
- **The cap was the problem.** Each scan has about **20,860** bundled candidates, and small, faint nodules rank below big bright vessels.
- **Fix:**
  - store up to **30,000** per scan
  - sweep two selection rules **on TRAIN only**:
    - global top-N
    - **per-size top-q**, where each blob size gets its own quota, so small faint blobs don't compete with big bright ones
  - the script **picks automatically** (fewest candidates reaching 95% TRAIN recall)

### Phase 2 result (`scripts/04_nominate.py`, chosen automatically on TRAIN)
- **per-size top-1200**: TRAIN recall **96.0%** at **2,643** candidates/scan.
  - Global top-N needs **5,000**/scan for 95.7% → per-size quotas **halve** the candidates.
- TRAIN recall by size: 3–6 mm **84.1%**, 6–10 **95.6%**, 10–20 **98.8%**, 20–30 **98.4%**, > 30 **100%**.
- R0 (blob strength only, no learning) VAL CPM **0.088**.

## Phase 3: discard (R1–R4), one command `tools/phase3.sh`
| Used | Why | Rejected alternative | Number that justified it |
|---|---|---|---|
| **TRAIN sample** per scan: positives + GT centre ×5 (jitter ±2 mm) + 150 random + 50 hardest negatives | 2,643 candidates × 628 scans = 1.66 M is too many; balance matters more than volume | all candidates | NOT MEASURED |
| **Balanced batches** (25% positives) | positives are < 5% of candidates | plain shuffling | prevention |
| **Small 3D CNN** (4 blocks, ~1.5 M weights) on 32 mm cubes | fast on H200; cubes are small | 3D ResNet-18 (33 M weights in v2) | NOT MEASURED |
| **R3 = focal loss + hard negatives + rot90 + 8-flip TTA** | each "advanced" trick in one rung vs R2 | separate rungs per trick | 2-day budget |
| **R4 fusion cross-fitted on VAL** (5 folds by patient) | the fusion never scores a candidate it was fitted on | fit and score on the same val | prevention |
| **5 seeds** for R2 and R3 | one run moves by chance (v2 lesson) | 1 seed | v2: ±0.01 AUC between runs |
