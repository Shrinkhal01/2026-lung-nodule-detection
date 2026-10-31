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
