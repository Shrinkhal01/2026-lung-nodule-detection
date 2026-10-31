# Phase 1: fetch + isolate (PLANET)

> **Written by Claude at your request** (Sat 31 Oct 2026, ~13:05), for you to read and understand.
> Add your own questions under "Still unclear".

Prediction (written at start): *(not given, by your choice)*

What it does: turns **1,018 raw CT scans + radiologist marks** into **897 clean, equal-ruler lung volumes** with a **ground-truth table**:
- **1,192 nodules** (≥ 3 of 4 readers, ≥ 3 mm) + **1,099 ignore zones**
- **patient-level splits**: train / val / test
- **one 1 mm `.npz` per scan** (HU + lung mask), 23 MB each

Flow:
**pylidc DB** → **Pick** scans ≤ 2.5 mm (because thick slices blur small nodules) → **Label** = cluster the readers' marks (because 4 readers outline one nodule 4 times) → **Assign** splits by patient (because a shared patient leaks) → **Normalise** clip HU to [−1000, 1000] (because of the −2048/−3024 padding) → **Equalise** to 1 mm (because a 3D filter needs the same mm per voxel) → **Trace** the lungs (because nodules live inside them) → **crop + save `.npz`**

## Decisions
| Used | Why | Rejected alternative | Number that justified it |
|---|---|---|---|
| LUNA16 rule (≤ 2.5 mm, ≥ 3 readers, ≥ 3 mm) | published, comparable | all scans / all marks | **897 scans (+1.0%)**, **1,192 nodules (+0.5%)** vs LUNA16's 888 / 1,186 |
| Split by patient 70/15/15, seed 42 | no leakage | split by nodule | **0** shared patients; train 627 / val 134 / test 135 patients |
| One coordinate rule (voxel v ↔ v × spacing mm) | volume and nodules can't drift apart | `ndimage.zoom` | median HU at GT centres **11 HU** (soft tissue → positions are right) |
| Clip HU [−1000, 1000] | removes the padding | no clip | scan #1 1st percentile **−2048** |
| Border air removed **per slice** | one leak no longer deletes a whole lung | 3D removal | broken masks **23 → 5** (sweep set), nodules inside **68.9% → 91.4%** |
| Closing radius **15 mm** | the knee: big wall nodules come inside | 10 / 20 mm | 10 → **91.4%**, 15 → **98.0%**, 20 → **98.7%** but **+4.7%** lung volume (train-only sweep) |
| Sanity gate 2–**10** L (was 8) | 42 scans at 8–10 L had lung-like HU | keep 8 L | HU −858 to −937 in those scans |
| Accept 5 named broken masks | too-big masks add FPs, they don't lose nodules | more fixing now | nodules inside **99.6%**; padding hypothesis tested → **false** |
| 64 `spawn` workers | 144 cores, 1 TB RAM | 1 process | **306 s** for 897 scans (vs ~5 h single-core) |

## Worked example (real numbers)
- **Resampling scan #1:** 512 × 512 × 133 voxels at 0.703 × 0.703 × 2.5 mm
  → last voxel at 511 × 0.703 = **359.2 mm** and 132 × 2.5 = **330 mm**
  → new grid **360 × 360 × 331** voxels at 1 mm.
- **The coordinate rule in one line:** a bright voxel at (20, 20, 8) with spacing (0.7, 0.7, 2.5) sits at (14, 14, 20) mm → index **(14, 14, 20)** in the 1 mm grid (unit test).
- **Lung volume:** 1 voxel = 1 mm³ = 0.001 mL → **5,230,000 mask voxels = 5.23 L** (the median scan).
- **Nodule sizes:** median **9.1 mm**, 90th percentile **21.8 mm**, max **46.8 mm**; **48** nodules > 30 mm → the Phase 2 size cap must be **≥ 50 mm**.
- **Splits:** train 820 / val 147 / test **225** nodules → test has more nodules per patient (1.67 vs 1.10), by chance of the random split.

## My image + acronym
*(Claude's suggestion; replace with your own)*
- **Image: kitchen prep.**
  - **Pick** the ingredients
  - **Label** them
  - **divide them into portions** for practice / tasting / the judge
  - **Normalise** = wash
  - **Equalise** = cut to the same size
  - **Trace** = peel off what you don't need
- **PLANET** = **P**ick → **L**abel → **A**ssign → **N**ormalise → **E**qualise → **T**race
- **TRACE** (the lung mask) = **T**hreshold −320 → **R**emove border air (per slice) → **A**ccept the 2 biggest blobs → **C**lose (15 mm ball) → **E**at holes (per slice)

## What breaks / limits
- **5 broken masks** (0619, 0697, 0403, 0523, 0655) include body or outside air. The cause is **unknown** (padding ruled out) → more false candidates in those scans.
- **5 GT nodules (0.4%) lie outside the mask** (14–31 mm, wall-attached) → **recall ceiling 99.6%** for every later stage.
- **13 clusters merged more than 4 marks** → two nearby nodules averaged into one centre.
- **Ignore zones only cover 1–2-reader marks.** As far as I know, pylidc has no < 3 mm or "non-nodule" marks, which LUNA16 also ignored → our FP count may be **slightly pessimistic**.
- **The test split holds 225 nodules vs 147 in val** (random patient split) → reported, not changed.
- The `padding_to_air` guard was added **after** the cache was built. It didn't change the 5 checked scans; it is not verified on all 897.

## What I wrote by hand
- Nothing in Phase 1 (your choice: reading over writing).

## Prediction vs result
- No prediction given. **Surprise:** the gates caught **3 real bugs** in one phase:
  - 3D border removal deleting whole lungs
  - too small a closing radius
  - a wrong guess (padding) disproved by measuring before rerunning
  - plus 2 bugs in **Claude's own tests**, caught because nothing is pushed while a test fails

## Still unclear
- *(your questions here)*

## Pipeline (whole project, as of the Phase 1 merge)
1. **Env check:** `scripts/00_check_env.py:main`
2. **Settings:** `configs/default.yaml` → `src/lungdet/common/config.py:load_config`
3. **Download:** inline idc-index command → `$SCRATCH/lidc_raw/`; check/resume with `scripts/01_fetch.py` → `fetch/inventory.py:scan_series_folders`, `series_status`
4. **pylidc on new Python:** `fetch/pylidc_compat.py:apply_shims`, `import_pylidc`, `write_pylidcrc`
5. **Gate on one scan:** `scripts/00_gate_one_scan.py:main`
6. **P-L-A:** `scripts/02_ground_truth.py:main`
   - scan table (eligible = downloaded and ≤ 2.5 mm)
   - `_cluster` → `fetch/ground_truth.py:summarise_cluster`
   - `assign_splits`
   - → `scans.csv`, `nodules.csv`, `gt_report.md`
7. **N-E-T:** `scripts/03_isolate.py:_process`
   - `isolate/padding.py:padding_to_air` → clip
   - → `isolate/geometry.py:resample`
   - → `isolate/lungmask.py:lung_mask` (`remove_border_air` per slice, keep 2 blobs, distance-transform closing, fill holes)
   - → `crop_box` → `.npz` in `$SCRATCH/cache_lungdet/`
   - nodule checks → `isolate_report.md`, `qc_montage`
8. **Tuning / QC:**
   - `scripts/03b_mask_sweep.py` (radius, train only)
   - `03c_qc_outliers.py` (outlier numbers + pictures)
   - `03d_padding_check.py` (hypothesis test)
9. **Workflow:**
   - `tools/ship.sh` (Mac: test → commit → push)
   - `tools/run.sh` (server: pull → run)
   - `tools/pull_results.sh` (server results → Mac)
   - server results live in `$SCRATCH/lungdet_results/` (outside git)
10. **Tests (Mac, CPU):** 15, in `tests/` (geometry, lung-mask phantom, leak, padding, ground truth, splits, config, fetch)
