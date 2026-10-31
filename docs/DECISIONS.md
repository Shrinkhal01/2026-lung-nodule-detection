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
