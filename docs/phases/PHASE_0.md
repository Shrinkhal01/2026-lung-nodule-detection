# Phase 0: setup

> **Written by Claude at your request** (Sat 31 Oct 2026, 10:20), for you to read and understand.
> The prediction below is your own. Add anything you find unclear under "Still unclear".

Prediction (written at start, Sat 31 Oct 2026, 08:29):
1. Full repository structure
2. Full libraries, packages installed everywhere so that next we code and run the code
3. Therefore phase 0 work is to produce everything that would be required later except the files, so that in the upcoming phases we just run the code/files

What it does: turns an **empty Mac folder** and a **full, shared GPU server** into a working lab:
- a repo on GitHub
- one Python env on the server with every package
- **all of LIDC-IDRI downloading to `/scratch`**
- **one real scan + its radiologist marks loading correctly** (the gate)

Flow:
**Mac** (write code + tests) → **GitHub** (because the server only pulls, it never commits) → **server `/scratch`** (because `/home` is 100% full) → **idc-index download** (because it is a public bucket: no login, parallel, resumable) → **pylidc** (because it holds the radiologists' marks) → **gate: 1 scan loads** (because nothing later can work if this doesn't)

## Decisions
| Used | Why | Rejected alternative | Number that justified it |
|---|---|---|---|
| **`/scratch/mtech_shrinkhal`** for data, env, caches | `/home` is shared and full | `/home`; RAM-only streaming | `/home` **0.4 GiB** free vs `/scratch` **2,683 GiB** |
| **idc-index** (public IDC bucket) | no login, parallel (s5cmd), resumable | NBIA Data Retriever (GUI + login) | speed test **4.94 MB/s**; full run **128 GB in 1 h 59 min = 18.0 MB/s average** (peak ~25.9) |
| **Reuse the old venv** (copied to scratch, paths fixed with `sed`) | it already had **torch 2.14.1 + CUDA 13** | fresh torch install | PyPI here: **~0.14–0.19 MB/s** → torch (~2.5 GB) would take hours |
| **`pip -c keep.txt`** | new packages must not silently replace torch/numpy | plain `pip install` | prevention, NOT MEASURED |
| **`setuptools<81`** | pylidc imports `pkg_resources` (removed in newer setuptools) | patching pylidc | import error: `No module named 'pkg_resources'` |
| **pylidc shims** (`np.int`, `SafeConfigParser`) | 2016-era pylidc on NumPy 2 + Python 3.13 | old NumPy (would break torch) | pylidc GitHub issue #69 |
| **`xgboost-cpu`** | ~20 numbers per candidate → CPU is enough | GPU xgboost (+ NCCL) | **5.8 MB** vs ~500 MB |
| **Public repo** + https clone | the server needs no GitHub key | private repo + deploy key | clone asked for a username while private |
| **Mac: test kit only** | the Mac never touches CT data | full server stack on the Mac | Mac: 8 GB RAM, 22 GiB free |

## Worked example (real numbers from scan #1, LIDC-IDRI-0001)
- Volume **512 × 512 × 133** voxels; pixel **0.703 mm**, slice **2.5 mm** → the scan covers 512 × 0.703 ≈ **360 mm** across and 133 × 2.5 ≈ **333 mm** deep.
- Loaded in **0.8 s**.
- HU percentiles 1 / 50 / 99 = **−2048 / −932 / 420**:
  - **−2048** = the scanner's **padding** outside the round field of view (not air)
  - **−932** ≈ air → most of a chest CT is air (outside the body + lungs)
  - **420** ≈ bone
- **4 reader marks → 1 nodule** (all 4 readers agree), mean diameter **32.8 mm** → counts as ground truth (≥ 3 readers, ≥ 3 mm).
- Download time: 128 GB × 1024 ÷ 18.0 MB/s ≈ **7,280 s ≈ 2 h** → matches the measured **1 h 59 min** (08:45 → 10:44).

## My image + acronym
*(Claude's suggestion, replace with your own if you like)*
- **Image:** **setting up a lab before the experiment.** You choose the room (scratch), stock the shelves (packages), order the samples (download), fix an old instrument (pylidc), and run one test sample (the gate).
- **Acronym: SETUP**
  - **S**pace → `/scratch`, because `/home` is full
  - **E**nvironment → the venv + 18 packages, with torch protected
  - **T**ransfer → GitHub for code, idc-index for data
  - **U**nlock pylidc → the shims + `setuptools<81`
  - **P**rove it → the gate on scan #1

## What breaks / limits
- **The full download was started by an inline command**, not a repo script → `scripts/01_fetch.py --download` can redo or resume it.
- **The server env isn't pinned in a file yet** → a `pip freeze` snapshot is needed so it can be rebuilt.
- **The gate's air check was too weak:** it passed because of the **−2048 padding**. Phase 1 checks air inside the body.
- **The 30 mm candidate cap would miss scan #1's 32.8 mm nodule** → this is decided in Phase 2 from all nodule diameters.
- **pylidc depends on shims** and on an old setuptools. A future setuptools upgrade would break it again.
- **`/scratch` backup policy is unknown** → code is safe on GitHub; data can be re-downloaded in ~2 h.

## What I wrote by hand
- Nothing in Phase 0.
- Phase 1 candidate for a hand-written core: the **mm ↔ voxel coordinate conversion**, because a mistake there silently moves every nodule.

## Prediction vs result
1. "Full repository structure" → ✅ **yes**: every folder, each with a README.
2. "Packages installed everywhere" → ✅ **server: all 18**. ⚠ **Mac: test kit only, by design**, because the Mac never runs real data.
3. "Everything except the files" → ❌ **different**: Phase 0 also needed **code files** (env check, config, fetch check, pylidc shims, gate) **and** started the **download**.
- **Surprise:** the server's **`/home` was full**, and **PyPI was ~100× slower** than the data bucket (0.19 vs 18.0 MB/s average).

## Still unclear
- 

## Pipeline (whole project, as of the Phase 0 merge)
1. **Environment check**: `scripts/00_check_env.py:main` → `check_packages`, `check_gpu`, `check_disk`.
2. **Settings**: `configs/default.yaml` → `src/lungdet/common/config.py:load_config` (expands `$SCRATCH`).
3. **Download**: inline idc-index command in tmux → `$SCRATCH/lidc_raw/lidc_idri/<PatientID>/<StudyUID>/CT_<SeriesUID>/*.dcm`.
4. **Download check / resume**: `scripts/01_fetch.py:main` → `src/lungdet/fetch/inventory.py:scan_series_folders`, `series_status`.
5. **pylidc on today's Python**: `src/lungdet/fetch/pylidc_compat.py:apply_shims` → `import_pylidc`; `write_pylidcrc` → `~/.pylidcrc`.
6. **Gate**: `scripts/00_gate_one_scan.py:main` → `pl.query(pl.Scan)` → `scan.to_volume` → `scan.cluster_annotations` → GT rule (≥ 3 readers, ≥ 3 mm).
7. **Tests (Mac)**: `tests/test_fetch_inventory.py`, `tests/test_config_and_compat.py` (5 tests).