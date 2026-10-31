# Phase 5: ship (S)

> **Skeleton written by Claude** (Sat 31 Oct 2026, evening). The parts marked **✍ you** are yours: write them **from memory, code closed**, then I compare them with the answer key at the end of `docs/DECISIONS.md`.

Prediction (written at start): *✍ you, one line, before you run `predict.py` on a real scan: "On a TEST scan, `predict.py` at 1 FP/scan will show about … findings, and … of them will be real."*

What it does: turns the measured model into something **others can run and check**: one command (`predict.py`), a demo (Gradio), honest docs (README results, reality check), slide figures, and a **GitHub Release v1.0** with the weights.

Flow:
**scan in** → **Isolate** (HU, 1 mm, lung mask) because every scan needs the same ruler → **Nominate** (blobs) because the CNN can't look at 100 million voxels → **Discard** (5 CNNs on cubes) because shape in 3D separates nodules from vessels → **threshold from VAL** because TEST was opened once → **merge** because one nodule gets several candidates → **findings + picture**

## Decisions
| Used | Why | Rejected alternative | Number that justified it |
|---|---|---|---|
| **`ship/pipeline.py` reuses the Phase 1–3 functions** (`lung_mask`, `log_peaks`, `bundle`, `crop_cubes`, `load_ensemble`) | the TEST numbers came from this code; a copy would drift | a new "fast" inference path | test: the planted phantom nodule is a **rank-1 candidate at distance 0 mm** |
| **Threshold per false-alarm rate, picked on VAL** (`scripts/12_operating_points.py`) | TEST was opened once; choosing on TEST = a second look | thresholds from TEST | **NOT MEASURED until you run `tools/phase5.sh` (Mac)** |
| **Merge detections within max(3 mm, √3·σ)** | one nodule → many candidates; one row per nodule reads better | show every candidate | **NOT MEASURED on TEST**: not part of the 0.823 |
| **pydicom is optional**; slices sorted by position along the slice normal; spacing = median gap | files in a folder have no order | sort by filename | test: 12 slices written in **random order** come back identical (16×16×12, spacing 0.7/0.8/2.5) |
| **Everything after `git pull` runs on the Mac**: `tools/phase5.sh` rsyncs 1 small parquet + 1 scan cache (~25 MB) and runs on CPU; Gradio on 127.0.0.1 | the college GPU server is shared: no jobs, no tmux, no public port | run `tools/phase5.sh` on the server / `share=True` | server touched twice, read-only (rsync `--bwlimit`) |
| **Occlusion batch 16 on CPU, 512 on GPU** | 344 cubes × 3 channels at batch 512 **killed the process** (exit 137) on CPU | fewer blocks | crash at 512, works at 16 |
| **FROC figure from the 7 measured points** | the saved CSV is a **500-point sample out to 2,650 FP/scan** (2nd point already at 3.4 FP/scan) | plot the CSV | `results/phase4/froc_test_R5.csv` |
| **Weights in the GitHub Release, not in git** | `.gitignore` blocks `*.pth`; 5 × 14.09 MB | commit them / Git LFS | 5 × 14,089,405 bytes |
| **`requirements.txt` with ranges** | one file for laptop and server | pin the server freeze | passes with torch 2.15 / numpy 2.5.3 (cloud) and is meant for 2.14.1 / 2.5.4 (server) |

## Worked example (real numbers)
- **Picking a threshold** (the test in `tests/test_ship.py`): 4 scans, false candidates scored 0.9, 0.8, 0.7, 0.6, 0.5, 0.4; two real nodules scored 0.95 and 0.65.
  - **1 FP/scan** × 4 scans = **4** false alarms allowed → threshold = the **5th** strongest false candidate = **0.5** → survivors above it: 0.9, 0.8, 0.7, 0.6 = **4 false alarms = 1.0 per scan** ✓ and **both** nodules (0.95, 0.65) stay → sensitivity **1.0**.
  - **0.5 FP/scan** → **2** allowed → threshold **0.7** → only nodule 0.95 stays → sensitivity **0.5**.
- **Phantom run** (random weights, so only the plumbing is checked): 1,308 candidates scored in **56 s** with 1 model on a cloud CPU → **≈ 43 ms per cube per model** → a real scan (≈ 2,643 cubes × 5 models) ≈ **10 min** on a similar CPU (extrapolation).

## My image + acronym
*(Claude's suggestion, invent your own)*
- **Image: opening night of a restaurant.** The kitchen (pipeline) is done; now the **menu** (README), the **health inspector's report** (reality check), a **take-away box** (release with weights), and a **table where guests can try a dish** (Gradio).
- **SHIP** = **S**cript (`predict.py`) → **H**ands-on demo (Gradio) → **I**ntegrity doc (reality check) → **P**ictures + **P**ublish (figures, Release v1.0)

## What breaks / limits
- `predict.py` was run end to end on a **synthetic phantom** (random weights for the full CLI, **real R5 weights** for the scoring check: planted ball = rank 1 of 258, p = 1.000); a **real scan** is the first true check (`bash tools/phase5.sh` runs scan LIDC-IDRI-0405 on the Mac CPU).
- **Threshold + merge are not in the TEST numbers.**
- **Memory:** the full-chest lung mask needs about **8–16 GB** RAM; the 8 GB Mac may struggle (**NOT MEASURED**).
- **No `LICENSE` file yet.**
- Details: `docs/REALITY_CHECK.md`.

## What I wrote by hand
- ✍ you (nothing yet).

## Prediction vs result
- ✍ you, after `tools/phase5.sh`: one line on what surprised you.

## Still unclear
- ✍ your questions here.

## Pipeline (whole project, as of commit `<hash>`)
✍ you, **from memory, code closed**: every stage from input to output of the WHOLE project, one line per stage, naming the file and function. Then compare with the answer key at the end of `docs/DECISIONS.md`.
