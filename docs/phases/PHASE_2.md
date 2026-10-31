# Phase 2: nominate (BLOB)

> **Written by Claude at your request** (Sat 31 Oct 2026, ~13:55), for you to read and understand.

Prediction (written at start): *(not given)*

What it does: sweeps every lung with a **blob detector** and lists **candidate spots** that might be nodules.
- **96% of TRAIN nodules** are on the list, at about **2,640 candidates per scan**.
- The list is the input for Phase 3, which throws out the false alarms.

Flow:
**1 mm lung `.npz`** → **Blur** at 7 sizes (because nodules are 3.5–47 mm) → **Laplacian** = curvature (because balls curve in 3 directions, vessels in 2) → **Only peaks** inside the lung, HU > −700 (because air isn't a nodule) → **Bundle** duplicates (because one blob shows up at several sizes) → **per-size top-1200** (because small faint blobs must not compete with big bright vessels) → **candidates.parquet**

## Decisions
| Used | Why | Rejected alternative | Number that justified it |
|---|---|---|---|
| Multi-scale **LoG** blob detector | balls respond more than tubes (vessels) | threshold + connected blobs | theory: ball **0.86 ×** contrast vs tube **0.73 ×** (`test_ball_beats_vessel`) |
| σ = 1 / 1.5 / 2.5 / 4 / 6 / 9 / 13 mm | diameter ≈ 3.5 σ → 3.5–45 mm | one size | largest GT nodule **46.8 mm** (Phase 1) |
| **Storage cap 30,000** (was 3,000) | the 3,000 cap hid faint nodules | keep 3,000 | diagnosis: **56 of 59** missed TRAIN nodules were below the cap |
| Old bundling kept | the scale-aware version recovered nothing | scale-aware bundling | recovered **0** of 59 (my hypothesis was wrong; the measurement decided) |
| **Per-size top-1200** (chosen automatically on TRAIN) | each blob size gets its own quota | global top-N | **96.0%** recall at **2,643**/scan vs global: **5,000**/scan for 95.7% |
| Test split untouched | it is opened once, in Phase 4 | — | — |

## Worked example (real numbers)
- **Blob size:** the median nodule is **9.1 mm** → radius 4.55 mm → best σ = 4.55 ÷ √3 ≈ **2.6 mm** → caught by the 2.5 mm level.
- **Why the cap failed:** a scan has about **45,000** raw peaks → about **20,900** after bundling.
  - The missed nodules were small (median **7.8 mm**) and faint (median **−520 HU**), so they ranked below the 3,000 strongest, mostly vessels.
- **Why per-size quotas win:** 7 sizes × up to 1,200 each, with overlaps removed → **2,643** per scan.
  - Small faint blobs only compete with **other small blobs**.
- **R0 (no learning):** rank by blob strength → VAL **CPM 0.088** (sensitivity 0.109 at 8 FP/scan). That's the floor Phase 3 must beat.

## My image + acronym
*(Claude's suggestion)*
- **Image: minesweeping, step 1.** The **metal detector** beeps at everything metal. To hear the quiet beeps of small coins, you give each **coin size its own volume knob** (per-size quotas), instead of letting the big pipes drown them out.
- **BLOB** = **B**lur → **L**aplacian → **O**nly peaks (in the lung, not air) → **B**undle duplicates

## What breaks / limits
- **Small nodules (3–6 mm):** only **84.1%** caught (69 TRAIN nodules) → the hardest group, and a recall ceiling for every later stage.
- **2,643 candidates per scan is a lot** → Phase 3 has to reject about 99.9% of them.
- Overall recall ceiling for Phase 3 ≈ **96% (candidates) × 99.6% (mask)**.
- **Broken-mask scans** (0403, 0523, 0619, 0697, 0655) got **2,118–3,342** candidates, close to the median of 2,660 → they don't dominate the false alarms.
- **R0 is very weak** (CPM 0.088) → blob strength alone can't tell nodules from vessels. That's the job of learning.

## What I wrote by hand
- Nothing in Phase 2 (your choice: reading over writing).

## Prediction vs result
- No prediction given.
- **Surprise:** my first suspect (bundling) was **wrong**. The diagnosis script pointed to the **cap** instead, which is why we measure before fixing.

## Still unclear
- *(your questions here)*

## Pipeline (whole project, as of the end of Phase 2)
1–8. As in `PHASE_1.md` (env check, config, download, pylidc, gate, ground truth + splits, isolate + lung mask, tuning/QC scripts).
9. **Nominate:** `scripts/04_nominate.py:_scan`
   - `nominate/blobs.py:log_peaks` (7 sizes, peaks in lung, HU > −700)
   - → `bundle` (strongest first)
   - → `rate/matching.py:match` (hit / ignore)
   - → **self-tuning selection on TRAIN** (global vs per-size)
   - → `$SCRATCH/candidates/candidates.parquet`
   - R0 via `rate/froc.py:froc` on VAL
10. **Diagnosis:** `scripts/04b_miss_diagnosis.py` (why each missed TRAIN nodule was missed)
11. **Tests:** 23 on the Mac (22 run, 1 needs torch → it runs on the server)
