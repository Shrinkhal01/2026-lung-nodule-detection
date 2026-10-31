# Phase 2 · nominate report (`scripts/04_nominate.py` --limit 5)

- scans: **5**, runtime **21 s**, median **15 s**/scan per worker
- stored peaks per scan before the threshold: median **3000** (raw scale-space peaks: median 40160)
- best possible TRAIN recall (no threshold): **100.0%**
- **chosen threshold 529.5** (fewest candidates with TRAIN recall ≥ 95%):
  - TRAIN recall **100.0%** at **233** candidates/scan
  - VAL recall **0.0%** at **nan** candidates/scan (check, not tuned)
- candidates per scan after the threshold: median **190**, 90th **281**
- broken-mask scans: none in this run

## TRAIN recall by nodule size

| size | nodules | recall |
|---|---|---|
| 3-6 | 0 | nan |
| 6-10 | 1 | 1.0 |
| 10-20 | 3 | 1.0 |
| 20-30 | 1 | 1.0 |
| >30 | 2 | 1.0 |

## R0 baseline (rank by blob strength, no learning), VAL

| FP/scan | 0.125 | 0.25 | 0.5 | 1 | 2 | 4 | 8 | **CPM** |
|---|---|---|---|---|---|---|---|---|
| sensitivity | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** |

TEST split: not used, not reported (opened once in Phase 4).
