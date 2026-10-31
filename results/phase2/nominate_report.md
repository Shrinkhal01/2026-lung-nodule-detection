# Phase 2 · nominate report (`scripts/04_nominate.py`)

- scans: **897**, runtime **253 s**
- stored candidates per scan (storage cap 30000): median **20465**
- best TRAIN recall in the sweep: **99.6%** (global_top = 20000)
- **chosen: per_size_top = 1200** (fewest candidates with TRAIN recall ≥ 95%)
  - TRAIN recall **96.0%** at **2643** candidates/scan
  - VAL recall **93.9%** at **2633** candidates/scan (check, not tuned)
- candidates per scan after selection: median **2660**, 90th **2933**
- broken-mask scans: 0403=2118, 0523=2290, 0619=3342, 0697=3301, 0655=2846

## TRAIN sweep

| rule | value | train_recall | train_cands_per_scan |
|---|---|---|---|
| global_top | 1000 | 0.811 | 1000 |
| global_top | 2000 | 0.8915 | 2000 |
| global_top | 3000 | 0.928 | 3000 |
| global_top | 5000 | 0.9573 | 5000 |
| global_top | 8000 | 0.9744 | 8000 |
| global_top | 12000 | 0.9866 | 11990 |
| global_top | 20000 | 0.9963 | 18542 |
| global_top | 30000 | 0.9963 | 20697 |
| per_size_top | 100 | 0.6524 | 445 |
| per_size_top | 200 | 0.7415 | 752 |
| per_size_top | 300 | 0.8 | 1000 |
| per_size_top | 500 | 0.8768 | 1408 |
| per_size_top | 800 | 0.9341 | 2000 |
| per_size_top | 1200 | 0.9598 | 2643 |
| per_size_top | 2000 | 0.972 | 3507 |
| per_size_top | 3000 | 0.9793 | 4507 |

## TRAIN recall by nodule size (chosen selection)

| size | nodules | recall |
|---|---|---|
| 3-6 | 69 | 0.841 |
| 6-10 | 411 | 0.956 |
| 10-20 | 240 | 0.988 |
| 20-30 | 62 | 0.984 |
| >30 | 38 | 1.0 |

## R0 baseline (rank by blob strength, no learning), VAL

| FP/scan | 0.125 | 0.25 | 0.5 | 1 | 2 | 4 | 8 | **CPM** |
|---|---|---|---|---|---|---|---|---|
| sensitivity | 0.048 | 0.061 | 0.075 | 0.109 | 0.109 | 0.109 | 0.109 | **0.088** |

TEST split: not used, not reported (opened once in Phase 4).
