# Phase 1 · lung-mask sweep (TRAIN scans only, `scripts/03b_mask_sweep.py`)

- sweep set: **36** problem scans + **60** random train scans; **151** GT nodules; runtime **0 s**

| radius | GT inside % | scans < 2 L | scans > 10 L | median L (random) |
|---|---|---|---|---|
| old (3D border, r=10) | 68.9 | 23 | 3 | 5.12 |
| new (per-slice), r=10 | 91.4 | 5 | 3 | 5.12 |
| new (per-slice), r=15 | 98.0 | 5 | 3 | 5.28 |
| new (per-slice), r=20 | 98.7 | 4 | 3 | 5.53 |
