# Phase 3 · staircase on VAL (`scripts/06c_summarize.py` --tag _smoke)

- val scans **10**, GT nodules **12**, candidates **27255** (ceiling = candidate recall; every sensitivity is capped by it)

| rung | CPM | seeds (mean ± SD) | sens@0.5 | sens@1 | sens@2 | sens@4 | what |
|---|---|---|---|---|---|---|---|
| R0 | 0.0 |  | 0.0 | 0.0 | 0.0 | 0.0 | blob strength only, no learning |
| R1 | 0.06 |  | 0.0 | 0.0 | 0.0 | 0.167 | XGBoost on hand-made features |
| R2 | 0.012 | 0.012 ± 0.000 (n=1) | 0.0 | 0.0 | 0.0 | 0.0 | 3D CNN, BCE, random negatives, 1-seed ensemble |
| R3 | 0.0 | 0.000 ± 0.000 (n=1) | 0.0 | 0.0 | 0.0 | 0.0 | 3D CNN + focal loss + hard negatives + TTA, 1-seed ensemble |
| R4 | 0.036 |  | 0.0 | 0.0 | 0.0 | 0.0 | fusion: logistic regression on [R3 ensemble, R1], cross-fitted by patient |

- Each rung must beat the one below it on VAL to earn its place. TEST is opened once, in Phase 4.
