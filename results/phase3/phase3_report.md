# Phase 3 · staircase on VAL (`scripts/06c_summarize.py`)

- val scans **134**, GT nodules **147**, candidates **352856** (ceiling = candidate recall; every sensitivity is capped by it)

| rung | CPM | seeds (mean ± SD) | sens@0.5 | sens@1 | sens@2 | sens@4 | what |
|---|---|---|---|---|---|---|---|
| R0 | 0.088 |  | 0.075 | 0.109 | 0.109 | 0.109 | blob strength only, no learning |
| R1 | 0.275 |  | 0.197 | 0.238 | 0.313 | 0.381 | XGBoost on hand-made features |
| R2 | 0.778 | 0.760 ± 0.013 (n=5) | 0.741 | 0.844 | 0.857 | 0.878 | 3D CNN, BCE, random negatives, 5-seed ensemble |
| R3 | 0.717 | 0.711 ± 0.010 (n=5) | 0.667 | 0.769 | 0.844 | 0.884 | 3D CNN + focal loss + hard negatives + TTA, 5-seed ensemble |
| R4 | 0.78 |  | 0.735 | 0.823 | 0.857 | 0.891 | fusion: logistic regression on [R2 ensemble, R1], cross-fitted by patient |

- Each rung must beat the one below it on VAL to earn its place. TEST is opened once, in Phase 4.
