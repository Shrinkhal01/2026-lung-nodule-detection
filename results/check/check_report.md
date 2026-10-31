# Length + fit check (`scripts/09_check_summary.py`, seeds 42–44, VAL)

| model | epochs | VAL CPM (3 seeds) | ensemble CPM | TRAIN AUC | VAL AUC | gap |
|---|---|---|---|---|---|---|
| R2 | 10 | 0.760 ± 0.012 | 0.771 | 0.9999 | 0.9889 | 0.011 |
| R5 | 10 | 0.768 ± 0.014 | 0.778 | 0.9999 | 0.9885 | 0.0114 |
| R2 long | 30 | 0.769 ± 0.002 | 0.783 | 1.0 | 0.9856 | 0.0144 |
| R5 long | 30 | 0.779 ± 0.008 | 0.78 | 1.0 | 0.9831 | 0.0169 |

- **R2: 30 vs 10 epochs → Δ +0.009** (needs > 0.012) → ❌ not above seed noise
- **R5: 30 vs 10 epochs → Δ +0.011** (needs > 0.014) → ❌ not above seed noise

- **Fit:** TRAIN AUC ≫ VAL AUC (gap > ~0.03) → overfitting; both low → underfitting; small gap → fits well.

## Learning curve of the long runs (VAL CPM, mean of seeds)

| epoch | r2_long | r5_long |
|---|---|---|
| 10 | 0.737 | 0.737 |
| 20 | 0.745 | 0.73 |
| 30 | 0.769 | 0.779 |

## Final model for Phase 4: **R5**, seeds [42, 43, 44, 45, 46] (default: HUSH R5 (10 epochs))
