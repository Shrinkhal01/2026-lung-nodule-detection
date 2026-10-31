# Phase 3b · HUSH staircase on VAL (`scripts/06d_summarize_hush.py`)

- val scans **134**, GT nodules **147**, candidates **352856**

| rung | CPM (ensemble) | seeds mean ± SD | sens@1 | sens@4 | verdict |
|---|---|---|---|---|---|
| R2 baseline 3D CNN (Phase 3) | 0.778 | 0.760 ± 0.013 (n=5) | 0.844 | 0.878 | baseline |
| R5 = R2 + H (Hessian channels) | 0.779 | 0.768 ± 0.012 (n=5) | 0.823 | 0.884 | ❌ not above seed noise (Δ +0.008, needs > 0.013) |
| R6 = R5 + U (masked-cube pretraining) | 0.776 | 0.762 ± 0.008 (n=5) | 0.837 | 0.878 | ❌ not above seed noise (Δ -0.006, needs > 0.012) |
| R7 = R6 + S (size head) | 0.772 | 0.751 ± 0.018 (n=5) | 0.844 | 0.878 | ❌ not above seed noise (Δ -0.011, needs > 0.018) |

- **HUSH = R5 ensemble: VAL CPM 0.779** (Δ vs R2 ensemble +0.001)

## Explanation by ablation: blank a physics channel at test time, measure the CPM drop

| rung (1 seed) | all 3 channels | no blobness | no vesselness | HU only |
|---|---|---|---|---|
| R5 | 0.776 | 0.663 | 0.695 | 0.397 |
| R6 | 0.748 | 0.577 | 0.729 | 0.199 |
| R7 | 0.774 | 0.528 | 0.727 | 0.157 |

- A big drop = the network **relies** on that channel. No drop = it ignores it.

- VAL was used for choices → slightly optimistic. TEST is opened once, in Phase 4.
