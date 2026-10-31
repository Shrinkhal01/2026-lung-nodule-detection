# Phase 1 · isolate report (`scripts/03_isolate.py`)

- scans processed: **897**, failed: **0**, runtime **306 s**
- lung volume (L): median **5.23**, min **1.51**, max **11.33**
- lung median HU: median **-854**
- cache: **21.4 GiB** total, **23.4 MB** per scan (median)
- GT nodules outside the mask: **5** (their diameters: [14.4, 23.6, 24.0, 30.0, 31.3])

## Gates

| Gate | Result | Value |
|---|:-:|---|
| 0 scans failed | PASS | 0 of 897 |
| lung volume 2–10 L in ≥ 99% of scans | FAIL | 98.7% |
| lung median HU in [−950, −700] in ≥ 99% of scans | FAIL | 98.9% |
| GT nodule centres inside the lung mask ≥ 98% | PASS | 99.6% of 1192 |
| median HU at GT centres > −400 (coordinates line up) | PASS | 11 HU |

## Scans to look at

| scan_id | patient_id | lung_L | lung_median_hu | error |
|---|---|---|---|---|
| 84 | LIDC-IDRI-0074 | 1.961 | -730.0 |  |
| 149 | LIDC-IDRI-0146 | 10.119 | -876.0 |  |
| 407 | LIDC-IDRI-0403 | 1.722 | -743.0 |  |
| 333 | LIDC-IDRI-0332 | 10.552 | -922.0 |  |
| 334 | LIDC-IDRI-0332 | 10.023 | -910.0 |  |
| 452 | LIDC-IDRI-0447 | 1.547 | -689.0 |  |
| 478 | LIDC-IDRI-0473 | 10.413 | -918.0 |  |
| 617 | LIDC-IDRI-0612 | 10.066 | -899.0 |  |
| 619 | LIDC-IDRI-0614 | 11.086 | -961.0 |  |
| 841 | LIDC-IDRI-0814 | 1.515 | -757.0 |  |
| 1000 | LIDC-IDRI-0655 | 11.329 | -903.0 |  |
| 1011 | LIDC-IDRI-0644 | 1.806 | -707.0 |  |
