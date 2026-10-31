# Phase 1 · isolate report (`scripts/03_isolate.py` --limit 5)

- scans processed: **5**, failed: **0**, runtime **28 s**
- lung volume (L): median **5.42**, min **3.61**, max **6.13**
- lung median HU: median **-866**
- cache: **0.1 GiB** total, **23.0 MB** per scan (median)
- GT nodules outside the mask: **0** (their diameters: [])

## Gates

| Gate | Result | Value |
|---|:-:|---|
| 0 scans failed | PASS | 0 of 5 |
| lung volume 2–8 L in ≥ 99% of scans | PASS | 100.0% |
| lung median HU in [−950, −700] in ≥ 99% of scans | PASS | 100.0% |
| GT nodule centres inside the lung mask ≥ 98% | PASS | 100.0% of 7 |
| median HU at GT centres > −400 (coordinates line up) | PASS | 56 HU |
