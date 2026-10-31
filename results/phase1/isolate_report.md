# Phase 1 · isolate report (`scripts/03_isolate.py`)

- scans processed: **897**, failed: **0**, runtime **315 s**
- lung volume (L): median **5.01**, min **0.00**, max **11.13**
- lung median HU: median **-856**
- cache: **20.7 GiB** total, **23.1 MB** per scan (median)
- GT nodules outside the mask: **61** (their diameters: [5.2, 5.4, 5.4, 5.4, 5.9, 6.0, 6.2, 6.3, 6.5, 6.7, 6.8, 6.8, 7.5, 7.6, 8.0])

## Gates

| Gate | Result | Value |
|---|:-:|---|
| 0 scans failed | PASS | 0 of 897 |
| lung volume 2–8 L in ≥ 99% of scans | FAIL | 91.4% |
| lung median HU in [−950, −700] in ≥ 99% of scans | FAIL | 97.8% |
| GT nodule centres inside the lung mask ≥ 98% | FAIL | 94.9% of 1192 |
| median HU at GT centres > −400 (coordinates line up) | PASS | 10 HU |

## Scans to look at

| scan_id | patient_id | lung_L | lung_median_hu | error |
|---|---|---|---|---|
| 54 | LIDC-IDRI-0043 | 0.024 | -898.0 |  |
| 84 | LIDC-IDRI-0074 | 1.929 | -732.0 |  |
| 52 | LIDC-IDRI-0041 | 8.106 | -916.0 |  |
| 219 | LIDC-IDRI-0218 | 0.023 | -434.0 |  |
| 121 | LIDC-IDRI-0116 | 9.406 | -883.0 |  |
| 149 | LIDC-IDRI-0146 | 9.954 | -878.0 |  |
| 204 | LIDC-IDRI-0203 | 8.013 | -906.0 |  |
| 182 | LIDC-IDRI-0181 | 0.18 | -905.0 |  |
| 188 | LIDC-IDRI-0187 | 8.27 | -896.0 |  |
| 264 | LIDC-IDRI-0264 | 0.013 | -973.0 |  |
| 210 | LIDC-IDRI-0209 | 8.314 | -868.0 |  |
| 232 | LIDC-IDRI-0231 | 9.388 | -858.0 |  |
| 261 | LIDC-IDRI-0261 | 8.146 | -892.0 |  |
| 319 | LIDC-IDRI-0318 | 0.009 | -433.0 |  |
| 242 | LIDC-IDRI-0242 | 8.176 | -874.0 |  |
| 304 | LIDC-IDRI-0304 | 8.801 | -911.0 |  |
| 311 | LIDC-IDRI-0311 | 0.001 | -599.0 |  |
| 309 | LIDC-IDRI-0309 | 9.523 | -937.0 |  |
| 337 | LIDC-IDRI-0335 | 0.002 | -431.5 |  |
| 341 | LIDC-IDRI-0339 | 0.08 | -929.0 |  |
