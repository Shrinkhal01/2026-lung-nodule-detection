# Phase 1 · sanity-range outliers (`scripts/03c_qc_outliers.py`)

20 of 897 scans are outside lung 2–10 L or median HU −950..−700.

| patient | lung_L | median_HU | blobs | 2nd/1st blob | lung z-extent mm | scan z mm | mm to image edge |
|---|---|---|---|---|---|---|---|
| LIDC-IDRI-0814 | 1.51 | -757 | 1 | 0.0 | 178 | 200 | 21 |
| LIDC-IDRI-0447 | 1.55 | -689 | 3 | 0.0 | 201 | 201 | 34 |
| LIDC-IDRI-0403 | 1.72 | -743 | 1 | 0.0 | 166 | 166 | 0 |
| LIDC-IDRI-0644 | 1.81 | -707 | 1 | 0.0 | 209 | 220 | 79 |
| LIDC-IDRI-0074 | 1.96 | -730 | 4 | 0.0 | 227 | 271 | 72 |
| LIDC-IDRI-0176 | 2.1 | -636 | 1 | 0.0 | 266 | 267 | 36 |
| LIDC-IDRI-0523 | 2.12 | -599 | 1 | 0.0 | 202 | 202 | 0 |
| LIDC-IDRI-0170 | 2.25 | -563 | 2 | 0.0 | 224 | 257 | 98 |
| LIDC-IDRI-0479 | 2.84 | -676 | 1 | 0.0 | 230 | 238 | 24 |
| LIDC-IDRI-0070 | 2.84 | -683 | 2 | 0.0 | 219 | 331 | 58 |
| LIDC-IDRI-0661 | 6.26 | -697 | 1 | 0.0 | 299 | 299 | 31 |
| LIDC-IDRI-0619 | 6.49 | -305 | 10 | 0.0 | 259 | 259 | 0 |
| LIDC-IDRI-0697 | 6.68 | -354 | 2 | 0.0 | 262 | 262 | 0 |
| LIDC-IDRI-0332 | 10.02 | -910 | 1 | 0.0 | 355 | 364 | 19 |
| LIDC-IDRI-0612 | 10.07 | -899 | 1 | 0.0 | 336 | 353 | 23 |
| LIDC-IDRI-0146 | 10.12 | -876 | 5 | 0.84 | 386 | 386 | 34 |
| LIDC-IDRI-0473 | 10.41 | -918 | 1 | 0.0 | 330 | 363 | 25 |
| LIDC-IDRI-0332 | 10.55 | -922 | 1 | 0.0 | 361 | 361 | 22 |
| LIDC-IDRI-0614 | 11.09 | -961 | 1 | 0.0 | 337 | 364 | 24 |
| LIDC-IDRI-0655 | 11.33 | -903 | 1 | 0.0 | 347 | 390 | 0 |

- **mm to image edge ≈ 0** → the mask reaches the image border (possible leak or a tight field of view).
- **lung z-extent ≪ scan z** → the scan does not cover the whole chest, or part of the lung was lost.
- **2nd/1st blob ≈ 0** → only one lung blob kept (lungs joined, or one lung missing).
