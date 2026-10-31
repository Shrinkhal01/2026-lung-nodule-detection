# v1.0: Lung nodule detection, a hybrid explainable 3D pipeline on LIDC-IDRI

**Research prototype, not a medical device.** Finds lung nodules ≥ 3 mm (the ones ≥ 3 of 4 radiologists agreed on) in a full chest CT. Says nothing about cancer.

## Headline (TEST: 135 unseen scans, 225 nodules, opened once)
- **CPM 0.823** (95% CI **0.767–0.868**); no-learning floor **0.058**.
- **83.6%** of nodules found at **1 false alarm per scan**; **90.2%** at 4.
- **HUSH vs plain 3D CNN: +0.026 [+0.001, +0.078]**: small, weakly supported.
- **Weak spot:** 3–6 mm nodules, **63%** at 1 false alarm per scan.
- Full evidence, 10 documented failures and what was **not** tested: `docs/REALITY_CHECK.md`.

## What is in this release
| Asset | What | Size |
|---|---|---|
| `lungdet-v1.0-hush-r5-weights.zip` | the shipped **HUSH (R5)** ensemble: `r5_seed42…46.pth` | 62 MiB |
| `lungdet-v1.0-baseline-r2-weights.zip` | the plain 3D CNN baseline **R2**: `r2_seed42…46.pth` (to reproduce the baseline row) | 62 MiB |
| `SHA256SUMS.txt` | check the downloads: `shasum -a 256 -c SHA256SUMS.txt` | |
| `operating_points.json` | *(made by `tools/phase5.sh`)* thresholds per false-alarm rate, picked on VAL | small |

## Use it
```bash
git clone https://github.com/Shrinkhal01/2026-lung-nodule-detection && cd 2026-lung-nodule-detection
pip install -r requirements.txt
gh release download v1.0 -p "lungdet-v1.0-hush-r5-weights.zip" && unzip lungdet-v1.0-hush-r5-weights.zip -d models
python predict.py /path/to/dicom_folder --fp-per-scan 1 --out out/
```

## Known limits (short)
- **One dataset (LIDC-IDRI)**, no external hospital; labels are radiologist consensus, not pathology.
- `predict.py` adds a **VAL-picked threshold** and a **merge step** that TEST did not measure.
- `predict.py` was run end to end on a **synthetic phantom** and unit tests; one real TEST scan was run on a Mac CPU (43 min, both nodules found, 1 false alarm); that is a plumbing check, not a result.
- Weights are trained on LIDC-IDRI (CC BY 3.0) only. **No code licence chosen yet.**

## Data and references
LIDC-IDRI: Armato SG III et al. (2015), TCIA, https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX. LUNA16 rule + CPM: Setio et al., *Med Image Anal* 2017. More in the README.
