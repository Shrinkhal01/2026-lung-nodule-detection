# rate/ (R: Phase 4)

**Turns detections into honest numbers.**

- **Hit matching:**
  - detection centre within the nodule radius → hit
  - duplicates on one nodule count once
  - ignore zones are skipped
- **FROC** → sensitivity vs false positives per scan.
- **CPM** → mean sensitivity at 1/8, 1/4, 1/2, 1, 2, 4, 8 FP/scan.
  - Example: 6.07 ÷ 7 = 0.867 (not a result).
- **Bootstrap 95% CI** → resample **patients** 1,000 times.
- **Test once** → `07_rate.py --final` refuses a second run.
