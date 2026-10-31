# app/ (S: Phase 5)

**Gradio demo.** Runs on your Mac (CPU) on one scan cache you copied with `tools/phase5.sh`; it needs no server session.

- Pick a **TEST scan** (or type a **DICOM folder / cached `.npz`** path) → **Find nodules**.
- **Red circles** = findings; **green dashed** = what the radiologists marked (only for TEST scans).
- The **dropdown** sets the operating point (false alarms per scan). Thresholds were picked on **VAL**, never on TEST.
- **Where did it look?** → occlusion map for finding #n (one model of the ensemble), with the blobness channel next to it.

```bash
pip install -r requirements-demo.txt
FORCE_CPU=1 python app/app.py          # (FORCE_CPU: skip Apple's MPS, untested here) binds 127.0.0.1:7860 only; open http://localhost:7860
# type in "path to a DICOM folder / cached .npz":   data/phase5_local/<series_uid>.npz
```

- A TEST scan typed as its cached `.npz` gets the **green dashed radiologist circles** too (looked up by series UID in `results/phase1/`).
- The **TEST-scan dropdown** stays empty on the Mac; it only fills on the server, where the whole cache exists.
- On the Mac the run is on CPU: about **43 minutes per scan** (measured) with the 5-model ensemble (`--models 1` is 5× faster, not the measured setup).
- No SHAP bars: the final model is the CNN ensemble (the XGBoost rung R1 scored VAL CPM 0.275 and was not used).
- Needs `gradio` and the weights in `./models/` (or `--models-dir`, or `LUNGDET_MODELS`). **Not a medical device.**
