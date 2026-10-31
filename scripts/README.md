# scripts/

**Numbered steps you run on the server**, in order, inside `tmux`. Each script is thin: it reads `configs/default.yaml` and calls code from `src/lungdet/`.

| Script | Phase | FINDERS | Does |
|---|:-:|:-:|---|
| `00_check_env.py` | 0 | — | Python, PyTorch, CUDA, `/scratch`; same on Mac and server |
| `00_gate_one_scan.py` | 0 | — | **gate**: read one real scan + its radiologist marks |
| `01_fetch.py` | 0–1 | F | check / resume the LIDC-IDRI download |
| `02_ground_truth.py` | 1 | F | **P-L-A**: pick scans, label nodules, assign patient splits |
| `03_isolate.py` | 1 | I | **N-E-T**: HU → 1 mm → lung mask → one `.npz` per scan |
| `03b_mask_sweep.py` | 1 | I | choose the mask closing radius on TRAIN scans |
| `03c_qc_outliers.py` | 1 | I | list scans outside the sanity ranges (lung 2–10 L, median HU) |
| `03d_padding_check.py` | 1 | I | test the padding fix on named scans before a full rerun |
| `04_nominate.py` | 2 | N | LoG candidates; selection rule chosen automatically on TRAIN |
| `04b_miss_diagnosis.py` | 2 | N | why TRAIN nodules are missed; does scale-aware bundling help? |
| `05_build_patches.py` | 3 | D | cut 32 mm cubes for candidates |
| `06_train_cnn.py` | 3 / 3b | D | train one 3D CNN rung on one GPU, score all VAL candidates |
| `06b_xgb.py` | 3 | D | R1: XGBoost on hand-made features (CPU) |
| `06c_summarize.py` | 3 | D | VAL staircase R0 → R4 |
| `06d_summarize_hush.py` | 3b | D | HUSH staircase R2 → R7 + channel ablation |
| `06e_check_h_channels.py` | 3b | D | do the blobness / vesselness channels carry signal on real cubes? |
| `08_pretrain_hush.py` | 3b | D | **U**: self-supervised pretraining by masked cube restoration |
| `09_check_summary.py` | 3b | D | does 3× longer training help? under/over-fit check |
| `10_score_test.py` | 4 | R | score every TEST candidate with ONE ensemble (**test set opened once**) |
| `10b_test_report.py` | 4 | R | TEST numbers with 95% CIs over patients |
| `11_explain.py` | 4 | E | occlusion heat maps + sanity check + troubles gallery on TEST |
| `12_operating_points.py` | 5 | S | pick the threshold per false-alarms-per-scan setting, on VAL |
| `13_make_figures.py` | 5 | S | PPT figures → `docs/figures/*.png` |
| `14_check_scan.py` | 5 | S | compare `predict.py` output with radiologist marks for one scan |

- **Rule:** every number in the README must come from one of these scripts, so anyone can rerun it.
- One-scan inference is `predict.py` in the repo root; the demo is `app/app.py`.
