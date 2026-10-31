# scripts/

**Numbered steps you run on the server**, in order, inside `tmux`. Each script is thin: it reads `configs/default.yaml` and calls code from `src/lungdet/`.

| Script | Phase | FINDERS | Does |
|---|:-:|:-:|---|
| `00_check_env.py` | 0 | — | Python, PyTorch, CUDA, GPUs 1–3, `/scratch` space, pylidc loads 1 scan |
| `01_fetch.py` | 0–1 | F | download CT scans (resumable, logs MB/s), test patients first |
| `02_ground_truth.py` | 1 | F | pylidc → `nodules.csv` (≥ 3 of 4 readers) + ignore zones + **patient splits** |
| `03_isolate.py` | 1 | I | HU → 1 mm → lung mask → `cache/<series>.npz` |
| `04_nominate.py` | 2 | N | classical candidates → `candidates.csv` + recall report |
| `05_build_patches.py` | 3 | D | 32 mm cubes + labels for every candidate |
| `06_train_discard.py` | 3 | D | XGBoost (R1), 3D CNN (R2/R3), fusion (R4); `--seed`, `--rung` |
| `07_rate.py` | 4 | R | FROC, CPM, bootstrap CI; `--final` opens the **test set once** |
| `08_explain.py` | 4 | E | occlusion + SHAP + random-weights sanity check |
| `09_predict.py` | 5 | S | one scan in → detections out (CSV + images) |

- **Rule:** every number in the README must come from one of these scripts, so anyone can rerun it.
