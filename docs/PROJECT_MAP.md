# Project map: where am I?

> **Read this first** whenever you come back after a break.
> **Now:** Phases 0–5 ✅, **v1.0 tagged**. **TEST (opened once): HUSH CPM 0.823 [0.767–0.868]**. Nothing is pending; new work starts on a new branch.

## The metro line
```
main ── phase-0-setup ── phase-1-fetch-isolate ── phase-2 ── phase-3 ── phase-4 ── phase-5
        SETUP ✅          PLANET ✅                N ✅        D ✅        E + R ✅    S ✅
```
Each phase: branch → work → notes (`docs/phases/PHASE_N.md`) → PR → merge into `main`.

## Where things live
| Where | What |
|---|---|
| **Mac** `~/Shrinkhal-Github/Projects/2026-lung-nodule-detection` | code, docs, tests (`source .venv/bin/activate`) |
| **GitHub** `Shrinkhal01/2026-lung-nodule-detection` (public) | code + small results |
| **Server** `/scratch/mtech_shrinkhal/2026-lung-nodule-detection` | the repo clone (only `git pull`, never commit) |
| **Server** `/scratch/mtech_shrinkhal/lidc_raw/` | LIDC-IDRI CT scans (125 GB) |
| **Server env** | `lungenv` (venv on scratch: torch 2.14 + CUDA 13, Python 3.13) |

## Daily commands
```bash
# SERVER
lungenv                                                    # turn on the project env
cd $SCRATCH/2026-lung-nodule-detection && git pull         # get the newest code
tmux ls                                                    # what is running?
tmux attach -t fetch                                       # look at a job (leave: Ctrl-b, d)
python scripts/01_fetch.py --check                         # download status

# MAC
cd ~/Shrinkhal-Github/Projects/2026-lung-nodule-detection && source .venv/bin/activate
python -m pytest tests/ -q                                 # before every commit
git cc "feat: ..."                                         # commit with Claude as co-author
```

## Phase checklist
- [x] **0 · setup**: repo, env, download, gate PASS (scan #1, 1 nodule, 4/4 readers)
- [x] **1 · fetch + isolate**: 897 scans, 1,192 GT nodules, 99.6% inside the lung mask, cache 21.4 GiB
- [x] **2 · nominate**: per-size quotas, TRAIN recall 96.0% at 2,643/scan, R0 CPM 0.088
- [x] **3 · discard**: R2 3D CNN VAL CPM 0.778 (R3 tricks hurt, fusion ties)
- [x] **3b · HUSH**: R5 VAL CPM 0.779 (tie); relies on blobness (no blobness → 0.663)
- [x] **check**: 30 epochs +0.01 (noise) · TRAIN AUC 1.000 vs VAL 0.988 → keep 10 epochs
- [x] **4 · TEST**: HUSH 0.823 [0.767–0.868] vs R2 0.797 · occlusion hit 100%, conc 0.91
- [x] **5 · ship**: `predict.py`, Gradio `app/app.py`, README results, `docs/REALITY_CHECK.md`, `docs/figures/` (PPT), `docs/RELEASE_v1.0.md`; merged, tag `v1.0`

## Open items carried forward
- Candidate size cap must be **≥ 50 mm** (48 GT nodules > 30 mm, max 46.8 mm) → done in Phase 2 (LoG sigmas up to ~45 mm).
- **5 broken masks** (0619, 0697, 0403, 0523, 0655): report their candidate counts separately in Phase 2.
- Recall ceiling **99.6%** (5 GT nodules outside the mask).
- Server results: `/scratch/mtech_shrinkhal/lungdet_results/` → Mac with `bash tools/pull_results.sh`.
