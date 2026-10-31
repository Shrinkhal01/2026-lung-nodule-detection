# Project map: where am I?

> **Read this first** whenever you come back after a break.
> **Now:** Phase 1 ✅ (closing) → next **Phase 2 · nominate** on branch `phase-2-nominate`.

## The metro line
```
main ── phase-0-setup ── phase-1-fetch-isolate ── phase-2 ── phase-3 ── phase-4 ── phase-5
        SETUP ✅          PLANET ✅                N (next)   D          E + R      S
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
- [ ] **2 · nominate**: classical candidates + R0
- [ ] **3 · discard**: R1–R4 training (overnight)
- [ ] **4 · explain + rate**: test once, CIs, heat maps
- [ ] **5 · ship**: predict.py, Gradio, final docs

## Open items carried forward
- Candidate size cap must be **≥ 50 mm** (48 GT nodules > 30 mm, max 46.8 mm) → Phase 2.
- **5 broken masks** (0619, 0697, 0403, 0523, 0655): report their candidate counts separately in Phase 2.
- Recall ceiling **99.6%** (5 GT nodules outside the mask).
- Server results: `/scratch/mtech_shrinkhal/lungdet_results/` → Mac with `bash tools/pull_results.sh`.
