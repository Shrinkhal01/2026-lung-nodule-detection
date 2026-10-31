# tools/

**One-command wrappers** around `scripts/`, so a phase is one line. Each file's header comment has the usage.

| Tool | Where | Does |
|---|:-:|---|
| `run.sh` | server | `git pull`, then run one script with the project env |
| `pick_gpus.sh` | server | asks which GPUs are free (sourced by the others; set `GPUS="1 2 3"` to skip the question) |
| `phase3.sh` | server | cubes → R1 + R2/R3 × 5 seeds → staircase |
| `phase3b.sh` | server | HUSH: pretraining, R5/R6/R7 × 5 seeds → staircase |
| `check.sh` | server | 30-epoch length + fit check (R2, R5) |
| `phase4.sh` | server | TEST, **opened once** (a second run refuses unless `FORCE=1`) |
| `pull_results.sh` | Mac | copy small server results here (needs `LUNG_SERVER`) |
| `phase5.sh` | Mac | operating points + `predict.py` on one real scan; reads the server only via rsync |
| `ship.sh` | Mac | tests → commit → push; stops if a test fails |

- `SMOKE=1` on the phase tools runs a few-minute check of the code path first.
