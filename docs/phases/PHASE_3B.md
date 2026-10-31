# Phase 3b: HUSH (Hessian channels, Unlabelled pretraining, Size head, on the R2 recipe)

> **Written by Claude at your request** (Sat 31 Oct 2026, ~16:30), for you to read and understand.

Prediction (written at start): *(not given)*. Claude's stated guess before the run: **0.79–0.81**, plausible range 0.76–0.82.

What it does: tests 3 published ideas, **one per rung**, on top of the Phase 3 winner (R2), with 5 seeds each.
- **Result:** none of them raises CPM beyond the seed noise.
- **HUSH (R5) = 0.779 vs R2 = 0.778**, a **tie**.
- But the ablation shows the network **reasons through the physics channels**: an explainability gain, not an accuracy gain.

Flow:
**R2 recipe** → **+H** blobness + vesselness as input channels (because vessel vs ball is a curvature question) → **+U** encoder pretrained by redrawing hidden blocks of ~314k unlabelled cubes (because labels are scarce) → **+S** extra head guesses log(diameter) (because it forces 3D extent) → **staircase + ablation on VAL**

## Decisions
| Used | Why | Rejected alternative | Number that justified it |
|---|---|---|---|
| One change per rung from R2 | Phase 3's bundled R3 lost 0.05 | bundle again | R3 0.711 ± 0.010 vs R2 0.760 ± 0.013 |
| "Earns its place" = mean gain > larger seed SD | don't read noise as progress | any gain | SDs 0.008–0.018 |
| **HUSH = R5** (chosen on VAL) | best ensemble, and explainable | R2 | R5 **0.779** vs R2 0.778 (tie) |
| Pretraining: 6,000 steps, 60% of 4³ blocks hidden | standard masked restoration | — | restoration loss **0.0037**, **19 min** |
| 3 GPUs for this run, 40 GB cap | your call (ahead of time) | 2 GPUs | ~1 h 10 min total |

## Result on VAL (134 scans, 147 nodules)
| Rung | CPM ensemble | 5 seeds | Verdict |
|---|:-:|:-:|---|
| R2 baseline | 0.778 | 0.760 ± 0.013 | baseline |
| R5 = R2 + **H** | **0.779** | 0.768 ± 0.012 | ❌ +0.008, inside the noise |
| R6 = R5 + **U** | 0.776 | 0.762 ± **0.008** | ❌ −0.006 (but the **most stable** across seeds) |
| R7 = R6 + **S** | 0.772 | 0.751 ± 0.018 | ❌ −0.011 (least stable) |

## The ablation (blank a channel at test time, 1 seed)
| Model | All 3 channels | No blobness | No vesselness | HU only |
|---|:-:|:-:|:-:|:-:|
| R5 | 0.776 | 0.663 | 0.695 | **0.397** |
| R6 | 0.748 | 0.577 | 0.729 | 0.199 |
| R7 | 0.774 | 0.528 | 0.727 | 0.157 |
- **The trained HUSH networks depend heavily on the physics channels**, blobness most of all: R5 drops **−0.113** without it.
- **Yet R2, which never had them, reaches the same CPM** → the CNN can learn curvature from HU by itself. H gives the **same accuracy with a readable reason**.
- ⚠ **Caveat:** a blanked channel is something the network never saw in training, so part of each drop is "surprise", not pure reliance. R2 (retrained **without** the channels, 0.760 mean) is the cleaner comparison.

## Worked example (real numbers)
- **Seed noise vs gain:** R5's mean gain over R2 is +0.008; R2's seed SD is 0.013 → **not significant**.
- **Pretraining in numbers:** 6,000 steps × 256 cubes = **1.5 M** restorations; the error fell to **0.0037** (mean squared error on the 0–1 intensity scale).

## My image + acronym
*(Claude's suggestion)*
- **Image: a student given a calculator.**
  - He scores **the same marks** as the student without one (R2).
  - But you can now **see which buttons he presses**.
  - Take the calculator away and he **fails**, because he learned to **lean on it**.
- **HUSH** = **H**essian channels → **U**nlabelled pretraining → **S**ize head → **H**ard-negative-free R2 recipe

## What breaks / limits
- **No accuracy gain** from H, U or S on this data and model size. Possible reasons, **NOT MEASURED**:
  - the small CNN already learns curvature from HU
  - about 6k positives seen ~42× each may already be enough
  - fine-tuning at full learning rate may wash out the pretrained features
- **Ablation by blanking is out-of-distribution** (see the caveat above).
- **VAL chose HUSH = R5** → slightly optimistic; TEST decides once, in Phase 4.
- Stage 1 left 2 GPUs idle for ~8 min (5 seeds over 2 GPUs = 3 + 2).
- Logs were buffered (no live progress) → future scripts use `python -u`.

## What I wrote by hand
- Nothing (your choice).

## Prediction vs result
- Claude predicted 0.79–0.81; the result is **0.779** → **at the bottom of the plausible range**, and **no rung beat the noise**.
- **Surprise:** the network relies **heavily** on blobness (−0.11 without it), yet it isn't **better** than the network that never had it.

## Still unclear
- *(your questions here)*

## Pipeline (whole project, as of the end of Phase 3b)
1–15. As in `PHASE_3.md`.
16. **H:** `discard/hessian.py:hessian_channels` (blur at 1, 2, 4 mm → Hessian via `torch.gradient` → closed-form `sym3_eigvals` → blobness, vesselness)
17. **U:** `scripts/08_pretrain_hush.py` → `discard/store.py:build_store` (~314k TRAIN cubes) → `discard/cnn.py:Restorer` + `block_mask` → encoder saved to `hush_pretrain.pth`
18. **R5–R7:** `scripts/06_train_cnn.py` (rung table; `Small3DNet(in_ch=3, size_head)`; ablation passes for the first seed)
19. **Staircase + ablation:** `scripts/06d_summarize_hush.py`
20. **One command:** `tools/phase3b.sh` (pretraining ‖ R5, then R6 + R7, spread over any number of GPUs)
