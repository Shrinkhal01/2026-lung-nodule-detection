# Phase 4: TEST (opened once)

> **Written by Claude at your request** (Sat 31 Oct 2026, ~20:00), for you to read and understand.

Prediction (written at start): *(not given)*

What it does: opens the **sealed TEST set once** (135 scans, 225 nodules, never used for any choice) and measures the **final model** (R5 = HUSH, locked on VAL before opening) against the baselines, **with 95% CIs**.
- It then shows **where** the model looks (occlusion maps), **proves** those maps aren't fake (random-weights sanity check), and collects the **failures** for the report.

Flow:
**final model locked** (length check, on VAL) → **Test once**: score 358,784 TEST candidates with R0 / R2 / R5 (because VAL was used for choices, so only TEST is unbiased) → **Error bars**: 1,000 bootstrap resamples of test scans (because candidates of one scan aren't independent) → **Saliency**: occlusion maps on 40 nodules + random-weights check (because a heat map must be proven, not trusted) → **Troubles**: galleries of hits, false alarms and misses + FROC plot

## Decisions
| Used | Why | Rejected alternative | Number that justified it |
|---|---|---|---|
| **Final model = R5, 10 epochs, 5 seeds** (chosen before TEST) | 30 epochs didn't beat the noise | R5 long | +0.011 vs seed SD 0.014 |
| **`TEST_OPENED` marker**; a 2nd run refuses | one honest look | free re-runs | — |
| **CI over scans**, paired between models | the R5 − R2 difference gets its own CI | CI over candidates | — |
| **Occlusion with normal lung (−850 HU)**, 8 mm block, stride 4 | "what if this spot were normal lung?" | black (air) occluder | — |
| **Scored maps + random-weights check** | v2: layer4 Grad-CAM was worse than random | show maps only | — |

## Results on TEST
| Model | CPM | 95% CI | Sens @ 1/8 FP | @ 1 FP | @ 4 FP |
|---|:-:|:-:|:-:|:-:|:-:|
| R0 (no learning) | 0.058 | 0.031–0.094 | 0.044 | 0.053 | 0.062 |
| R2 (baseline CNN) | 0.797 | 0.725–0.847 | 0.591 | 0.831 | 0.884 |
| **R5 = HUSH** | **0.823** | **0.767–0.868** | **0.702** | **0.836** | **0.902** |
- **R5 − R2 = +0.026, CI +0.001 to +0.078** → a small advantage, **weakly supported** (the lower bound is a hair above 0; on VAL they tied).
- **The gain is at the strictest setting:** 1/8 FP/scan → **70.2% vs 59.1%**.
- **Candidate ceiling on TEST: 0.969.**

**Recall by size (R5, at 1 FP/scan):**

| Size | Nodules | Found |
|---|:-:|:-:|
| 3–6 mm | 27 | **0.63** |
| 6–10 mm | 107 | 0.85 |
| 10–20 mm | 68 | 0.90 |
| 20–30 mm | 18 | 0.89 |
| > 30 mm | 5 | 0.60 (3 of 5; too few to judge) |

**Explanations (R5 seed 42, the 40 most confident nodules):**
- **hit 100%** (hottest spot on the nodule) and **concentration 0.91** (share of heat on it)
- the random-weights network: 0% and 0.00 → **sanity check passed**

## Worked example (real numbers)
- **CPM 0.823:** the mean of 0.702, 0.733, 0.787, 0.836, 0.867, 0.902, 0.933 (at 1/8 … 8 FP/scan) = 5.760 ÷ 7 = **0.823**.
- **In words:** let the model raise **1 false alarm per scan** and it finds **84%** of nodules; allow **4** and it finds **90%**.
- **The CI:** 1,000 re-drawn test sets (135 scans each, drawn with replacement) → the middle 95% of their CPMs = **0.767–0.868**.

## My image + acronym
*(Claude's suggestion)*
- **Image: the board exam.** The practice papers (VAL) are done, the answer sheet is locked (the final model), the sealed paper (TEST) is opened once, and the mark comes with a margin of error.
- **TEST** = **T**est once → **E**rror bars → **S**aliency + sanity check → **T**roubles gallery

## What breaks / limits
- **3–6 mm nodules:** only **63%** found at 1 FP/scan → the weakest group.
- **R5 > R2 is weakly supported:** the CI lower bound is +0.001, and the two tied on VAL.
- **One test set of 135 scans** → wide CIs (±0.05). No external hospital.
- **The explanation scores use the 40 most confident nodules** (the easiest cases).
- **The random-weights maps are almost flat** → a weak sanity check; layer-by-layer randomisation would be stronger (future work).
- **The labels are radiologist consensus**, not pathology.

## What I wrote by hand
- Nothing (your choice).

## Prediction vs result
- No prediction given.
- **Surprise:** TEST (0.823) came out **higher** than VAL (0.779). The likely reason is the **higher candidate ceiling** on TEST (0.969 vs ~0.90), plus wide CIs.

## Still unclear
- *(your questions here)*

## Pipeline (whole project, as of the end of Phase 4)
1–20. As in `PHASE_3B.md`.
21. **Length + fit check:** `scripts/06_train_cnn.py --suffix _long --eval-every 10` + `scripts/09_check_summary.py` → `final_choice.json`
22. **Test once:** `tools/phase4.sh` (marker) → `scripts/10_score_test.py` → `discard/models.py:model_specs`, `load_ensemble`, `ensemble_predict`
23. **Numbers:** `scripts/10b_test_report.py` → `rate/bootstrap.py:per_scan_tables`, `bootstrap_cpm`, `ci` + `rate/froc.py:cpm_from_scores` → `test_report.md`, `froc_test.png`
24. **Explanations:** `scripts/11_explain.py` → `explain/occlusion.py:occlusion_map`, `score_map`, `rank_corr` → galleries + `explain_report.md`
