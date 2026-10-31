# Phase 3: discard (CUBES), the baseline staircase

> **Written by Claude at your request** (Sat 31 Oct 2026, ~14:45), for you to read and understand.

Prediction (written at start): *(not given)*

What it does: teaches models to look at each of the ~2,640 candidates per scan and say **"nodule" or "not"**.
- The best one, a **3D CNN**, reaches **VAL CPM 0.778**: at 1 false alarm per scan it finds **84%** of nodules.
- That's up from **0.088** with no learning.

Flow:
**candidates** → **Cut 32 mm cubes** (because the network needs the 3D shape around each spot) → **Use hand-made features** in XGBoost = R1 (because numbers are explainable) → **Build a 3D CNN** = R2 (because vessels vs nodules is a 3D-shape question) → **Extra tricks** = R3 (focal loss, hard negatives, rotations, TTA) → **Stack scores** = R4 (fuse the best CNN with R1) → **staircase on VAL**

## Decisions
| Used | Why | Rejected alternative | Number that justified it |
|---|---|---|---|
| TRAIN **sample** per scan: 150 random + 50 hardest negatives, positives, GT centre ×5 (±2 mm) | 1.66 M candidates is too many; balance matters | all candidates | **131,634** cubes, **6,034** positives (1 : 21) |
| **Balanced batches**, 25% positives | without it, "always no" is 95% right | plain shuffling | 128 positives + 384 negatives per batch of 512 |
| Cubes **on the GPU**, no data loader | fast and simple | DataLoader from disk | train 8.0 GiB + val 21.5 GiB fit in 141 GB |
| **Small 3D CNN** (~1.5 M weights) | fast; 32³ inputs | 3D ResNet-18 (33 M, v2) | R2 **0.778** |
| **5 seeds** per CNN rung | one run moves by chance | 1 seed | R2 seeds **0.746–0.776** (SD **0.013**) |
| **R4 fuses the best CNN** (picked on VAL) | fusing the weaker R3 gave 0.730 | fuse R3 | R4 on R2: **0.780** |
| **2 GPUs** from now on | fair use of a shared server | 3 GPUs | your choice |

## The staircase on VAL (134 scans, 147 GT nodules)
| Rung | CPM | Sens @ 1 FP/scan | Verdict |
|---|:-:|:-:|---|
| R0 blob strength | 0.088 | 0.109 | floor |
| R1 XGBoost | 0.275 | 0.238 | ✅ > R0 |
| **R2 3D CNN** | **0.778** (0.760 ± 0.013) | **0.844** | ✅ **best single model** |
| R3 + 4 tricks | 0.717 (0.711 ± 0.010) | 0.769 | ❌ **worse than R2 on all 5 seeds** |
| R4 = R2 + R1 fused | 0.780 | 0.823 | ⚖ **tie** with R2 (+0.002, inside the 0.013 seed noise) |

## Worked example (real numbers)
- **CPM 0.778 in words:** average sensitivity over 1/8 … 8 false alarms per scan.
  - At **1 FP/scan → 84.4%** of nodules found.
  - At **4 FP/scan → 87.8%** (R4: 89.1%), close to the **candidate ceiling of about 90%**.
- **From 2,640 to about 1:** the CNN must push about **2,639 of every 2,640 candidates** below the threshold to reach 1 FP/scan.
- **R3's drop is not chance:** R2 seeds span 0.746–0.776, R3 seeds span 0.695–0.721 → the ranges **don't overlap**.

## My image + acronym
*(Claude's suggestion)*
- **Image: training a sniffer dog with flashcards.**
  - Each card is a 32 mm cube with "nodule" or "not" on the back.
  - R2 = a dog trained on **normal** cards.
  - R3 = the same dog also drilled on **trick cards**. If some trick cards have **wrong answers on the back**, the drill makes the dog worse.
- **CUBES** = **C**ut cubes → **U**se hand features (R1) → **B**uild the 3D CNN (R2) → **E**xtra tricks (R3) → **S**tack scores (R4)

## What breaks / limits
- **R3 bundled 4 changes** (focal loss, hard negatives, rot90, TTA) → we **can't tell which one hurt**.
  - Main suspect: hard negatives that are really unlabelled small nodules or marks pylidc lacks (label noise). **NOT MEASURED.**
- **Fusion doesn't help** (R4 ties R2) → the hand-made features add nothing the CNN doesn't already see.
- **VAL was used for choices** (R4's best CNN) → VAL numbers are **slightly optimistic**. TEST (135 scans, 358,784 cubes already cut) is opened **once**, in Phase 4.
- **The ceiling is the candidate recall** (about 90% on VAL). No rung can find a nodule Phase 2 missed.
- Numbers **aren't comparable** to LUNA16 papers (different candidates, folds and splits).

## What I wrote by hand
- Nothing (your choice: reading over writing).

## Prediction vs result
- No prediction given.
- **Surprise:** the "advanced" rung (R3) **lost 0.05 CPM**. Simple beat fancy. → HUSH starts from **R2's recipe** and adds **one change per rung**.

## Still unclear
- *(your questions here)*

## Pipeline (whole project, as of the end of Phase 3)
1–10. As in `PHASE_2.md` (setup, ground truth, lung mask, candidates).
11. **Cubes:** `scripts/05_build_patches.py` → `discard/patches.py:crop_cubes` → `$SCRATCH/patches/{train,val,test}_cubes.npy` + `_meta.parquet`
12. **R1:** `scripts/06b_xgb.py` → `discard/features.py:cube_features` + detector numbers → XGBoost → VAL scores
13. **R2 / R3:** `scripts/06_train_cnn.py` → `discard/cnn.py:Small3DNet`, `augment`, `focal_loss`, `predict` → checkpoint + VAL scores
14. **Staircase:** `scripts/06c_summarize.py` → R0–R4 CPM via `rate/froc.py`, R4 cross-fitted fusion
15. **One command:** `tools/phase3.sh` (tests → cubes → R1 + CNNs on 2 GPUs → staircase); `SMOKE=1` for a 5-minute check
