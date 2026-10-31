# Brutal Honesty & Project Reality Check: lung nodule detection v1.0

> Written Sat 31 Oct 2026 for the **v1.0 release**, the **viva**, and any demo.
> Rule: **no number without a source.** Anything not measured is written `NOT MEASURED`.
> Results were committed in `83b38d5` (Phase 4 TEST), server = 4 × NVIDIA H200 NVL (2 GPUs used), data = LIDC-IDRI.

---

## 1. The truth in one paragraph

- This is a **research prototype** that finds **≥ 3 mm lung nodules that ≥ 3 of 4 LIDC radiologists agreed on**, in CT scans from **one public dataset**.
- It works **clearly better than no learning** (TEST CPM **0.823** vs **0.058**) and finds **83.6%** of nodules at **1 false alarm per scan**.
- **The "advanced ML" on top of a plain 3D CNN did not help beyond noise on VAL** (R3, R5, R6, R7). The shipped HUSH model (R5) is **a choice**, not a proven win: on TEST it is **+0.026 CPM [+0.001, +0.078]** over the plain CNN, and on VAL they **tied**.
- **Nothing here is validated outside LIDC-IDRI.** The shipped `predict.py` adds a threshold and a merge step that **TEST did not measure**, and it has **only been run end to end on a synthetic phantom** (random weights), not on a real new scan.
- It says **nothing about cancer** and **is not a medical device**.

## 2. What it does / what it does not

| Does | Does not |
|---|---|
| Finds nodule **candidates** (3D blobs inside the lung mask) | Tell benign from malignant (that was the separate v2 project) |
| Scores each with a 3D CNN (5-seed ensemble) | Use any radiologist rating as an input (circular) |
| Shows **where** it looked (occlusion, scored) | Prove the CNN "understands" nodules (the check is weak, see §6) |
| Reports FROC / CPM **with 95% CIs**, TEST opened once | Say anything about other scanners, hospitals, thick slices, children, or non-LIDC labels |

## 3. Numbers

All on **TEST = 135 scans, 225 nodules**, from `results/phase4/test_metrics.json`, made by `scripts/10_score_test.py` + `scripts/10b_test_report.py` (`bash tools/phase4.sh`). Split by patient (seed 42).

| Metric | Value | 95% CI | n | Dataset / split | Reproduce |
|---|:-:|:-:|:-:|---|---|
| **CPM, R5 (HUSH, shipped)** | **0.823** | 0.767–0.868 (bootstrap, 1,000 × scans) | 135 scans | LIDC-IDRI TEST | `python scripts/10b_test_report.py` |
| CPM, R2 (plain 3D CNN) | 0.797 | 0.725–0.847 | 135 scans | TEST | same |
| CPM, R0 (no learning) | 0.058 | 0.031–0.094 | 135 scans | TEST | same |
| R5 − R2 (paired bootstrap) | +0.026 | +0.001 to +0.078 | 135 scans | TEST | same |
| Sensitivity @ 1 FP/scan (R5) | 0.836 (188/225) | 0.782–0.878 (Wilson) | 225 nodules | TEST | same |
| Sensitivity @ 1/8 FP/scan (R5) | 0.702 (158/225) | 0.639–0.758 (Wilson) | 225 nodules | TEST | same |
| Sensitivity @ 4 FP/scan (R5) | 0.902 (203/225) | 0.856–0.935 (Wilson) | 225 nodules | TEST | same |
| Candidate ceiling (nodules the blob detector reached) | 0.969 (218/225) | 0.937–0.985 (Wilson) | 225 nodules | TEST | same |
| Occlusion hit / concentration (trained) | 100% / 0.91 | NOT MEASURED | 40 nodules | TEST, seed 42 | `python scripts/11_explain.py` |
| Occlusion hit / concentration (random weights) | 0% / 0.00 | NOT MEASURED | 40 nodules | TEST | same |

- **Wilson intervals treat nodules as independent.** Several nodules sit in one scan, so they are **optimistic**; the **bootstrap over scans** (first four rows) is the honest one.
- **VAL numbers** (used for choices, so slightly optimistic): R2 0.778, **R5 0.779**, 134 scans / 147 nodules, `results/phase3b/phase3b_report.md`.
- **Runtime:** training one R5 seed took **685 s** (`results/phase3b/scores/r5_seed42.json`). **`predict.py` on one real scan (LIDC-IDRI-0405), Mac CPU (Apple Silicon, 8 GB), 5-model ensemble: 2,578 s ≈ 43 min** (scoring 2,525 s for 2,522 candidates ≈ 0.2 s per cube per model). My earlier extrapolation from a cloud-CPU phantom (43 ms per cube per model, ≈ 10 min) was about **4× too optimistic**.
- **Sample-size flags:** every size group is **n < 100** except 6–10 mm (n = 107); **> 30 mm has n = 5**.

## 4. Baselines

| Method | TEST CPM | VAL CPM | Verdict |
|---|:-:|:-:|---|
| R0 blob strength, no learning | 0.058 [0.031, 0.094] | 0.088 | the floor |
| R1 XGBoost on ~20 hand-made features | NOT MEASURED (not a final candidate) | 0.275 | far below a CNN |
| R2 plain 3D CNN | 0.797 [0.725, 0.847] | 0.778 | strong baseline |
| **R5 HUSH** | **0.823 [0.767, 0.868]** | 0.779 | **CI overlaps R2 heavily** |

- **R5 vs R2:** the paired difference's lower bound is **+0.001**. Plainly: **the advantage is small and weakly supported**, and the auto-written line in `results/phase4/test_report.md` ("a real difference") overstates it.
- **Seed noise:** single seeds move by about ±0.01 CPM (R2 0.760 ± 0.013, R5 0.768 ± 0.012), which is as big as any rung's gain on VAL.
- **All of the gain on TEST sits at the strictest setting:** 1/8 FP/scan → **70.2% vs 59.1%**. At 1 FP/scan they are **83.6% vs 83.1%**.

## 5. Failure catalogue (real failures, with numbers)

| # | Input | Expected | Got | Why |
|--:|---|---|---|---|
| 1 | Full run of `03_isolate.py`, 897 scans | lung masks of 2–10 L | **31 scans < 2 L**, holding **52 GT nodules** (45 outside the mask) | border air was removed **in 3D**: one thin leak path deleted a whole lung → fixed with per-slice removal (23 → 5 bad scans on the sweep set) |
| 2 | Wall-attached large nodules, closing r = 10 mm | inside mask | **16 GT nodules (14–38 mm) missed** | closing only fills wall "bites" narrower than ~20 mm → r = 15 chosen on TRAIN (91.4% → 98.0% inside) |
| 3 | 5 scans: 0619, 0697, 0403, 0523, 0655 | clean lung mask | mask includes body / outside air | **not fixed** (hypothesis "tissue-like padding" was **false**: padding −3024 or none); accepted because a too-big mask adds false candidates, it does not lose nodules |
| 4 | Candidate step with a 3,000/scan cap | ≥ 95% TRAIN recall | **92.8%**; **56 of 59** misses ranked below the cap (faint, ~8 mm) | small faint blobs lose to big bright vessels → per-size top-1200 (96.0% at 2,643/scan) |
| 5 | R3: focal loss + hard negatives + rot90 + TTA in one rung | better than R2 | **0.717 vs 0.778** (seeds 0.711 ± 0.010 vs 0.760 ± 0.013) | bundled tricks, one of them hurts, and we cannot say which → later rungs change **one thing** |
| 6 | TEST nodules 3–6 mm | found | **17 of 27 (63%)** at 1 FP/scan; **3 never became candidates** | small and faint; the blob detector and the CNN both struggle |
| 7 | TEST nodules > 30 mm | found | **3 of 5** | too few cases to know why (NOT MEASURED) |
| 8 | 7 of 225 TEST nodules | proposed as candidates | **never proposed** | recall ceiling 0.969 is a hard limit of the first stage |
| 9 | v2 project: Grad-CAM (layer 4) | explains the CNN | scored **worse than a random-weights network** | → replaced by occlusion + a random-weights check |
| 10 | 30-epoch run, TRAIN + VAL on one GPU | fits the 40 GB cap | **out of memory** (36 GB used, +8 GB needed) | VAL cubes now stream from RAM; same bug class hit the new `--explain` on a CPU (512 cubes per batch) → batch size is now an argument |

- **Per-scan misses and false alarms** you can look at: `results/phase4/explain/gallery_misses.png`, `gallery_false_alarms.png`.
- **Overfitting:** TRAIN AUC 1.000 vs VAL AUC 0.988 (gap 0.011, 0.017 at 30 epochs) → mild; 30 epochs gave +0.009 / +0.011, **below** the seed SD → kept 10 epochs.

## 6. Robustness

**Tried**
- 5 seeds per rung; ensemble vs single seeds.
- 1,000-resample bootstrap over test scans (paired between models).
- **Ablation:** blank a channel at test time (R5, 1 seed, VAL): all 0.776 · no vesselness 0.695 · no blobness **0.663** · HU only **0.397**. *Caveat:* a blanked channel is an input the network never saw.
- Random-weights sanity check for the occlusion maps (hit 0%, concentration 0.00; rank correlation with the trained map median −0.06). **The random maps are almost flat, so this check is weak** (a layer-by-layer randomisation would be stronger).
- Length check: 10 vs 30 epochs.
- **Plumbing check with the real shipped weights** (R5 seeds 42–44, strict `state_dict` load) on the synthetic phantom: the planted 10 mm ball was **candidate rank 1 of 258, 0 mm from the centre, probability 1.000** (median candidate 0.0003). This only shows the code and weights fit together; a phantom is **not** a clinical result.

**NOT tried (so we know nothing about it)**
- Any other scanner, hospital, country or dataset (no external test).
- Slice thickness > 2.5 mm, low-dose vs standard-dose, contrast vs non-contrast, different reconstruction kernels.
- Noise, blur, motion or JPEG-style perturbations; adversarial examples.
- Probability **calibration** (the "probability" is a score for ranking, **NOT MEASURED** as a probability).
- Per-radiologist agreement levels (3 of 4 vs 4 of 4).
- `predict.py` on a real DICOM chest; the DICOM reader was tested on a **synthetic series** only.

## 7. Trade-offs with numbers

| Choice | Gain | Cost |
|---|---|---|
| Per-size top-1200 instead of global top-N | same ≥ 95% TRAIN recall with **2,643** instead of **5,000** candidates/scan (**−47%**) | 4 more knobs (one quota per blob size) |
| Closing radius 15 mm, not 20 | 20 mm adds only **+0.7 points** of GT inside (≈ 1 of 151 nodules) | 20 mm adds **+4.7% lung volume** (more wall and mediastinum → more false candidates) |
| 5-seed ensemble | VAL CPM **0.779** vs single-seed mean **0.768** (+0.011) | **5×** inference (≈ 2,643 × 5 cubes per scan; ≈ 43 min measured on a MacBook CPU) |
| 10 epochs, not 30 | 30 epochs gave +0.009 / +0.011, below the SD (0.012 / 0.014) | none, 3× less training |
| HUSH (R5) over R2 | TEST +0.026 at the strictest setting (1/8 FP: 0.702 vs 0.591); explainable through its channels | **+0.001 lower CI bound**, tie on VAL, needs 3-channel Hessian step (about 3 extra Gaussian blurs per cube) |
| Merge nearby detections into one finding (predict.py) | one row per nodule, readable output | **not part of the TEST numbers**; could merge two real nodules closer than the blob radius |
| LoG blob detector instead of a segmentation network | no training, every rule explainable | recall ceiling **0.969** |

- **Complexity:** LoG is 7 sizes × O(V) filters over V lung voxels; closing is two O(V) distance transforms; the CNN is O(candidates × models). The CNN dominates run time.

## 8. What we do **not** claim

- That it **detects cancer**, or that it is safe or useful in a clinic.
- That it works on **other data** (scanners, hospitals, thick slices).
- That **HUSH beats a plain 3D CNN** (the evidence is weak).
- That the **maps prove** the model "reasons like a radiologist" (they show where removing the region hurts the score).
- That the **detection probability is calibrated**.
- That **`predict.py` reproduces the TEST numbers exactly** (threshold + merge step were not measured on TEST).
- That the result is **state-of-the-art** or **real-time**. No comparison with published LUNA16 leaderboards was made, and run times are in §3.

## 9. Tough questions (and honest answers)

1. **"Your headline is 0.823. Is that real?"** It is one TEST set, opened once, with a bootstrap CI of **0.767–0.868**. It means "on LIDC-IDRI, on these 135 scans", not "in general".
2. **"Is HUSH better than a plain CNN?"** Weakly at best: **+0.026 [+0.001, +0.078]** on TEST and a **tie** on VAL. I shipped it for explainability and the small TEST edge, not because it is proven.
3. **"Why did TEST come out higher than VAL (0.823 vs 0.779)?"** Probably a **higher candidate ceiling on TEST** (0.969 vs about 0.90) plus wide CIs. I did not run a test to confirm; **NOT MEASURED**.
4. **"Did you tune on the test set?"** No. Every choice (mask radius, candidate rule, rungs, epochs, final model) was made on TRAIN or VAL. TEST was opened once (`TEST_OPENED` marker).
5. **"VAL was used for choices, so isn't VAL optimistic?"** Yes, slightly. That is why only TEST numbers are in the headline.
6. **"Why not use LUNA16 pretrained weights?"** LUNA16 is built from LIDC, so those weights **have seen our test scans** (leakage).
7. **"Why a blob detector before the CNN?"** To cut ~20,860 peaks per scan to 2,643 without training, at the price of a **0.969 ceiling**. A 3D detector would remove that ceiling but needs masks and much more training.
8. **"Does the occlusion map prove the model looks at the nodule?"** It shows the score drops most when the nodule region is replaced by normal lung (hit 100%, concentration 0.91 on the 40 most confident nodules, which are the **easy** cases), and a random network does not (0%). The random maps are almost flat, so this is **a weak test**.
9. **"How do I know the demo threshold gives 1 false alarm per scan?"** Thresholds are picked on **VAL** (`scripts/12_operating_points.py`) and merging lowers the visible count. The **actual false-alarm rate on a new scan set: NOT MEASURED.**
10. **"What would you do next?"** (a) an **external test set**; (b) layer-by-layer randomisation for the maps; (c) fix the **5 broken masks**; (d) look at the **3–6 mm** misses specifically; (e) report 3-of-4 vs 4-of-4 agreement separately.
