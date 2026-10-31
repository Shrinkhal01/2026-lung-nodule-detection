# configs/

**One YAML file holds every path and setting**, because a number hidden inside code can't be checked or changed.

- `default.yaml` (Phase 0) holds:
  - **paths:** `/scratch/mtech_shrinkhal/...` (raw scans, cache, candidates, results)
  - **ground truth:** slice thickness ≤ 2.5 mm, ≥ 3 of 4 readers, ≥ 3 mm
  - **isolate:** 1 mm spacing, lung threshold (start: −320 HU)
  - **nominate:** HU threshold and size range 3–30 mm (tuned on **train** scans only)
  - **discard:** cube size 32 mm, seeds 42–46, batch size, learning rate
  - **GPUs:** `CUDA_VISIBLE_DEVICES=1,2,3`
- **Rule:** scripts read the config; they never hard-code a path.
- 🖼 Picture it as the **dashboard of the car**: every knob in one place.
