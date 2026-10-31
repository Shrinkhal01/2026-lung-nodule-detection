# results/

**Small outputs only:** metrics (CSV/JSON/Markdown), figures (PNG), and predictions.

- One folder per run: `results/<phase>_<name>/seed_<n>/`.
- Every run also saves its **config** and **git commit**, so each number can be traced back.
- **Checkpoints (`.pth`) stay on the server** and are git-ignored (100+ MB each).
- Bring results to the Mac:
  ```bash
  rsync -av --exclude '*.pth' <user>@<server>:/scratch/mtech_shrinkhal/2026-lung-nodule-detection/results/ results/
  ```
