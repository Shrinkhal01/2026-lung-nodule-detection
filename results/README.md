# results/

**Small outputs only:** metrics (CSV/JSON/Markdown), figures (PNG), and predictions.

- **On the server, scripts write to `/scratch/mtech_shrinkhal/lungdet_results/`**, outside the git clone, so `git pull` never collides with results.
- One folder per run: `results/<phase>_<name>/seed_<n>/`.
- Every run also saves its **config** and **git commit**, so each number can be traced back.
- **Checkpoints (`.pth`) stay on the server** and are git-ignored (100+ MB each).
- Bring results to the Mac:
  ```bash
  bash tools/pull_results.sh        # copies /scratch/mtech_shrinkhal/lungdet_results/ (server) -> results/ (Mac)
  ```
