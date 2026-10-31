# tests/

**CPU-only tests on synthetic scans.** They run on the Mac with **no real data**, so they need **0 MB** of downloads.

- **Fake scan** = a box of −1000 HU air with a 10 mm ball of 0 HU inside.
  - The candidate finder must return **1 candidate** at the ball's centre.
- One test file per FINDERS layer, written **with** the code in each phase.
- Run: `python -m pytest tests/ -q` (before every commit).
