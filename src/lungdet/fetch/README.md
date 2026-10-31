# fetch/ (F: Phase 0–1)

**Gets the scans and the truth.**

- **download** → `idc-index` pulls LIDC CT series from the public IDC bucket into `/scratch/.../lidc_raw/`.
  - Resumable: finished series are skipped.
  - The speed is logged in MB/s.
- **ground truth** → `pylidc` clusters the up-to-4 readers' marks into nodules.
  - **nodule** = ≥ 3 of 4 readers **and** ≥ 3 mm
  - **ignore zone** = 1–2 readers, < 3 mm, or a non-nodule
- **splits** → by **patient**, never by scan or nodule (PRISM: **P**).
- ⚠ pylidc is old code: it calls `np.int`, which NumPy removed in 1.24, so it needs a small compatibility shim.
- **Gate:** counts close to LUNA16's 888 scans / 1,186 nodules (for the scans we have).
