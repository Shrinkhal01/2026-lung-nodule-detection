# nominate/ (N: Phase 2)

**The metal detector: beeps at anything that could be a nodule.**

- Inside the lung mask: voxels denser than a threshold → connected blobs → keep blobs of **3–30 mm**.
- Optional: a multi-scale blob detector (Laplacian of Gaussian) for small or faint nodules.
- **Recall over precision:** a nodule missed here is lost forever. False alarms are fine; step D removes them.
- Thresholds are tuned on **train** scans only.
- **R0 baseline:** rank candidates by a hand rule (size × density) → the CPM with **no learning**.
- **Gate:** recall on validation (target ≥ 90%, TBD) and candidates per scan (TBD).
