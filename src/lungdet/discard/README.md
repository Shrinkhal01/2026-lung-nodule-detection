# discard/ (D: Phase 3)

**The sniffer dog: keeps real nodules, throws out vessels and scars.**

- **patches** → a 32 mm cube (32³ voxels at 1 mm) around every candidate, labelled nodule / not / ignore.
- **R1 XGBoost** → ~20 hand-made features per candidate (size, sphericity, HU statistics, distance to the lung wall).
- **R2 3D CNN** → a 3D ResNet-style network on the cube.
- **R3 tricks** → hard-negative mining, focal loss, 3D flips and rotations, test-time flips (each one measured).
- **R4 fusion** → stacked logistic regression on [CNN score, XGBoost score] + a 5-seed ensemble.
- **Imbalance:** about TBD false candidates per real nodule, so balanced batches + focal loss.
- **Rule:** every rung is chosen on **validation**; the test set waits for Phase 4.
