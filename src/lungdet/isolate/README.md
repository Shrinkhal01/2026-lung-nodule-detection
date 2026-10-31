# isolate/ (I: Phase 1)

**Puts every scan on the same ruler and cuts out the lungs.**

- **HU** = stored value × slope + intercept. Example: 224 × 1 − 1024 = **−800 HU** (lung).
- **Resample to 1 mm:**
  - 512 × 512 × 130 voxels at 0.7 × 0.7 × 2.5 mm → **358 × 358 × 325** voxels at 1 mm
  - because a 3D filter must see the **same mm per voxel** in every scan
- **Lung mask (classical):** threshold (start −320 HU) → keep the 2 largest air blobs inside the body → close holes.
  - Closing the holes keeps nodules **attached to the lung wall** inside the mask.
- 🖼 **Ironing every scan to the same paper size, then cutting out the lungs with scissors.**
- **Gate:** air ≈ −1000 HU; the mask covers both lungs in a visual check of 10 random scans.
