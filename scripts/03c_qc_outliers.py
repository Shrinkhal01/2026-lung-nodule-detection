"""Phase 1 · look at the scans that fall outside the sanity ranges (lung 2-10 L, median HU -950..-700).

  python scripts/03c_qc_outliers.py

For each outlier: numbers that separate "real anatomy" from "broken mask", plus an axial and a
coronal picture with the mask outline. Output: qc_outliers.md + qc_outliers.png in the results folder.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.common.tables import df_to_md  # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = load_config()
    tables, cache = cfg["paths"]["tables"], cfg["paths"]["cache"]
    s = pd.read_csv(os.path.join(tables, "isolate_scans.csv"))
    out = s[~s.lung_L.between(2, 10) | ~s.lung_median_hu.between(-950, -700)].sort_values("lung_L")
    rows = []
    fig, axes = plt.subplots(len(out), 2, figsize=(8, 3.2 * len(out)))
    axes = np.atleast_2d(axes)
    for ax_row, r in zip(axes, out.itertuples()):
        z = np.load(os.path.join(cache, f"{r.series_uid}.npz"))
        hu, mask, off, full = z["hu"], z["mask"], z["offset"], z["shape_1mm"]
        lab, n = ndimage.label(mask)
        sizes = sorted(np.bincount(lab.ravel())[1:], reverse=True)
        lo = off + np.argwhere(mask).min(axis=0)
        hi = off + np.argwhere(mask).max(axis=0)
        edge = int(min(lo[0], lo[1], full[0] - 1 - hi[0], full[1] - 1 - hi[1]))
        rows.append({"patient": r.patient_id, "lung_L": round(r.lung_L, 2), "median_HU": int(r.lung_median_hu),
                     "blobs": n, "2nd/1st blob": round(sizes[1] / sizes[0], 2) if len(sizes) > 1 else 0,
                     "lung z-extent mm": int(hi[2] - lo[2] + 1), "scan z mm": int(full[2]),
                     "mm to image edge": edge})
        ci, _, ck = np.round(ndimage.center_of_mass(mask)).astype(int)
        ax_row[0].imshow(hu[:, :, ck], cmap="gray", vmin=-1000, vmax=400)
        ax_row[0].contour(mask[:, :, ck], levels=[0.5], colors="lime", linewidths=0.7)
        ax_row[0].set_title(f"{r.patient_id}  {r.lung_L:.2f} L  axial", fontsize=8)
        ax_row[1].imshow(np.rot90(hu[ci, :, :]), cmap="gray", vmin=-1000, vmax=400, aspect="auto")
        ax_row[1].contour(np.rot90(mask[ci, :, :]), levels=[0.5], colors="lime", linewidths=0.7)
        ax_row[1].set_title("coronal (head at top)", fontsize=8)
        for a in ax_row:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(tables, "qc_outliers.png"), dpi=80)
    lines = ["# Phase 1 · sanity-range outliers (`scripts/03c_qc_outliers.py`)", "",
             f"{len(out)} of {len(s)} scans are outside lung 2–10 L or median HU −950..−700.", "",
             *df_to_md(pd.DataFrame(rows)), "",
             "- **mm to image edge ≈ 0** → the mask reaches the image border (possible leak or a tight field of view).",
             "- **lung z-extent ≪ scan z** → the scan does not cover the whole chest, or part of the lung was lost.",
             "- **2nd/1st blob ≈ 0** → only one lung blob kept (lungs joined, or one lung missing)."]
    open(os.path.join(tables, "qc_outliers.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
