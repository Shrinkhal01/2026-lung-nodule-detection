"""Phase 1 · test the padding fix on named scans BEFORE rerunning everything.

  python scripts/03d_padding_check.py LIDC-IDRI-0619 LIDC-IDRI-0697 LIDC-IDRI-0403 LIDC-IDRI-0523 LIDC-IDRI-0655

For each scan: the corner (padding) value, then lung volume + median HU WITHOUT and WITH the fix,
and a picture (left = without, right = with). Nothing is written to the cache.
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.fetch.pylidc_compat import import_pylidc  # noqa: E402
from lungdet.isolate.geometry import resample  # noqa: E402
from lungdet.isolate.lungmask import lung_mask  # noqa: E402
from lungdet.isolate.padding import padding_to_air  # noqa: E402


def mask_of(vol, spacing, iso):
    v = np.clip(vol, *iso["clip_hu"])
    full = resample(v, spacing, iso["spacing_mm"], cval=iso["clip_hu"][0])
    return full, lung_mask(full, iso["lung_threshold_hu"], iso["closing_radius_mm"], iso["keep_ratio"])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore")
    cfg = load_config()
    iso = cfg["isolate"]
    pl = import_pylidc()
    pids = sys.argv[1:]
    fig, axes = plt.subplots(len(pids), 2, figsize=(8, 3.6 * len(pids)))
    axes = np.atleast_2d(axes)
    print("| patient | padding value | lung L before → after | median HU before → after |\n|---|---|---|---|")
    for ax_row, pid in zip(axes, pids):
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
        vol = scan.to_volume(verbose=False).astype(np.float32)
        sp = (scan.pixel_spacing, scan.pixel_spacing, scan.slice_spacing)
        fixed, pad = padding_to_air(vol, air=iso["clip_hu"][0])
        full_a, m_a = mask_of(vol, sp, iso)
        full_b, m_b = mask_of(fixed, sp, iso)
        med = lambda f, m: int(np.median(f[m])) if m.any() else 0  # noqa: E731
        print(f"| {pid} | {pad} | {m_a.sum() / 1e6:.2f} → {m_b.sum() / 1e6:.2f} | {med(full_a, m_a)} → {med(full_b, m_b)} |")
        k = full_b.shape[2] // 2
        for ax, f, m, t in [(ax_row[0], full_a, m_a, "before"), (ax_row[1], full_b, m_b, "after fix")]:
            ax.imshow(f[:, :, k], cmap="gray", vmin=-1000, vmax=400)
            ax.contour(m[:, :, k], levels=[0.5], colors="lime", linewidths=0.7)
            ax.set_title(f"{pid}  {t}  {m.sum() / 1e6:.2f} L", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    out = os.path.join(cfg["paths"]["tables"], "padding_check.png")
    fig.savefig(out, dpi=80)
    print(f"\npicture -> {out}")


if __name__ == "__main__":
    main()
