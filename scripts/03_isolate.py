"""Phase 1 · N-E-T: Normalise HU, Equalise spacing (1 mm), Trace lungs. One .npz per scan.

  python scripts/03_isolate.py --limit 5       # smoke test on 5 scans first
  python scripts/03_isolate.py                 # all eligible scans (resumable: finished scans are skipped)

Reads  results/phase1/scans.csv + nodules.csv  (from 02_ground_truth.py)
Writes $SCRATCH/cache_lungdet/<series_uid>.npz  (hu int16, mask bool, offset, spacing_in, shape_1mm)
       results/phase1/isolate_scans.csv, isolate_nodules.csv, isolate_report.md, qc_montage.png
"""
import argparse
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.common.tables import df_to_md  # noqa: E402
from lungdet.fetch.pylidc_compat import import_pylidc, write_pylidcrc  # noqa: E402
from lungdet.isolate.geometry import resample  # noqa: E402
from lungdet.isolate.lungmask import crop_box, lung_mask  # noqa: E402

_pl = None


def _init():
    global _pl
    import warnings
    warnings.filterwarnings("ignore")
    _pl = import_pylidc()


def _process(task):
    scan_id, series_uid, out_path, nods, iso, overwrite = task
    t0 = time.time()
    stat = {"scan_id": scan_id, "series_uid": series_uid, "error": ""}
    nod_rows = []
    try:
        if os.path.exists(out_path) and not overwrite:
            z = np.load(out_path)
            hu, mask, offset = z["hu"], z["mask"], z["offset"]
            stat.update(shape_1mm=tuple(z["shape_1mm"]), reused=True)
        else:
            scan = _pl.query(_pl.Scan).filter(_pl.Scan.id == scan_id).first()
            vol = scan.to_volume(verbose=False).astype(np.float32)
            spacing = (scan.pixel_spacing, scan.pixel_spacing, scan.slice_spacing)
            lo_hu, hi_hu = iso["clip_hu"]
            np.clip(vol, lo_hu, hi_hu, out=vol)                                    # N
            full = resample(vol, spacing, iso["spacing_mm"], order=1, cval=lo_hu)  # E
            full_mask = lung_mask(full, iso["lung_threshold_hu"],
                                  iso["closing_radius_mm"], iso["keep_ratio"])     # T
            lo, hi = crop_box(full_mask, int(iso["crop_margin_mm"]))
            sl = tuple(slice(a, b) for a, b in zip(lo, hi))
            hu = np.rint(full[sl]).astype(np.int16)
            mask = full_mask[sl]
            offset = lo
            np.savez_compressed(out_path, hu=hu, mask=mask, offset=offset,
                                spacing_in=np.array(spacing), shape_1mm=np.array(full.shape))
            stat.update(shape_in=tuple(vol.shape), shape_1mm=tuple(full.shape), reused=False)
        stat.update(crop_shape=tuple(hu.shape), lung_L=round(mask.sum() / 1e6, 3),
                    lung_median_hu=float(np.median(hu[mask])) if mask.any() else np.nan,
                    file_MB=round(os.path.getsize(out_path) / 2**20, 1))
        for nidx, mm, diam, role in nods:
            c = np.rint(np.asarray(mm) / iso["spacing_mm"]).astype(int) - np.asarray(offset)
            inside_box = bool(np.all(c >= 0) and np.all(c < hu.shape))
            nod_rows.append({"scan_id": scan_id, "nodule_idx": nidx, "role": role, "diameter_mm": diam,
                             "inside_mask": bool(mask[tuple(c)]) if inside_box else False,
                             "hu_at_centre": float(hu[tuple(c)]) if inside_box else np.nan})
    except Exception as e:  # noqa: BLE001
        stat["error"] = f"{type(e).__name__}: {e}"
    stat["seconds"] = round(time.time() - t0, 1)
    return stat, nod_rows


def qc_montage(stats, nodules, cache_dir, out_png, n=12, seed=0):
    """12 random scans with a GT nodule: the slice through the nodule, lung mask outline, nodule circle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gt = nodules[(nodules.role == "gt")].merge(stats[["scan_id", "series_uid"]], on="scan_id")
    pick = gt.sample(min(n, len(gt)), random_state=seed)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax, (_, r) in zip(axes.ravel(), pick.iterrows()):
        z = np.load(os.path.join(cache_dir, f"{r.series_uid}.npz"))
        c = np.rint([r.mm_i, r.mm_j, r.mm_k]).astype(int) - z["offset"]
        k = int(np.clip(c[2], 0, z["hu"].shape[2] - 1))
        ax.imshow(z["hu"][:, :, k], cmap="gray", vmin=-1000, vmax=400)
        ax.contour(z["mask"][:, :, k], levels=[0.5], colors="lime", linewidths=0.8)
        ax.add_patch(plt.Circle((c[1], c[0]), r.diameter_mm / 2 + 2, fill=False, color="red", lw=1.5))
        ax.set_title(f"{r.patient_id}  {r.diameter_mm:.0f} mm  {r.n_readers} readers", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle("Phase 1 QC: lung mask (green) and ground-truth nodule (red)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only the first N eligible scans")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    iso, tables = cfg["isolate"], cfg["paths"]["tables"]
    cache = cfg["paths"]["cache"]
    os.makedirs(cache, exist_ok=True)
    write_pylidcrc(cfg["paths"]["lidc_dicom"])

    scans = pd.read_csv(os.path.join(tables, "scans.csv"))
    nodules = pd.read_csv(os.path.join(tables, "nodules.csv"))
    el = scans[scans.eligible].sort_values("scan_id")
    if args.limit:
        el = el.head(args.limit)
    by_scan = {sid: [(r.nodule_idx, (r.mm_i, r.mm_j, r.mm_k), r.diameter_mm, r.role) for r in g.itertuples()]
               for sid, g in nodules.groupby("scan_id")}
    tasks = [(int(s.scan_id), s.series_uid, os.path.join(cache, f"{s.series_uid}.npz"),
              by_scan.get(s.scan_id, []), iso, args.overwrite) for s in el.itertuples()]

    t0 = time.time()
    stats, nod_rows = [], []
    with get_context("spawn").Pool(args.workers or iso["workers"], initializer=_init) as pool:
        for i, (st, nr) in enumerate(pool.imap_unordered(_process, tasks), 1):
            stats.append(st)
            nod_rows.extend(nr)
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} scans, {time.time() - t0:.0f} s")
    stats = pd.DataFrame(stats).merge(el[["scan_id", "patient_id", "split"]], on="scan_id")
    nstat = pd.DataFrame(nod_rows, columns=["scan_id", "nodule_idx", "role", "diameter_mm",
                                            "inside_mask", "hu_at_centre"])
    nstat = nstat.merge(nodules[["scan_id", "nodule_idx", "patient_id", "n_readers", "mm_i", "mm_j", "mm_k"]],
                        on=["scan_id", "nodule_idx"])
    suffix = f"_limit{args.limit}" if args.limit else ""
    stats.to_csv(os.path.join(tables, f"isolate_scans{suffix}.csv"), index=False)
    nstat.to_csv(os.path.join(tables, f"isolate_nodules{suffix}.csv"), index=False)

    ok = stats[stats.error == ""]
    gt = nstat[nstat.role == "gt"]
    vol_ok = ok.lung_L.between(2, 8).mean() if len(ok) else 0
    hu_ok = ok.lung_median_hu.between(-950, -700).mean() if len(ok) else 0
    inside = gt.inside_mask.mean() if len(gt) else float("nan")
    hu_c = gt.hu_at_centre.median() if len(gt) else float("nan")
    gates = [
        ("0 scans failed", (stats.error != "").sum() == 0, f"{(stats.error != '').sum()} of {len(stats)}"),
        ("lung volume 2–8 L in ≥ 99% of scans", vol_ok >= 0.99, f"{100 * vol_ok:.1f}%"),
        ("lung median HU in [−950, −700] in ≥ 99% of scans", hu_ok >= 0.99, f"{100 * hu_ok:.1f}%"),
        ("GT nodule centres inside the lung mask ≥ 98%", inside >= 0.98, f"{100 * inside:.1f}% of {len(gt)}"),
        ("median HU at GT centres > −400 (coordinates line up)", hu_c > -400, f"{hu_c:.0f} HU"),
    ]
    lines = [
        f"# Phase 1 · isolate report (`scripts/03_isolate.py`{' --limit ' + str(args.limit) if args.limit else ''})", "",
        f"- scans processed: **{len(stats)}**, failed: **{(stats.error != '').sum()}**, runtime **{time.time() - t0:.0f} s**",
        f"- lung volume (L): median **{ok.lung_L.median():.2f}**, min **{ok.lung_L.min():.2f}**, max **{ok.lung_L.max():.2f}**",
        f"- lung median HU: median **{ok.lung_median_hu.median():.0f}**",
        f"- cache: **{ok.file_MB.sum() / 1024:.1f} GiB** total, **{ok.file_MB.median():.1f} MB** per scan (median)",
        f"- GT nodules outside the mask: **{int((~gt.inside_mask).sum())}** "
        f"(their diameters: {sorted(gt.loc[~gt.inside_mask, 'diameter_mm'].round(1).tolist())[:15]})",
        "", "## Gates", "", "| Gate | Result | Value |", "|---|:-:|---|",
        *[f"| {g} | {'PASS' if okk else 'FAIL'} | {v} |" for g, okk, v in gates],
    ]
    bad = stats[(stats.error != "") | ~stats.lung_L.between(2, 8)]
    if len(bad):
        lines += ["", "## Scans to look at", "",
                  *df_to_md(bad[["scan_id", "patient_id", "lung_L", "lung_median_hu", "error"]].head(20))]
    open(os.path.join(tables, f"isolate_report{suffix}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if len(gt):
        qc_montage(ok, nstat, cache, os.path.join(tables, f"qc_montage{suffix}.png"))
        print(f"\nQC picture -> {os.path.join(tables, f'qc_montage{suffix}.png')}")


if __name__ == "__main__":
    main()
