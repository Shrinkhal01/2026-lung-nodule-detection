"""Phase 1 · choose the closing radius from data (TRAIN scans only).

  python scripts/03b_mask_sweep.py --radii 10,15,20 --workers 64

Sweep set = train scans whose first-run mask was broken (< 2 L or > 10 L) or missed a GT nodule,
plus 60 random train scans. Nothing is written to the cache. Output: results/phase1/mask_sweep.md
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
from lungdet.fetch.pylidc_compat import import_pylidc  # noqa: E402
from lungdet.isolate.geometry import resample  # noqa: E402
from lungdet.isolate.lungmask import lung_mask  # noqa: E402

_pl = None


def _init():
    global _pl
    import warnings
    warnings.filterwarnings("ignore")
    _pl = import_pylidc()


def _sweep(task):
    scan_id, nods, iso, radii = task
    scan = _pl.query(_pl.Scan).filter(_pl.Scan.id == scan_id).first()
    vol = scan.to_volume(verbose=False).astype(np.float32)
    lo_hu, hi_hu = iso["clip_hu"]
    np.clip(vol, lo_hu, hi_hu, out=vol)
    full = resample(vol, (scan.pixel_spacing, scan.pixel_spacing, scan.slice_spacing), iso["spacing_mm"], cval=lo_hu)
    rows = []
    for r in radii:
        m = lung_mask(full, iso["lung_threshold_hu"], r, iso["keep_ratio"], per_slice_border=True)
        inside = [bool(m[tuple(np.clip(np.rint(np.asarray(mm)).astype(int), 0, np.array(m.shape) - 1))])
                  for mm, role in nods if role == "gt"]
        rows.append({"scan_id": scan_id, "radius": r, "lung_L": m.sum() / 1e6,
                     "lung_median_hu": float(np.median(full[m])) if m.any() else np.nan,
                     "n_gt": len(inside), "gt_inside": sum(inside)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", default="10,15,20")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--n-random", type=int, default=60)
    ap.add_argument("--summary-only", action="store_true", help="rebuild the table from mask_sweep.csv")
    args = ap.parse_args()
    cfg = load_config()
    iso, tables = cfg["isolate"], cfg["paths"]["tables"]
    radii = [float(x) for x in args.radii.split(",")]
    s = pd.read_csv(os.path.join(tables, "isolate_scans.csv"))
    n = pd.read_csv(os.path.join(tables, "isolate_nodules.csv"))
    train = s[s.split == "train"]
    missed = set(n[(n.role == "gt") & ~n.inside_mask].scan_id)
    bad = train[~train.lung_L.between(2, 10) | train.scan_id.isin(missed)]
    rand = train[~train.scan_id.isin(bad.scan_id)].sample(args.n_random, random_state=0)
    pick = pd.concat([bad.assign(group="problem"), rand.assign(group="random")])
    nods = {sid: [((r.mm_i, r.mm_j, r.mm_k), r.role) for r in g.itertuples()] for sid, g in n.groupby("scan_id")}
    tasks = [(int(sid), nods.get(sid, []), iso, radii) for sid in pick.scan_id]
    print(f"sweep set: {len(bad)} problem + {len(rand)} random TRAIN scans, radii {radii}")
    t0 = time.time()
    csv_path = os.path.join(tables, "mask_sweep.csv")
    if args.summary_only:
        res = pd.read_csv(csv_path).rename(columns={"gt": "n_gt"})
        res = res.drop(columns=["group"]).merge(pick[["scan_id", "group"]], on="scan_id")
        radii = sorted(res.radius.unique())
    else:
        rows = []
        with get_context("spawn").Pool(args.workers, initializer=_init) as pool:
            for r in pool.imap_unordered(_sweep, tasks):
                rows.extend(r)
        res = pd.DataFrame(rows).merge(pick[["scan_id", "group"]], on="scan_id")
        res.to_csv(csv_path, index=False)

    old = pick.assign(radius="old (3D border, r=10)")
    summary = [{"radius": "old (3D border, r=10)",
                "GT inside %": round(100 * n[n.scan_id.isin(pick.scan_id) & (n.role == "gt")].inside_mask.mean(), 1),
                "scans < 2 L": int((old.lung_L < 2).sum()), "scans > 10 L": int((old.lung_L > 10).sum()),
                "median L (random)": round(old[old.group == "random"].lung_L.median(), 2)}]
    for r, g in res.groupby("radius"):
        summary.append({"radius": f"new (per-slice), r={r:g}",
                        "GT inside %": round(100 * g.gt_inside.sum() / max(g.n_gt.sum(), 1), 1),
                        "scans < 2 L": int((g.lung_L < 2).sum()), "scans > 10 L": int((g.lung_L > 10).sum()),
                        "median L (random)": round(g[g.group == "random"].lung_L.median(), 2)})
    lines = ["# Phase 1 · lung-mask sweep (TRAIN scans only, `scripts/03b_mask_sweep.py`)", "",
             f"- sweep set: **{len(bad)}** problem scans + **{len(rand)}** random train scans; "
             f"**{int(res[res.radius == radii[0]].n_gt.sum())}** GT nodules; runtime **{time.time() - t0:.0f} s**", "",
             *df_to_md(pd.DataFrame(summary))]
    open(os.path.join(tables, "mask_sweep.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
