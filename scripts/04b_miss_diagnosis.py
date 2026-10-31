"""Phase 2 · why are TRAIN nodules missed? (and does scale-aware bundling fix it?)

  python scripts/04b_miss_diagnosis.py

For every TRAIN ground-truth nodule that no stored candidate hits, recompute its scan's peaks and name the cause:
  no_peak    no LoG peak above the floor inside its radius at all
  hu_rule    a peak exists, but only at HU <= min_hu (faint / ground-glass)
  bundling   peaks exist, the old bundling erased them
  cap        it survives bundling but ranks below the 3,000 cap
Then: rank under scale-aware bundling (with and without the HU rule). TRAIN only.
"""
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.common.tables import df_to_md  # noqa: E402
from lungdet.nominate.blobs import bundle, log_peaks  # noqa: E402


def _rank_hit(kept, c, r):
    d = np.linalg.norm(kept[:, :3] - c, axis=1)
    idx = np.where(d <= r)[0]
    return int(idx.min()) if len(idx) else -1               # kept is sorted strongest first -> rank


def _diag(task):
    npz, nods, nom = task
    z = np.load(npz)
    hu, mask, off = z["hu"], z["mask"], z["offset"]
    allp = log_peaks(hu, mask, nom["sigmas_mm"], nom["floor"], min_hu=-1001)   # no HU rule
    huv = hu[tuple(allp[:, :3].astype(int).T)] if len(allp) else np.zeros(0)
    ok = allp[huv > nom["min_hu"]]
    old = bundle(ok, max_keep=10**9, mode="blob")
    new = bundle(ok, max_keep=10**9, mode="scale")
    new_nohu = bundle(allp, max_keep=10**9, mode="scale")
    rows = []
    for gid, mm, diam in nods:
        c = np.asarray(mm) - off
        r = max(diam / 2, 1.5)
        near_all = np.linalg.norm(allp[:, :3] - c, axis=1) <= r if len(allp) else np.zeros(0, bool)
        near_ok = np.linalg.norm(ok[:, :3] - c, axis=1) <= r if len(ok) else np.zeros(0, bool)
        rank_old = _rank_hit(old, c, r)
        if not near_all.any():
            cause = "no_peak"
        elif not near_ok.any():
            cause = "hu_rule"
        elif rank_old < 0:
            cause = "bundling"
        else:
            cause = "cap" if rank_old >= nom["max_per_scan"] else "other"
        ci = np.clip(np.rint(c).astype(int), 0, np.array(hu.shape) - 1)
        rows.append({"gid": gid, "diameter_mm": diam, "hu_centre": int(hu[tuple(ci)]), "cause": cause,
                     "rank_old": rank_old, "rank_scale": _rank_hit(new, c, r), "rank_scale_nohu": _rank_hit(new_nohu, c, r),
                     "n_old": len(old), "n_scale": len(new)})
    return rows


def main():
    cfg = load_config()
    nom = cfg["nominate"]
    tables = cfg["paths"]["tables"]
    out_dir = os.path.join(cfg["paths"]["results"], "phase2")
    cand = pd.read_parquet(os.path.join(cfg["paths"]["candidates"], "candidates_all.parquet"), columns=["gt_id", "split"])
    nod = pd.read_csv(os.path.join(tables, "nodules.csv"))
    nod["gid"] = np.arange(len(nod))
    scans = pd.read_csv(os.path.join(tables, "scans.csv")).set_index("scan_id")
    gt_tr = nod[(nod.role == "gt") & (nod.split == "train")]
    missed = gt_tr[~gt_tr.gid.isin(cand[cand.gt_id >= 0].gt_id)]
    tasks = [(os.path.join(cfg["paths"]["cache"], f"{scans.loc[sid, 'series_uid']}.npz"),
              [(r.gid, (r.mm_i, r.mm_j, r.mm_k), r.diameter_mm) for r in g.itertuples()], nom)
             for sid, g in missed.groupby("scan_id")]
    print(f"{len(missed)} missed TRAIN nodules in {len(tasks)} scans")
    t0 = time.time()
    with get_context("spawn").Pool(min(64, len(tasks))) as pool:
        rows = [r for part in pool.imap_unordered(_diag, tasks) for r in part]
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out_dir, "miss_diagnosis.csv"), index=False)
    cap = nom["max_per_scan"]
    causes = d.groupby("cause").agg(nodules=("gid", "size"), median_diam=("diameter_mm", "median"),
                                    median_hu=("hu_centre", "median")).reset_index()
    rec = lambda col: int(((d[col] >= 0) & (d[col] < cap)).sum())  # noqa: E731
    n_tr = len(gt_tr)
    base = n_tr - len(d)
    lines = ["# Phase 2 · missed TRAIN nodules (`scripts/04b_miss_diagnosis.py`)", "",
             f"- missed: **{len(d)}** of **{n_tr}** TRAIN nodules, runtime {time.time() - t0:.0f} s", "",
             "## Causes (old bundling)", "", *df_to_md(causes), "",
             "## What a fix would recover (within the 3,000 cap)", "",
             "| Change | Recovered | TRAIN recall would be |", "|---|:-:|:-:|",
             f"| scale-aware bundling | {rec('rank_scale')} | {100 * (base + rec('rank_scale')) / n_tr:.1f}% |",
             f"| scale-aware bundling + no HU rule | {rec('rank_scale_nohu')} | {100 * (base + rec('rank_scale_nohu')) / n_tr:.1f}% |",
             "", f"- candidates per scan before the cap (these scans): old median **{d.n_old.median():.0f}**, "
             f"scale-aware median **{d.n_scale.median():.0f}**"]
    open(os.path.join(out_dir, "miss_diagnosis.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
