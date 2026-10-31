"""Phase 2 · N (Nominate): LoG candidates for every scan, threshold chosen on TRAIN, R0 baseline on VAL.

  python scripts/04_nominate.py --limit 5     # smoke test
  python scripts/04_nominate.py               # all 897 scans

Writes $SCRATCH/candidates/candidates_all.parquet  (every stored peak, labelled)
       $SCRATCH/candidates/candidates.parquet      (after the chosen threshold; input for Phase 3)
       results: nominate_sweep.csv, nominate_report.md, froc_r0_val.csv
The TEST split is not used for any choice and its recall is NOT reported here.
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
from lungdet.nominate.blobs import bundle, log_peaks  # noqa: E402
from lungdet.rate.froc import froc  # noqa: E402
from lungdet.rate.matching import match  # noqa: E402

BROKEN_MASKS = ["LIDC-IDRI-0619", "LIDC-IDRI-0697", "LIDC-IDRI-0403", "LIDC-IDRI-0523", "LIDC-IDRI-0655"]


def _scan(task):
    scan_id, npz_path, nods, nom = task
    t0 = time.time()
    z = np.load(npz_path)
    hu, mask, off = z["hu"], z["mask"], z["offset"]
    peaks = log_peaks(hu, mask, nom["sigmas_mm"], nom["floor"], nom["min_hu"])
    n_raw = len(peaks)
    c = bundle(peaks, nom["max_per_scan"])
    xyz = c[:, :3] + off                                     # full 1 mm grid = mm
    nod_xyz = [n[1] for n in nods]
    hit, ign = match(xyz, nod_xyz, [n[2] for n in nods], [n[3] == "gt" for n in nods])
    gt_ids = np.array([n[0] for n in nods] + [-1])           # global id of the GT nodule hit (or -1)
    df = pd.DataFrame({"scan_id": scan_id, "i": xyz[:, 0].astype(np.int16), "j": xyz[:, 1].astype(np.int16),
                       "k": xyz[:, 2].astype(np.int16), "sigma": c[:, 3].astype(np.float32),
                       "response": c[:, 4].astype(np.float32),
                       "gt_id": gt_ids[np.where(hit >= 0, hit, -1)], "ignore": ign})
    return df, {"scan_id": scan_id, "n_raw_peaks": n_raw, "n_kept": len(c), "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    nom, tables = cfg["nominate"], cfg["paths"]["tables"]
    out_dir = os.path.join(cfg["paths"]["results"], "phase2")
    cand_dir = cfg["paths"]["candidates"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cand_dir, exist_ok=True)

    scans = pd.read_csv(os.path.join(tables, "scans.csv"))
    scans = scans[scans.eligible].sort_values("scan_id")
    if args.limit:
        scans = scans.head(args.limit)
    nod = pd.read_csv(os.path.join(tables, "nodules.csv"))
    nod["gid"] = np.arange(len(nod))                         # one global id per nodule (GT or ignore)
    by_scan = {sid: [(r.gid if r.role == "gt" else -1, (r.mm_i, r.mm_j, r.mm_k), r.diameter_mm, r.role)
                     for r in g.itertuples()] for sid, g in nod.groupby("scan_id")}
    tasks = [(int(s.scan_id), os.path.join(cfg["paths"]["cache"], f"{s.series_uid}.npz"), by_scan.get(s.scan_id, []), nom)
             for s in scans.itertuples()]

    t0 = time.time()
    parts, stats = [], []
    with get_context("spawn").Pool(args.workers or nom["workers"]) as pool:
        for n, (df, st) in enumerate(pool.imap_unordered(_scan, tasks), 1):
            parts.append(df)
            stats.append(st)
            if n % 100 == 0 or n == len(tasks):
                print(f"  {n}/{len(tasks)} scans, {time.time() - t0:.0f} s")
    cand = pd.concat(parts, ignore_index=True).merge(scans[["scan_id", "patient_id", "split"]], on="scan_id")
    stats = pd.DataFrame(stats).merge(scans[["scan_id", "patient_id", "split"]], on="scan_id")
    suffix = f"_limit{args.limit}" if args.limit else ""
    cand.to_parquet(os.path.join(cand_dir, f"candidates_all{suffix}.parquet"))

    gt = nod[(nod.role == "gt") & nod.scan_id.isin(scans.scan_id)]       # nodules.csv already has the split

    def recall_at(split, t):
        c = cand[(cand.split == split) & (cand.response >= t) & (cand.gt_id >= 0)]
        g = gt[gt.split == split]
        return c.gt_id.nunique() / max(len(g), 1), (cand[(cand.split == split) & (cand.response >= t)]
                                                    .groupby("scan_id").size().reindex(
                                                        scans[scans.split == split].scan_id, fill_value=0).mean())

    # sweep the threshold on TRAIN only
    ts = np.unique(np.quantile(cand.response, np.linspace(0, 0.99, 100)))
    sweep = pd.DataFrame([{"threshold": round(float(t), 1), "train_recall": r, "train_cands_per_scan": c}
                          for t in ts for r, c in [recall_at("train", t)]])
    sweep.to_csv(os.path.join(out_dir, f"nominate_sweep{suffix}.csv"), index=False)
    ok = sweep[sweep.train_recall >= nom["recall_target"]]
    chosen = ok.sort_values("train_cands_per_scan").iloc[0] if len(ok) else sweep.sort_values("train_recall").iloc[-1]
    t_star = float(chosen.threshold)
    val_recall, val_cps = recall_at("val", t_star)
    max_train, _ = recall_at("train", -np.inf)

    final = cand[cand.response >= t_star]
    final.to_parquet(os.path.join(cand_dir, f"candidates{suffix}.parquet"))

    # recall by size (TRAIN), and missed nodules
    g_tr = gt[gt.split == "train"].copy()
    g_tr["found"] = g_tr.gid.isin(final[final.split == "train"].gt_id)
    g_tr["size"] = pd.cut(g_tr.diameter_mm, [0, 6, 10, 20, 30, 100], labels=["3-6", "6-10", "10-20", "20-30", ">30"])
    by_size = g_tr.groupby("size", observed=False).agg(nodules=("gid", "size"), recall=("found", "mean")).reset_index()
    by_size["recall"] = by_size.recall.round(3)

    # R0: rank by blob strength alone, FROC on VAL
    v = final[final.split == "val"]
    n_val_scans = (scans.split == "val").sum()
    r0 = froc(v.response, v.gt_id, v.ignore, n_gt=(gt.split == "val").sum(), n_scans=n_val_scans)
    pd.DataFrame({"fp_per_scan": r0["curve"][0], "sensitivity": r0["curve"][1]}).to_csv(
        os.path.join(out_dir, f"froc_r0_val{suffix}.csv"), index=False)

    per_scan = final.groupby("scan_id").size()
    broken = stats[stats.patient_id.isin(BROKEN_MASKS)][["scan_id", "patient_id"]].copy()
    broken["candidates"] = broken.scan_id.map(per_scan).fillna(0).astype(int)
    lines = [
        f"# Phase 2 · nominate report (`scripts/04_nominate.py`{' --limit ' + str(args.limit) if args.limit else ''})", "",
        f"- scans: **{len(scans)}**, runtime **{time.time() - t0:.0f} s**, median **{stats.seconds.median():.0f} s**/scan per worker",
        f"- stored peaks per scan before the threshold: median **{stats.n_kept.median():.0f}** "
        f"(raw scale-space peaks: median {stats.n_raw_peaks.median():.0f})",
        f"- best possible TRAIN recall (no threshold): **{100 * max_train:.1f}%**",
        f"- **chosen threshold {t_star:.1f}** (fewest candidates with TRAIN recall ≥ {100 * nom['recall_target']:.0f}%):",
        f"  - TRAIN recall **{100 * chosen.train_recall:.1f}%** at **{chosen.train_cands_per_scan:.0f}** candidates/scan",
        f"  - VAL recall **{100 * val_recall:.1f}%** at **{val_cps:.0f}** candidates/scan (check, not tuned)",
        f"- candidates per scan after the threshold: median **{per_scan.median():.0f}**, 90th **{per_scan.quantile(.9):.0f}**",
        f"- broken-mask scans: {', '.join(f'{r.patient_id[-4:]}={r.candidates}' for r in broken.itertuples()) or 'none in this run'}",
        "", "## TRAIN recall by nodule size", "", *df_to_md(by_size), "",
        "## R0 baseline (rank by blob strength, no learning), VAL", "",
        "| FP/scan | " + " | ".join(str(k) for k in r0["sens_at"]) + " | **CPM** |",
        "|---|" + "---|" * (len(r0["sens_at"]) + 1),
        "| sensitivity | " + " | ".join(f"{s:.3f}" for s in r0["sens_at"].values()) + f" | **{r0['cpm']:.3f}** |",
        "", "TEST split: not used, not reported (opened once in Phase 4).",
    ]
    open(os.path.join(out_dir, f"nominate_report{suffix}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
