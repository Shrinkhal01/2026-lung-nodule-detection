"""Phase 5 · pick the probability threshold for each false-alarms-per-scan setting, on VAL only.

  python scripts/12_operating_points.py        # CPU, seconds. Runs on the MAC: it only reads small files
Reads  val_meta.parquet (copied once from $SCRATCH/patches/) + results/phase3b/scores/r5_seed{42..46}_val.parquet
       + results/phase1/{scans,nodules}.csv   (all already on the Mac except val_meta)
Writes results/phase5/operating_points.json   (commit it)
Paths can be overridden: --val-meta --scores-dir --tables --out
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.ship.operating import operating_points  # noqa: E402


def main():
    import argparse
    root = os.path.join(os.path.dirname(__file__), "..")
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-meta", default=os.path.join(root, "data", "phase5_local", "val_meta.parquet"))
    ap.add_argument("--scores-dir", default=os.path.join(root, "results", "phase3b", "scores"))
    ap.add_argument("--tables", default=os.path.join(root, "results", "phase1"))
    ap.add_argument("--out", default=os.path.join(root, "results", "phase5"))
    a = ap.parse_args()
    cfg = load_config()
    meta = pd.read_parquet(a.val_meta).set_index("row")
    seeds = cfg["ship"]["seeds"]
    p = np.mean([pd.read_parquet(os.path.join(a.scores_dir, f"r5_seed{s}_val.parquet")).set_index("row").prob
                 .reindex(meta.index).to_numpy() for s in seeds], axis=0)
    assert not np.isnan(p).any(), "score rows do not match val_meta rows"
    scans = pd.read_csv(os.path.join(a.tables, "scans.csv"))
    n_scans = int(((scans.split == "val") & scans.eligible).sum())
    nod = pd.read_csv(os.path.join(a.tables, "nodules.csv"))
    sc_val = scans[(scans.split == "val") & scans.eligible].scan_id
    n_gt = int(((nod.role == "gt") & nod.scan_id.isin(sc_val)).sum())
    pts = operating_points(p, meta.gt_id.to_numpy(), meta.ignore.to_numpy(), n_gt, n_scans)
    out = {"model": "R5 (HUSH) 5-seed mean", "split": "val", "n_scans": n_scans, "n_gt": n_gt,
           "rule": "keep prob > threshold; threshold = the (floor(rate*n_scans)+1)-th strongest false candidate",
           "points": {(str(int(r)) if float(r).is_integer() else str(r)): {"threshold": round(t, 6), "val_sensitivity": round(s, 4),
                                                                          "val_fp_per_scan": round(f, 4)}
                      for r, (t, s, f) in pts.items()}}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "operating_points.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
