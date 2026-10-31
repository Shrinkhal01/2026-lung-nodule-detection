"""Phase 5 · pick the probability threshold for each false-alarms-per-scan setting, on VAL only.

  CUDA_VISIBLE_DEVICES= python scripts/12_operating_points.py        # CPU is enough, it only reads score files
Reads  $PATCHES/val_meta.parquet + $RESULTS/phase3b/scores/r5_seed{42..46}_val.parquet  (the R5 ensemble)
Writes $RESULTS/phase5/operating_points.json   (copy to the Mac with tools/pull_results.sh, then commit)
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
    cfg = load_config()
    res = cfg["paths"]["results"]
    meta = pd.read_parquet(os.path.join(cfg["paths"]["patches"], "val_meta.parquet")).set_index("row")
    seeds = cfg["ship"]["seeds"]
    p = np.mean([pd.read_parquet(os.path.join(res, "phase3b", "scores", f"r5_seed{s}_val.parquet")).set_index("row").prob
                 .reindex(meta.index).to_numpy() for s in seeds], axis=0)
    assert not np.isnan(p).any(), "score rows do not match val_meta rows"
    scans = pd.read_csv(os.path.join(cfg["paths"]["tables"], "scans.csv"))
    n_scans = int(((scans.split == "val") & scans.eligible).sum())
    nod = pd.read_csv(os.path.join(cfg["paths"]["tables"], "nodules.csv"))
    sc_val = scans[(scans.split == "val") & scans.eligible].scan_id
    n_gt = int(((nod.role == "gt") & nod.scan_id.isin(sc_val)).sum())
    pts = operating_points(p, meta.gt_id.to_numpy(), meta.ignore.to_numpy(), n_gt, n_scans)
    out = {"model": "R5 (HUSH) 5-seed mean", "split": "val", "n_scans": n_scans, "n_gt": n_gt,
           "rule": "keep prob > threshold; threshold = the (floor(rate*n_scans)+1)-th strongest false candidate",
           "points": {(str(int(r)) if float(r).is_integer() else str(r)): {"threshold": round(t, 6), "val_sensitivity": round(s, 4),
                                                                          "val_fp_per_scan": round(f, 4)}
                      for r, (t, s, f) in pts.items()}}
    os.makedirs(os.path.join(res, "phase5"), exist_ok=True)
    json.dump(out, open(os.path.join(res, "phase5", "operating_points.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
