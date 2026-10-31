"""Phase 3 · R1: XGBoost on hand-made candidate features (CPU). Scores ALL val candidates.

  python scripts/06b_xgb.py [--tag _smoke]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.discard.features import FEATURES, cube_features  # noqa: E402
from lungdet.rate.froc import froc  # noqa: E402

DETECTOR = ["sigma", "response", "log_scale_rank", "log_global_rank"]


def table(meta, cubes):
    X = pd.DataFrame(cube_features(cubes, meta.sigma.to_numpy()), columns=FEATURES)
    X["sigma"], X["response"] = meta.sigma.to_numpy(), meta.response.to_numpy()
    X["log_scale_rank"], X["log_global_rank"] = np.log1p(meta.scale_rank.to_numpy()), np.log1p(meta.global_rank.to_numpy())
    return X


def main():
    import xgboost as xgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config()
    pdir = cfg["paths"]["patches"] + args.tag
    out_dir = os.path.join(cfg["paths"]["results"], "phase3" + args.tag)
    os.makedirs(os.path.join(out_dir, "scores"), exist_ok=True)
    t0 = time.time()
    m = pd.read_parquet(os.path.join(pdir, "train_meta.parquet"))
    m = m[m.kind != "gt_jitter"]                       # jittered GT centres have no detector numbers
    cubes = np.load(os.path.join(pdir, "train_cubes.npy"), mmap_mode="r")
    X = table(m, cubes[m.row.to_numpy()])
    y = m.label.to_numpy()
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)), tree_method="hist",
                            n_jobs=32, random_state=42)
    clf.fit(X, y)
    clf.save_model(os.path.join(cfg["paths"]["checkpoints"], f"r1_xgb{args.tag}.json"))

    vm = pd.read_parquet(os.path.join(pdir, "val_meta.parquet"))
    vc = np.load(os.path.join(pdir, "val_cubes.npy"), mmap_mode="r")
    prob = np.concatenate([clf.predict_proba(table(vm.iloc[s:s + 50000], vc[s:s + 50000]))[:, 1]
                           for s in range(0, len(vm), 50000)])
    pd.DataFrame({"row": vm.row, "prob": prob}).to_parquet(os.path.join(out_dir, "scores", "r1_val.parquet"))
    tables = cfg["paths"]["tables"]
    nod = pd.read_csv(os.path.join(tables, "nodules.csv"))
    scans = pd.read_csv(os.path.join(tables, "scans.csv"))
    vs = scans[(scans.split == "val") & scans.eligible & scans.scan_id.isin(vm.scan_id)]
    r = froc(prob, vm.gt_id, vm.ignore, n_gt=((nod.role == "gt") & nod.scan_id.isin(vs.scan_id)).sum(), n_scans=len(vs))
    imp = dict(sorted(zip(X.columns, clf.feature_importances_.round(3).tolist()), key=lambda kv: -kv[1]))
    json.dump({"rung": "r1", "val_cpm": r["cpm"], "sens_at": r["sens_at"], "importance": imp,
               "seconds": round(time.time() - t0)}, open(os.path.join(out_dir, "scores", "r1.json"), "w"), indent=2)
    print(f"R1 XGBoost: VAL CPM {r['cpm']:.3f}, top features {list(imp)[:4]}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
