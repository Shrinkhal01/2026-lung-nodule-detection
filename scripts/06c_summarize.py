"""Phase 3 · the staircase on VAL: R0 -> R1 -> R2 -> R3 -> R4 (fusion + 5-seed ensemble).

  python scripts/06c_summarize.py [--tag _smoke]

R4 fusion = logistic regression on [logit(best CNN ensemble), logit(R1)], CROSS-FITTED on val (5 folds by patient),
so no candidate is scored by a fusion model that saw it. TEST is not touched.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.rate.froc import froc  # noqa: E402


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config()
    pdir = cfg["paths"]["patches"] + args.tag
    out_dir = os.path.join(cfg["paths"]["results"], "phase3" + args.tag)
    sdir = os.path.join(out_dir, "scores")
    vm = pd.read_parquet(os.path.join(pdir, "val_meta.parquet")).set_index("row")
    tables = cfg["paths"]["tables"]
    nod = pd.read_csv(os.path.join(tables, "nodules.csv"))
    scans = pd.read_csv(os.path.join(tables, "scans.csv"))
    vs = scans[(scans.split == "val") & scans.eligible & scans.scan_id.isin(vm.scan_id)]
    n_gt = ((nod.role == "gt") & nod.scan_id.isin(vs.scan_id)).sum()
    cpm = lambda p: froc(p, vm.gt_id, vm.ignore, n_gt=n_gt, n_scans=len(vs))  # noqa: E731
    load = lambda f: pd.read_parquet(f).set_index("row").prob.reindex(vm.index).to_numpy()  # noqa: E731

    rows, curves = [], {}

    def add(name, p, note, spread=""):
        r = cpm(p)
        curves[name] = r["curve"]
        rows.append({"rung": name, "CPM": round(r["cpm"], 3), "seeds (mean ± SD)": spread,
                     **{f"sens@{k}": round(v, 3) for k, v in r["sens_at"].items() if k in (0.5, 1, 2, 4)}, "what": note})

    add("R0", vm.response.to_numpy(), "blob strength only, no learning")
    if os.path.exists(os.path.join(sdir, "r1_val.parquet")):
        add("R1", load(os.path.join(sdir, "r1_val.parquet")), "XGBoost on hand-made features")
    ens = {}
    for rung in ("r2", "r3"):
        files = sorted(glob.glob(os.path.join(sdir, f"{rung}_seed*_val.parquet")))
        if not files:
            continue
        probs = [load(f) for f in files]
        c = [cpm(p)["cpm"] for p in probs]
        ens[rung] = np.mean(probs, axis=0)
        add(rung.upper(), ens[rung], {"r2": "3D CNN, BCE, random negatives",
                                      "r3": "3D CNN + focal loss + hard negatives + TTA"}[rung] + f", {len(files)}-seed ensemble",
            f"{np.mean(c):.3f} ± {np.std(c):.3f} (n={len(c)})")
    if ens and os.path.exists(os.path.join(sdir, "r1_val.parquet")):
        best = max(ens, key=lambda k: cpm(ens[k])["cpm"])          # fuse the BEST CNN rung (chosen on val)
        X = np.column_stack([logit(ens[best]), logit(load(os.path.join(sdir, "r1_val.parquet")))])
        y = (vm.gt_id >= 0).to_numpy().astype(int)
        fit_ok = ~vm.ignore.to_numpy()
        fused = np.zeros(len(vm))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups=vm.patient_id):
            tr = tr[fit_ok[tr]]
            fused[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        add("R4", fused, f"fusion: logistic regression on [{best.upper()} ensemble, R1], cross-fitted by patient")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "staircase_val.csv"), index=False)
    for name, (fps, sens) in curves.items():
        pd.DataFrame({"fp_per_scan": fps, "sensitivity": sens}).to_csv(os.path.join(out_dir, f"froc_{name}_val.csv"), index=False)
    head = "| " + " | ".join(res.columns) + " |"
    lines = [f"# Phase 3 · staircase on VAL (`scripts/06c_summarize.py`{' --tag ' + args.tag if args.tag else ''})", "",
             f"- val scans **{len(vs)}**, GT nodules **{n_gt}**, candidates **{len(vm)}** "
             f"(ceiling = candidate recall; every sensitivity is capped by it)", "",
             head, "|" + "---|" * len(res.columns),
             *("| " + " | ".join(map(str, r)) + " |" for r in res.itertuples(index=False)), "",
             "- Each rung must beat the one below it on VAL to earn its place. TEST is opened once, in Phase 4."]
    open(os.path.join(out_dir, "phase3_report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
