"""Phase 4 · T-E: the TEST numbers (opened once) with 95% CIs over patients, R0 / R2 / R5 (/ FINAL).

  python scripts/10b_test_report.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.discard.models import model_specs  # noqa: E402
from lungdet.rate.bootstrap import bootstrap_cpm, ci, per_scan_tables  # noqa: E402
from lungdet.rate.froc import RATES, froc  # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = load_config()
    out = os.path.join(cfg["paths"]["results"], "phase4")
    tm = pd.read_parquet(os.path.join(cfg["paths"]["patches"], "test_meta.parquet")).set_index("row")
    tables = cfg["paths"]["tables"]
    nod = pd.read_csv(os.path.join(tables, "nodules.csv"))
    nod["gid"] = np.arange(len(nod))
    scans = pd.read_csv(os.path.join(tables, "scans.csv"))
    ts = scans[(scans.split == "test") & scans.eligible]
    gt = nod[(nod.role == "gt") & nod.scan_id.isin(ts.scan_id)]
    n_gt, n_scans = len(gt), len(ts)
    gt_per_scan = gt.groupby("scan_id").size().reindex(ts.scan_id, fill_value=0).to_dict()
    specs, final_name = model_specs(cfg)

    scores = {"R0": tm.response.to_numpy()}
    for f in sorted(glob.glob(os.path.join(out, "scores_test_*.parquet"))):
        name = os.path.basename(f)[len("scores_test_"):-len(".parquet")]
        scores[name] = pd.read_parquet(f).set_index("row").prob.reindex(tm.index).to_numpy()
    res, tabs = {}, {}
    for name, s in scores.items():
        res[name] = froc(s, tm.gt_id, tm.ignore, n_gt=n_gt, n_scans=n_scans)
        tabs[name] = per_scan_tables(tm.scan_id, tm.gt_id, tm.ignore, s, gt_per_scan)
    boots = bootstrap_cpm(tabs, n_boot=1000)
    ceiling = tm[tm.gt_id >= 0].gt_id.nunique() / n_gt
    main_model = "FINAL" if "FINAL" in scores else "R5"

    rows = []
    for name in scores:
        lo, hi = ci(boots[name])
        r = res[name]
        rows.append({"model": name, "CPM": round(r["cpm"], 3), "95% CI": f"{lo:.3f} – {hi:.3f}",
                     **{f"sens@{k}": round(v, 3) for k, v in r["sens_at"].items()}})
    diff = boots[main_model] - boots["R2"]
    d_lo, d_hi = ci(diff)

    # sensitivity by size at 1 FP/scan for the main model
    s = scores[main_model]
    fp = np.sort(s[(tm.gt_id.to_numpy() < 0) & ~tm.ignore.to_numpy()])[::-1]
    thr = fp[n_scans] if len(fp) > n_scans else -np.inf          # the score where FP/scan reaches 1
    best = pd.Series(s, index=tm.index)[tm.gt_id >= 0].groupby(tm.gt_id[tm.gt_id >= 0]).max()
    g = gt.copy()
    g["found_at_1fp"] = g.gid.map(best).fillna(-np.inf) > thr
    g["candidate"] = g.gid.isin(best.index)
    g["size"] = pd.cut(g.diameter_mm, [0, 6, 10, 20, 30, 100], labels=["3-6", "6-10", "10-20", "20-30", ">30"])
    by_size = g.groupby("size", observed=False).agg(nodules=("gid", "size"), candidate=("candidate", "mean"),
                                                     found_at_1FP=("found_at_1fp", "mean")).round(3).reset_index()

    plt.figure(figsize=(6, 4.5))
    for name, r in res.items():
        fps, sens = r["curve"]
        plt.plot(np.clip(fps, 1e-3, None), sens, label=f"{name} (CPM {r['cpm']:.3f})")
    plt.xscale("log")
    plt.xlim(0.1, 16)
    plt.ylim(0, 1)
    plt.axhline(ceiling, ls="--", c="grey", lw=0.8, label=f"candidate ceiling {ceiling:.2f}")
    plt.xlabel("false positives per scan")
    plt.ylabel("sensitivity")
    plt.title(f"FROC on TEST ({n_scans} scans, {n_gt} nodules)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "froc_test.png"), dpi=150)
    for name, r in res.items():
        fps, sens = r["curve"]
        keep = np.unique(np.linspace(0, len(fps) - 1, min(500, len(fps))).astype(int))
        pd.DataFrame({"fp_per_scan": fps[keep], "sensitivity": sens[keep]}).to_csv(os.path.join(out, f"froc_test_{name}.csv"), index=False)

    md = lambda df: ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns),  # noqa: E731
                     *("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))]
    lines = ["# Phase 4 · TEST results (opened once) (`scripts/10b_test_report.py`)", "",
             f"- TEST: **{n_scans}** scans, **{n_gt}** GT nodules, **{len(tm)}** candidates; "
             f"candidate ceiling **{ceiling:.3f}** (no model can exceed it)",
             f"- main model: **{main_model}** ({final_name if main_model == 'FINAL' else 'R5 = HUSH'}); "
             "95% CIs from 1,000 bootstrap resamples of test scans", "",
             *md(pd.DataFrame(rows)), "",
             f"- **{main_model} − R2 = {res[main_model]['cpm'] - res['R2']['cpm']:+.3f}**, 95% CI **{d_lo:+.3f} to {d_hi:+.3f}** "
             f"→ {'a real difference' if d_lo > 0 or d_hi < 0 else 'the CI includes 0: NO proven difference'}", "",
             f"## {main_model}: recall by nodule size (TEST)", "", *md(by_size), "",
             "- `candidate` = found by Phase 2 at all; `found_at_1FP` = still found when the model allows 1 false alarm per scan.",
             "", "![FROC](froc_test.png)"]
    open(os.path.join(out, "test_report.md"), "w").write("\n".join(lines) + "\n")
    json.dump({n: {"cpm": res[n]["cpm"], "ci": ci(boots[n]), "sens_at": res[n]["sens_at"]} for n in res} |
              {"diff_main_minus_r2": {"value": res[main_model]["cpm"] - res["R2"]["cpm"], "ci": [d_lo, d_hi]},
               "ceiling": ceiling, "rates": list(RATES)},
              open(os.path.join(out, "test_metrics.json"), "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
