"""Phase 3b · HUSH staircase on VAL: R2 (baseline) -> R5 (+H) -> R6 (+U) -> R7 (+S), plus the channel ablation.

  python scripts/06d_summarize_hush.py [--tag _smoke]

A rung "earns its place" if its 5-seed mean CPM beats the rung below by more than the larger of the two seed SDs.
HUSH = the best rung's 5-seed ensemble (chosen on VAL -> slightly optimistic; TEST is opened once in Phase 4).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.rate.froc import froc  # noqa: E402

NAMES = {"r2": "R2 baseline 3D CNN (Phase 3)", "r5": "R5 = R2 + H (Hessian channels)",
         "r6": "R6 = R5 + U (masked-cube pretraining)", "r7": "R7 = R6 + S (size head)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config()
    res = cfg["paths"]["results"]
    pdir = cfg["paths"]["patches"] + args.tag
    out_dir = os.path.join(res, "phase3b" + args.tag)
    vm = pd.read_parquet(os.path.join(pdir, "val_meta.parquet")).set_index("row")
    tables = cfg["paths"]["tables"]
    nod = pd.read_csv(os.path.join(tables, "nodules.csv"))
    scans = pd.read_csv(os.path.join(tables, "scans.csv"))
    vs = scans[(scans.split == "val") & scans.eligible & scans.scan_id.isin(vm.scan_id)]
    n_gt = ((nod.role == "gt") & nod.scan_id.isin(vs.scan_id)).sum()
    cpm = lambda p: froc(p, vm.gt_id, vm.ignore, n_gt=n_gt, n_scans=len(vs))  # noqa: E731

    rows, prev, ens, curves = [], None, {}, {}
    for rung in ("r2", "r5", "r6", "r7"):
        d = os.path.join(res, ("phase3" if rung == "r2" else "phase3b") + args.tag, "scores")
        files = sorted(glob.glob(os.path.join(d, f"{rung}_seed*_val.parquet")))
        if not files:
            continue
        probs = [pd.read_parquet(f).set_index("row").prob.reindex(vm.index).to_numpy() for f in files]
        c = np.array([cpm(p)["cpm"] for p in probs])
        ens[rung] = np.mean(probs, axis=0)
        r = cpm(ens[rung])
        curves[rung] = r["curve"]
        verdict = "baseline"
        if prev is not None:
            delta = c.mean() - prev[0]
            need = max(prev[1], c.std())
            verdict = f"{'✅ earns its place' if delta > need else '❌ not above seed noise'} (Δ {delta:+.3f}, needs > {need:.3f})"
        rows.append({"rung": NAMES[rung], "CPM (ensemble)": round(r["cpm"], 3),
                     "seeds mean ± SD": f"{c.mean():.3f} ± {c.std():.3f} (n={len(c)})",
                     "sens@1": round(r["sens_at"][1], 3), "sens@4": round(r["sens_at"][4], 3), "verdict": verdict})
        prev = (c.mean(), c.std())
    table = pd.DataFrame(rows)
    hush = max((k for k in ens if k != "r2"), key=lambda k: cpm(ens[k])["cpm"], default=None)

    abl = []
    for rung in ("r5", "r6", "r7"):
        f = os.path.join(out_dir, "scores", f"{rung}_seed{cfg['discard']['seeds'][0]}_val.parquet")
        if os.path.exists(f):
            s = pd.read_parquet(f).set_index("row").reindex(vm.index)
            if "prob_no_blob" in s:
                abl.append({"rung (1 seed)": rung.upper(), "all 3 channels": round(cpm(s.prob)["cpm"], 3),
                            "no blobness": round(cpm(s.prob_no_blob)["cpm"], 3),
                            "no vesselness": round(cpm(s.prob_no_vessel)["cpm"], 3),
                            "HU only": round(cpm(s.prob_hu_only)["cpm"], 3)})
    for name, (fps, sens) in curves.items():          # small curve files (<= 500 points)
        keep = np.unique(np.linspace(0, len(fps) - 1, min(500, len(fps))).astype(int))
        pd.DataFrame({"fp_per_scan": fps[keep], "sensitivity": sens[keep]}).to_csv(
            os.path.join(out_dir, f"froc_{name}_val.csv"), index=False)
    md = lambda df: ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns),  # noqa: E731
                     *("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))]
    lines = [f"# Phase 3b · HUSH staircase on VAL (`scripts/06d_summarize_hush.py`{' --tag ' + args.tag if args.tag else ''})", "",
             f"- val scans **{len(vs)}**, GT nodules **{n_gt}**, candidates **{len(vm)}**", "", *md(table), ""]
    if hush:
        d = cpm(ens[hush])["cpm"] - cpm(ens["r2"])["cpm"] if "r2" in ens else float("nan")
        lines += [f"- **HUSH = {hush.upper()} ensemble: VAL CPM {cpm(ens[hush])['cpm']:.3f}** (Δ vs R2 ensemble {d:+.3f})", ""]
    if abl:
        lines += ["## Explanation by ablation: blank a physics channel at test time, measure the CPM drop", "",
                  *md(pd.DataFrame(abl)), "",
                  "- A big drop = the network **relies** on that channel. No drop = it ignores it."]
    lines += ["", "- VAL was used for choices → slightly optimistic. TEST is opened once, in Phase 4."]
    open(os.path.join(out_dir, "phase3b_report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
