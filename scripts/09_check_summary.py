"""Length + fit check: does 3x longer training help? Does the model under- or over-fit?

  CUDA_VISIBLE_DEVICES=1 python scripts/09_check_summary.py

Compares, on the SAME seeds (42-44), the 10-epoch models (Phase 3 / 3b) with the 30-epoch "_long" models.
Fit = cube-level ROC AUC on TRAIN candidates (positives + random negatives) vs on VAL candidates.
Writes results/check/check_report.md + final_choice.json (the model Phase 4 will test ONCE).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.discard.cnn import Small3DNet, predict  # noqa: E402
from lungdet.discard.hessian import hessian_channels  # noqa: E402
from lungdet.rate.froc import froc  # noqa: E402

SEEDS = [42, 43, 44]


def main():
    cfg = load_config()
    res, ck = cfg["paths"]["results"], cfg["paths"]["checkpoints"]
    pdir = cfg["paths"]["patches"]
    out_dir = os.path.join(res, "check")
    os.makedirs(out_dir, exist_ok=True)
    vm = pd.read_parquet(os.path.join(pdir, "val_meta.parquet")).set_index("row")
    tables = cfg["paths"]["tables"]
    nod = pd.read_csv(os.path.join(tables, "nodules.csv"))
    scans = pd.read_csv(os.path.join(tables, "scans.csv"))
    vs = scans[(scans.split == "val") & scans.eligible & scans.scan_id.isin(vm.scan_id)]
    n_gt = ((nod.role == "gt") & nod.scan_id.isin(vs.scan_id)).sum()
    cpm = lambda p: froc(p, vm.gt_id, vm.ignore, n_gt=n_gt, n_scans=len(vs))["cpm"]  # noqa: E731
    lab, keep = (vm.gt_id >= 0).to_numpy(), ~vm.ignore.to_numpy()

    tm = pd.read_parquet(os.path.join(pdir, "train_meta.parquet"))
    tm = tm[tm.kind.isin(["cand_pos", "neg_random"])]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcubes = torch.from_numpy(np.load(os.path.join(pdir, "train_cubes.npy"), mmap_mode="r")[tm.row.to_numpy()]).to(dev)
    sig = tuple(cfg["hush"]["hessian_sigmas_mm"])

    variants = [("r2", "", "phase3", 10), ("r5", "", "phase3b", 10),
                ("r2", "_long", "check", 30), ("r5", "_long", "check", 30)]
    rows, ens, curves = [], {}, []
    for rung, suf, folder, ep in variants:
        probs, cpms, tr_auc, va_auc = [], [], [], []
        for s in SEEDS:
            f = os.path.join(res, folder, "scores", f"{rung}{suf}_seed{s}_val.parquet")
            if not os.path.exists(f):
                continue
            p = pd.read_parquet(f).set_index("row").prob.reindex(vm.index).to_numpy()
            probs.append(p)
            cpms.append(cpm(p))
            va_auc.append(roc_auc_score(lab[keep], p[keep]))
            js = os.path.join(res, folder, "scores", f"{rung}{suf}_seed{s}.json")
            j = json.load(open(js)) if os.path.exists(js) else {}
            if "train_auc" in j:
                tr_auc.append(j["train_auc"])
                curves += [{"model": f"{rung}{suf}", "seed": s, **c} for c in j.get("curve", [])]
            else:                                   # old 10-epoch run: score its TRAIN cubes now
                in_ch = 3 if rung == "r5" else 1
                model = Small3DNet(in_ch).to(dev)
                model.load_state_dict(torch.load(os.path.join(ck, f"{rung}{suf}_seed{s}.pth"), map_location=dev))
                prep = (lambda x: hessian_channels(x, sig)) if in_ch == 3 else None
                pt = predict(model, tcubes, batch=2048, transform=prep)
                tr_auc.append(roc_auc_score(tm.label.to_numpy(), pt))
        if not probs:
            continue
        ens[f"{rung}{suf}"] = np.mean(probs, axis=0)
        rows.append({"model": f"{rung.upper()}{' long' if suf else ''}", "epochs": ep,
                     "VAL CPM (3 seeds)": f"{np.mean(cpms):.3f} ± {np.std(cpms):.3f}",
                     "ensemble CPM": round(cpm(ens[f'{rung}{suf}']), 3),
                     "TRAIN AUC": round(float(np.mean(tr_auc)), 4), "VAL AUC": round(float(np.mean(va_auc)), 4),
                     "gap": round(float(np.mean(tr_auc) - np.mean(va_auc)), 4), "_mean": np.mean(cpms), "_sd": np.std(cpms)})
    t = pd.DataFrame(rows)

    # decision: switch to a long model only if it beats its own 10-epoch version by more than the seed noise
    choice = {"rung": "r5", "suffix": "", "seeds": [42, 43, 44, 45, 46], "why": "default: HUSH R5 (10 epochs)"}
    notes = []
    for rung in ("r2", "r5"):
        b = t[t.model == rung.upper()]
        lg = t[t.model == f"{rung.upper()} long"]
        if len(b) and len(lg):
            d = float(lg._mean.iloc[0] - b._mean.iloc[0])
            need = float(max(lg._sd.iloc[0], b._sd.iloc[0]))
            notes.append(f"- **{rung.upper()}: 30 vs 10 epochs → Δ {d:+.3f}** (needs > {need:.3f}) → "
                         f"{'✅ longer training helps' if d > need else '❌ not above seed noise'}")
            if rung == "r5" and d > need:
                choice = {"rung": "r5", "suffix": "_long", "seeds": SEEDS, "why": f"R5 long beats R5 by {d:+.3f}"}
    json.dump(choice, open(os.path.join(out_dir, "final_choice.json"), "w"), indent=2)
    if curves:
        pd.DataFrame(curves).to_csv(os.path.join(out_dir, "learning_curve.csv"), index=False)
        cv = pd.DataFrame(curves).groupby(["model", "epoch"]).val_cpm.mean().unstack(0).round(3).reset_index()
    show = t.drop(columns=["_mean", "_sd"])
    md = lambda df: ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns),  # noqa: E731
                     *("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))]
    lines = ["# Length + fit check (`scripts/09_check_summary.py`, seeds 42–44, VAL)", "", *md(show), "",
             *notes, "",
             "- **Fit:** TRAIN AUC ≫ VAL AUC (gap > ~0.03) → overfitting; both low → underfitting; small gap → fits well.", ""]
    if curves:
        lines += ["## Learning curve of the long runs (VAL CPM, mean of seeds)", "", *md(cv), ""]
    lines += [f"## Final model for Phase 4: **{choice['rung'].upper()}{choice['suffix']}**, seeds {choice['seeds']} ({choice['why']})"]
    open(os.path.join(out_dir, "check_report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
