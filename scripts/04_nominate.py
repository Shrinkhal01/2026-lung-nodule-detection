"""Phase 2 · N (Nominate): LoG candidates for every scan; the selection rule is CHOSEN AUTOMATICALLY on TRAIN.

  python scripts/04_nominate.py --limit 5     # smoke test
  python scripts/04_nominate.py               # all 897 scans (~5-8 min on 64 workers)

Two selection families are swept on TRAIN scans only:
  global top-N      keep the N strongest candidates per scan
  per-size top-q    keep the q strongest candidates of EACH blob size per scan
                    (small faint nodules no longer compete with big bright vessels)
Chosen = the setting with the fewest candidates/scan that reaches TRAIN recall >= recall_target.
VAL is only a check. TEST is not used and not reported.

Writes $SCRATCH/candidates/candidates_all.parquet  (every stored candidate, labelled, with ranks)
       $SCRATCH/candidates/candidates.parquet      (chosen selection = input for Phase 3)
       results/phase2: nominate_sweep.csv, nominate_report.md, nominate_choice.json, froc_r0_val.csv
"""
import argparse
import json
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
    c = bundle(peaks, nom["max_per_scan"], mode="blob")        # sorted strongest first
    xyz = c[:, :3] + off                                       # full 1 mm grid = mm
    hit, ign = match(xyz, [n[1] for n in nods], [n[2] for n in nods], [n[3] == "gt" for n in nods])
    gt_ids = np.array([n[0] for n in nods] + [-1])
    df = pd.DataFrame({"scan_id": np.int32(scan_id), "i": xyz[:, 0].astype(np.int16), "j": xyz[:, 1].astype(np.int16),
                       "k": xyz[:, 2].astype(np.int16), "sigma": c[:, 3].astype(np.float32),
                       "response": c[:, 4].astype(np.float32),
                       "gt_id": gt_ids[np.where(hit >= 0, hit, -1)].astype(np.int32), "ignore": ign,
                       "global_rank": np.arange(len(c), dtype=np.int32)})
    df["scale_rank"] = df.groupby("sigma").cumcount().astype(np.int32)   # rank within its blob size
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
    suffix = f"_limit{args.limit}" if args.limit else ""

    scans = pd.read_csv(os.path.join(tables, "scans.csv"))
    scans = scans[scans.eligible].sort_values("scan_id")
    if args.limit:
        scans = scans.head(args.limit)
    nod = pd.read_csv(os.path.join(tables, "nodules.csv"))
    nod["gid"] = np.arange(len(nod))
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
    cand = pd.concat(parts, ignore_index=True).merge(scans[["scan_id", "split"]], on="scan_id")
    stats = pd.DataFrame(stats).merge(scans[["scan_id", "patient_id", "split"]], on="scan_id")
    cand.to_parquet(os.path.join(cand_dir, f"candidates_all{suffix}.parquet"))

    gt = nod[(nod.role == "gt") & nod.scan_id.isin(scans.scan_id)]
    n_scans = scans.split.value_counts()

    def evaluate(sel, split):
        c = cand[sel & (cand.split == split)]
        n_gt = (gt.split == split).sum()
        return (c[c.gt_id >= 0].gt_id.nunique() / max(n_gt, 1), len(c) / max(n_scans.get(split, 0), 1))

    # sweep both selection families on TRAIN
    rows = []
    for n_top in nom["sweep_global_top"]:
        r, cps = evaluate(cand.global_rank < n_top, "train")
        rows.append({"rule": "global_top", "value": n_top, "train_recall": round(r, 4), "train_cands_per_scan": round(cps)})
    for q in nom["sweep_per_scale_top"]:
        r, cps = evaluate(cand.scale_rank < q, "train")
        rows.append({"rule": "per_size_top", "value": q, "train_recall": round(r, 4), "train_cands_per_scan": round(cps)})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(out_dir, f"nominate_sweep{suffix}.csv"), index=False)
    ok = sweep[sweep.train_recall >= nom["recall_target"]]
    chosen = (ok.sort_values("train_cands_per_scan").iloc[0] if len(ok)
              else sweep.sort_values(["train_recall", "train_cands_per_scan"], ascending=[False, True]).iloc[0])
    col = "global_rank" if chosen.rule == "global_top" else "scale_rank"
    sel = cand[col] < int(chosen.value)
    final = cand[sel]
    final.to_parquet(os.path.join(cand_dir, f"candidates{suffix}.parquet"))
    val_recall, val_cps = evaluate(sel, "val")
    choice = {"rule": chosen.rule, "value": int(chosen.value), "train_recall": float(chosen.train_recall),
              "train_cands_per_scan": float(chosen.train_cands_per_scan), "val_recall": float(val_recall),
              "val_cands_per_scan": float(val_cps), "target_reached": bool(len(ok))}
    json.dump(choice, open(os.path.join(out_dir, f"nominate_choice{suffix}.json"), "w"), indent=2)

    g_tr = gt[gt.split == "train"].copy()
    g_tr["found"] = g_tr.gid.isin(final[final.split == "train"].gt_id)
    g_tr["size"] = pd.cut(g_tr.diameter_mm, [0, 6, 10, 20, 30, 100], labels=["3-6", "6-10", "10-20", "20-30", ">30"])
    by_size = g_tr.groupby("size", observed=False).agg(nodules=("gid", "size"), recall=("found", "mean")).reset_index()
    by_size["recall"] = by_size.recall.round(3)

    v = final[final.split == "val"]
    r0 = froc(v.response, v.gt_id, v.ignore, n_gt=(gt.split == "val").sum(), n_scans=n_scans.get("val", 0))
    pd.DataFrame({"fp_per_scan": r0["curve"][0], "sensitivity": r0["curve"][1]}).to_csv(
        os.path.join(out_dir, f"froc_r0_val{suffix}.csv"), index=False)

    per_scan = final.groupby("scan_id").size()
    broken = stats[stats.patient_id.isin(BROKEN_MASKS)]
    best = sweep.sort_values("train_recall").iloc[-1]
    lines = [
        f"# Phase 2 · nominate report (`scripts/04_nominate.py`{' --limit ' + str(args.limit) if args.limit else ''})", "",
        f"- scans: **{len(scans)}**, runtime **{time.time() - t0:.0f} s**",
        f"- stored candidates per scan (storage cap {nom['max_per_scan']}): median **{stats.n_kept.median():.0f}**",
        f"- best TRAIN recall in the sweep: **{100 * best.train_recall:.1f}%** ({best.rule} = {int(best.value)})",
        f"- **chosen: {chosen.rule} = {int(chosen.value)}** "
        f"({'fewest candidates with' if len(ok) else 'TARGET NOT REACHED; best'} TRAIN recall ≥ {100 * nom['recall_target']:.0f}%)",
        f"  - TRAIN recall **{100 * chosen.train_recall:.1f}%** at **{chosen.train_cands_per_scan:.0f}** candidates/scan",
        f"  - VAL recall **{100 * val_recall:.1f}%** at **{val_cps:.0f}** candidates/scan (check, not tuned)",
        f"- candidates per scan after selection: median **{per_scan.median():.0f}**, 90th **{per_scan.quantile(.9):.0f}**",
        f"- broken-mask scans: {', '.join(f'{r.patient_id[-4:]}={int(per_scan.get(r.scan_id, 0))}' for r in broken.itertuples()) or 'none in this run'}",
        "", "## TRAIN sweep", "", *df_to_md(sweep), "",
        "## TRAIN recall by nodule size (chosen selection)", "", *df_to_md(by_size), "",
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
