"""Phase 1 · P-L-A: Pick scans, Label nodules, Assign splits.

  python scripts/02_ground_truth.py [--workers 32]

Writes results/phase1/: scans.csv, nodules.csv (role = gt | ignore), gt_report.md
Needs only pylidc's database (no images), so it is fast.
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
from lungdet.fetch.ground_truth import assign_splits, summarise_cluster  # noqa: E402
from lungdet.fetch.inventory import scan_series_folders  # noqa: E402
from lungdet.fetch.pylidc_compat import import_pylidc, write_pylidcrc  # noqa: E402

LUNA16 = {"scans": 888, "nodules": 1186}


_pl = None


def _init():
    global _pl
    import warnings
    warnings.filterwarnings("ignore")
    _pl = import_pylidc()


def _cluster(task):
    scan_id, min_readers, min_diam = task
    scan = _pl.query(_pl.Scan).filter(_pl.Scan.id == scan_id).first()
    spacing = (scan.pixel_spacing, scan.pixel_spacing, scan.slice_spacing)
    try:
        try:
            clusters = scan.cluster_annotations(verbose=False)
        except TypeError:
            clusters = scan.cluster_annotations()
    except Exception as e:  # noqa: BLE001
        return scan_id, [], f"{type(e).__name__}: {e}"
    rows = []
    for n, anns in enumerate(clusters):
        row = summarise_cluster(anns, spacing, min_readers, min_diam)
        row.update(scan_id=scan_id, nodule_idx=n)
        rows.append(row)
    return scan_id, rows, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    gt_cfg = cfg["ground_truth"]
    out_dir = cfg["paths"]["tables"]
    os.makedirs(out_dir, exist_ok=True)
    write_pylidcrc(cfg["paths"]["lidc_dicom"])
    t0 = time.time()

    # P: pick scans
    pl = import_pylidc()
    on_disk = scan_series_folders(cfg["paths"]["raw"])
    scans = pd.DataFrame([{
        "scan_id": s.id, "patient_id": s.patient_id, "series_uid": s.series_instance_uid,
        "slice_thickness": float(s.slice_thickness), "slice_spacing": float(s.slice_spacing),
        "pixel_spacing": float(s.pixel_spacing), "n_marks": len(s.annotations),
        "downloaded": s.series_instance_uid in on_disk,
    } for s in pl.query(pl.Scan).all()])
    scans["eligible"] = scans.downloaded & (scans.slice_thickness <= gt_cfg["max_slice_thickness_mm"])
    print(f"P: {len(scans)} scans in pylidc, {scans.downloaded.sum()} downloaded, "
          f"{scans.eligible.sum()} with slice thickness <= {gt_cfg['max_slice_thickness_mm']} mm")

    # L: label nodules (cluster the readers' marks), in parallel
    tasks = [(int(i), gt_cfg["min_readers"], gt_cfg["min_diameter_mm"]) for i in scans.loc[scans.eligible, "scan_id"]]
    rows, errors = [], []
    with get_context("spawn").Pool(args.workers or cfg["isolate"]["workers"], initializer=_init) as pool:
        for scan_id, r, err in pool.imap_unordered(_cluster, tasks, chunksize=4):
            rows.extend(r)
            if err:
                errors.append((scan_id, err))
    nod = pd.DataFrame(rows).merge(scans[["scan_id", "patient_id", "series_uid"]], on="scan_id")
    print(f"L: {len(nod)} clusters -> {(nod.role == 'gt').sum()} ground truth, {(nod.role == 'ignore').sum()} ignore; "
          f"{len(errors)} scans failed")

    # A: assign splits by patient
    split = assign_splits(scans.loc[scans.eligible, "patient_id"], cfg["split"]["fractions"], cfg["split"]["seed"])
    scans["split"] = scans.patient_id.map(split).fillna("none")
    nod["split"] = nod.patient_id.map(split)
    shared = scans[scans.eligible].groupby("patient_id").split.nunique().gt(1).sum()

    cols = ["scan_id", "patient_id", "series_uid", "split", "nodule_idx", "role", "n_readers", "diameter_mm",
            "i", "j", "k", "mm_i", "mm_j", "mm_k"]
    nod = nod.sort_values(["scan_id", "nodule_idx"])[cols]
    scans.to_csv(os.path.join(out_dir, "scans.csv"), index=False)
    nod.to_csv(os.path.join(out_dir, "nodules.csv"), index=False)

    # report
    gt = nod[nod.role == "gt"]
    el = scans[scans.eligible]
    d = gt.diameter_mm
    per_split = (el.groupby("split").agg(patients=("patient_id", "nunique"), scans=("scan_id", "size"))
                 .join(gt.groupby("split").size().rename("gt_nodules")))
    pct = lambda a, b: 100.0 * (a - b) / b  # noqa: E731
    gates = [
        ("eligible scans within 5% of LUNA16's 888", abs(pct(len(el), LUNA16["scans"])) <= 5, f"{len(el)} ({pct(len(el), LUNA16['scans']):+.1f}%)"),
        ("GT nodules within 5% of LUNA16's 1,186", abs(pct(len(gt), LUNA16["nodules"])) <= 5, f"{len(gt)} ({pct(len(gt), LUNA16['nodules']):+.1f}%)"),
        ("0 patients shared between splits", shared == 0, str(shared)),
        ("0 scans failed clustering", not errors, str(len(errors))),
    ]
    lines = [
        "# Phase 1 · ground truth report (`scripts/02_ground_truth.py`)", "",
        f"- pylidc scans: **{len(scans)}**, downloaded: **{scans.downloaded.sum()}**, "
        f"eligible (≤ {gt_cfg['max_slice_thickness_mm']} mm): **{len(el)}** from **{el.patient_id.nunique()}** patients",
        f"- clusters: **{len(nod)}** → ground truth **{len(gt)}** (≥ {gt_cfg['min_readers']} readers, ≥ {gt_cfg['min_diameter_mm']} mm), "
        f"ignore zones **{(nod.role == 'ignore').sum()}**",
        f"- clusters with more than 4 marks (pylidc merged too much): **{(nod.n_readers > 4).sum()}**",
        f"- GT diameter (mm): min **{d.min():.1f}**, median **{d.median():.1f}**, 90th **{d.quantile(.9):.1f}**, "
        f"99th **{d.quantile(.99):.1f}**, max **{d.max():.1f}**",
        f"- GT nodules above 30 mm: **{(d > 30).sum()}**, above 40 mm: **{(d > 40).sum()}** (input for the Phase 2 size cap)",
        "", "## Per split", "", *df_to_md(per_split.reset_index()),
        "", "## Gates", "", "| Gate | Result | Value |", "|---|:-:|---|",
        *[f"| {g} | {'PASS' if ok else 'FAIL'} | {v} |" for g, ok, v in gates],
        "", f"Runtime: {time.time() - t0:.0f} s",
    ]
    if errors:
        lines += ["", "## Failed scans", *[f"- scan {s}: {e}" for s, e in errors[:20]]]
    open(os.path.join(out_dir, "gt_report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
