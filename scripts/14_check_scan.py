"""Phase 5 · compare predict.py's findings.csv with the radiologists' marks for ONE cached scan.

  python scripts/14_check_scan.py data/phase5_local/<series_uid>.npz results/phase5/smoke/findings.csv
Needs results/phase1/{scans,nodules}.csv (already in the repo). Same hit rule as the evaluation: centre inside the nodule's radius.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lungdet.rate.matching import match  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("findings")
    ap.add_argument("--tables", default=str(ROOT / "results" / "phase1"))
    a = ap.parse_args()
    uid = Path(a.npz).stem
    scans = pd.read_csv(os.path.join(a.tables, "scans.csv"))
    nod = pd.read_csv(os.path.join(a.tables, "nodules.csv"))
    sc = scans[scans.series_uid == uid]
    if sc.empty:
        sys.exit(f"series {uid} is not in scans.csv")
    sc = sc.iloc[0]
    n = nod[nod.scan_id == sc.scan_id].reset_index(drop=True)
    f = pd.read_csv(a.findings)
    print(f"{sc.patient_id} ({sc.split} split): {int((n.role == 'gt').sum())} radiologist nodule(s), "
          f"{int((n.role != 'gt').sum())} ignore zone(s), {len(f)} finding(s)\n")
    if len(f) and len(n):
        hit, ign = match(f[["i", "j", "k"]].to_numpy(float), n[["mm_i", "mm_j", "mm_k"]].to_numpy(float),
                         n.diameter_mm.to_numpy(), (n.role == "gt").to_numpy())
    else:
        hit, ign = np.full(len(f), -1), np.zeros(len(f), bool)
    for idx, g in n[n.role == "gt"].iterrows():
        rows = f[hit == idx]
        if len(rows):
            b = rows.sort_values("prob", ascending=False).iloc[0]
            print(f"  FOUND   {g.diameter_mm:5.1f} mm, {int(g.n_readers)} readers -> finding #{int(b['rank'])}, prob {b.prob:.3f}")
        else:
            print(f"  MISSED  {g.diameter_mm:5.1f} mm, {int(g.n_readers)} readers (no finding inside its radius)")
    fp = int(((hit < 0) & ~ign).sum())
    print(f"\nfalse alarms on this scan: {fp}   (findings in ignore zones, counted as nothing: {int(ign.sum())})")
    print("ONE scan says nothing about the CPM; it only shows the whole chain works on real data.")


if __name__ == "__main__":
    main()
