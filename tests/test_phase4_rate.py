import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.rate.bootstrap import bootstrap_cpm, ci, per_scan_tables
from lungdet.rate.froc import cpm_from_scores, froc


def _toy():
    # scan 1: GT 0 hit twice (0.9, 0.4), FP 0.8 · scan 2: GT 1 hit (0.5), FP 0.95 · scan 3: GT 2 never a candidate
    scan = [1, 1, 1, 2, 2]
    gt = [0, 0, -1, 1, -1]
    ign = [False] * 5
    sc = [0.9, 0.4, 0.8, 0.5, 0.95]
    return scan, gt, ign, sc, {1: 1, 2: 1, 3: 1}


def test_tables_give_the_same_cpm_as_froc():
    scan, gt, ign, sc, gps = _toy()
    full = froc(sc, gt, ign, n_gt=3, n_scans=3)["cpm"]
    t = per_scan_tables(scan, gt, ign, sc, gps)
    tp = np.concatenate([v[0] for v in t.values()])
    fp = np.concatenate([v[1] for v in t.values()])
    assert abs(cpm_from_scores(tp, fp, 3, 3)["cpm"] - full) < 1e-12
    assert len(t[3][0]) == 0 and t[3][2] == 1          # a missed GT still counts in the denominator


def test_bootstrap_is_paired_and_reproducible():
    scan, gt, ign, sc, gps = _toy()
    t = per_scan_tables(scan, gt, ign, sc, gps)
    a = bootstrap_cpm({"m": t, "same": t}, n_boot=50, seed=1)
    b = bootstrap_cpm({"m": t}, n_boot=50, seed=1)
    assert np.allclose(a["m"], a["same"]) and np.allclose(a["m"], b["m"])
    lo, hi = ci(a["m"])
    assert 0 <= lo <= hi <= 1
