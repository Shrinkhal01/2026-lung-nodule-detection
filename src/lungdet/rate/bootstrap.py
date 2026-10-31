"""95% confidence intervals for CPM by resampling PATIENTS' scans (not candidates: candidates of one scan are not independent).

Per scan we keep: the best score of each GT nodule it holds (or nothing if never a candidate),
and its top false candidates (only the strongest ones can matter at <= 8 FP/scan).
"""
import numpy as np

from lungdet.rate.froc import cpm_from_scores


def per_scan_tables(scan_id, gt_id, ignore, scores, gt_per_scan, top_fp=200):
    """-> {scan: (tp_scores array, fp_scores array, n_gt)}. gt_per_scan: {scan: number of GT nodules}."""
    scan_id, gt_id, ignore, scores = map(np.asarray, (scan_id, gt_id, ignore, scores))
    out = {}
    order = np.argsort(scan_id, kind="stable")
    sid_sorted = scan_id[order]
    starts = np.searchsorted(sid_sorted, list(gt_per_scan), side="left")
    ends = np.searchsorted(sid_sorted, list(gt_per_scan), side="right")
    for s, a, b in zip(gt_per_scan, starts, ends):
        idx = order[a:b]
        g, ig, sc = gt_id[idx], ignore[idx], scores[idx]
        tp = {}
        for v, k in zip(sc[g >= 0], g[g >= 0]):
            tp[k] = max(tp.get(k, -np.inf), v)
        fp = np.sort(sc[(g < 0) & ~ig])[::-1][:top_fp]
        out[s] = (np.array(list(tp.values()), float), fp, gt_per_scan[s])
    return out


def bootstrap_cpm(tables_by_model, n_boot=1000, seed=0):
    """tables_by_model: {model: per_scan_tables(...)} on the SAME scans. Paired resamples.
    Returns {model: array of n_boot CPMs}."""
    scans = list(next(iter(tables_by_model.values())).keys())
    rng = np.random.default_rng(seed)
    out = {m: np.zeros(n_boot) for m in tables_by_model}
    for b in range(n_boot):
        pick = rng.integers(0, len(scans), len(scans))
        for m, tab in tables_by_model.items():
            rows = [tab[scans[i]] for i in pick]
            tp = np.concatenate([r[0] for r in rows]) if rows else np.zeros(0)
            fp = np.concatenate([r[1] for r in rows]) if rows else np.zeros(0)
            out[m][b] = cpm_from_scores(tp, fp, sum(r[2] for r in rows), len(scans))["cpm"]
    return out


def ci(values, level=0.95):
    lo, hi = np.quantile(values, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
