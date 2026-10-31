"""Operating points: which probability threshold gives about k false alarms per scan?

Picked on VAL (never on TEST). A false alarm = a candidate that hits no ground-truth nodule and lies in no
ignore zone, exactly as in rate/froc.py. Detections are kept when prob > threshold.
"""
import numpy as np

RATES = (0.125, 0.25, 0.5, 1, 2, 4, 8)


def operating_points(prob, gt_id, ignore, n_gt, n_scans, rates=RATES):
    """{rate: (threshold, sensitivity, fp_per_scan)}. Sensitivity = share of GT nodules with a candidate above it."""
    prob, gt_id, ignore = np.asarray(prob, float), np.asarray(gt_id), np.asarray(ignore, bool)
    fp = np.sort(prob[(gt_id < 0) & ~ignore])[::-1]
    best = {}
    for p, g in zip(prob[gt_id >= 0], gt_id[gt_id >= 0]):
        best[g] = max(best.get(g, -np.inf), p)
    tp = np.array(list(best.values()))
    out = {}
    for r in rates:
        k = int(np.floor(r * n_scans))              # false alarms we may afford in total
        thr = float(fp[k]) if k < len(fp) else 0.0  # keep prob > thr: at most k false alarms survive
        out[r] = (thr, float((tp > thr).sum() / max(n_gt, 1)), float((fp > thr).sum() / max(n_scans, 1)))
    return out
