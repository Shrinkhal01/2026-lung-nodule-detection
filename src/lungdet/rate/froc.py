"""FROC + CPM (LUNA16 style).

- each GT nodule counts once: its score = the best candidate that hits it (extra hits are not FPs)
- candidates in ignore zones count as nothing
- sensitivity at k FP/scan = best sensitivity whose FP/scan <= k
- CPM = mean sensitivity at 1/8, 1/4, 1/2, 1, 2, 4, 8 FP/scan
"""
import numpy as np

RATES = (0.125, 0.25, 0.5, 1, 2, 4, 8)


def froc(scores, gt_id, ignore, n_gt, n_scans, rates=RATES):
    scores, gt_id, ignore = np.asarray(scores, float), np.asarray(gt_id), np.asarray(ignore, bool)
    tp = {}
    for s, g in zip(scores[gt_id >= 0], gt_id[gt_id >= 0]):
        tp[g] = max(tp.get(g, -np.inf), s)
    tp_scores = np.sort(np.array(list(tp.values()), float))[::-1]
    fp_scores = np.sort(scores[(gt_id < 0) & ~ignore])[::-1]
    thresholds = np.unique(np.concatenate([tp_scores, fp_scores]))[::-1]
    sens = np.searchsorted(-tp_scores, -thresholds, side="right") / max(n_gt, 1)
    fps = np.searchsorted(-fp_scores, -thresholds, side="right") / max(n_scans, 1)
    at = {r: float(sens[fps <= r].max()) if (fps <= r).any() else 0.0 for r in rates}
    return {"sens_at": at, "cpm": float(np.mean(list(at.values()))),
            "max_sens": float(len(tp_scores) / max(n_gt, 1)), "curve": (fps, sens)}
