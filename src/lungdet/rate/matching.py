"""Which candidate hits which nodule (LUNA16 rule: centre within the nodule's radius)."""
import numpy as np


def match(cand_xyz, nod_xyz, nod_diam, nod_is_gt):
    """Returns (hit, ignore) per candidate:
    hit    = index of the nearest GT nodule whose radius contains the candidate, else -1
    ignore = True if it hits no GT nodule but lies inside an ignore zone
    """
    cand_xyz = np.asarray(cand_xyz, float).reshape(-1, 3)
    hit = np.full(len(cand_xyz), -1)
    ignore = np.zeros(len(cand_xyz), bool)
    if len(nod_xyz) == 0 or len(cand_xyz) == 0:
        return hit, ignore
    nod_xyz = np.asarray(nod_xyz, float).reshape(-1, 3)
    radius = np.maximum(np.asarray(nod_diam, float) / 2, 1.5)
    gt = np.asarray(nod_is_gt, bool)
    d = np.linalg.norm(cand_xyz[:, None, :] - nod_xyz[None, :, :], axis=2)
    inside = d <= radius[None, :]
    d_gt = np.where(inside & gt[None, :], d, np.inf)
    has_gt = np.isfinite(d_gt).any(axis=1)
    hit[has_gt] = d_gt[has_gt].argmin(axis=1)
    ignore = ~has_gt & (inside & ~gt[None, :]).any(axis=1)
    return hit, ignore
