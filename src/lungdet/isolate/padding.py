"""Scanner padding: the constant value OUTSIDE the round field of view (the corners of the square image).

Most LIDC scans pad with about -2048 HU (air-like, harmless once clipped to -1000).
Some pad with a tissue-like value (e.g. 0). Then the padding is a solid ring that seals the air around
the body off from the image border, and that air is mistaken for lung (Phase 1 finding: 5 scans).
Fix: padding voxels connected to the image border become air.
"""
import numpy as np
from scipy import ndimage


def detect_padding(vol, min_share=0.9):
    """The value shared by the 4 corner columns in >= min_share of all slices, else None."""
    corners = np.concatenate([vol[0, 0, :], vol[0, -1, :], vol[-1, 0, :], vol[-1, -1, :]])
    vals, counts = np.unique(corners, return_counts=True)
    if counts.max() < min_share * len(corners):
        return None
    return float(vals[counts.argmax()])


def padding_to_air(vol, air=-1000.0):
    """Returns (volume, padding value or None). Only acts when the padding is NOT already air-like."""
    pad = detect_padding(vol)
    if pad is None or pad <= air:
        return vol, pad
    out = np.array(vol, dtype=np.float32, copy=True)
    for k in range(out.shape[2]):
        lab, _ = ndimage.label(out[:, :, k] == pad)
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        edge = edge[edge > 0]
        if len(edge):
            out[:, :, k][np.isin(lab, edge)] = air
    return out, pad
