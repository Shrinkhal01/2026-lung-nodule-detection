"""N · the metal detector: multi-scale LoG blob detection inside the lung mask.

B-L-O-B:
  Blur        at several sizes sigma (mm)
  Laplacian   -sigma^2 * LoG: big where a bright blob of radius ~ sqrt(3)*sigma sits
  Only peaks  local maxima inside the lung mask, above a floor, at HU > min_hu (not air)
  Bundle      peaks closer than the stronger peak's blob radius become one candidate
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def log_peaks(hu, mask, sigmas, floor=30.0, min_hu=-700.0):
    """All scale-space peaks. Returns array (n, 5): i, j, k, sigma, response."""
    v = np.where(mask, hu, -1000).astype(np.float32)      # outside the lung = air, so the wall makes no blobs
    out = []
    for s in sigmas:
        r = -(s ** 2) * ndimage.gaussian_laplace(v, sigma=s)
        peak = (r == ndimage.maximum_filter(r, size=3)) & (r > floor) & mask & (v > min_hu)
        idx = np.argwhere(peak)
        if len(idx):
            out.append(np.column_stack([idx, np.full(len(idx), s), r[peak]]))
    return np.vstack(out) if out else np.zeros((0, 5))


def bundle(peaks, max_keep=3000, mode="blob"):
    """Greedy non-maximum suppression, strongest first.
    mode="blob":  drop weaker peaks inside the STRONGER blob's radius (>= 3 mm)
                  -> a big blob (sigma 13 -> 22.5 mm) also wipes out small nodules next to it
    mode="scale": drop weaker peaks only inside the SMALLER blob's radius (>= 3 mm) = the same blob seen twice
    """
    if len(peaks) == 0:
        return peaks
    peaks = peaks[np.argsort(-peaks[:, 4])]
    tree = cKDTree(peaks[:, :3])
    alive = np.ones(len(peaks), bool)
    keep = []
    for n in range(len(peaks)):
        if not alive[n]:
            continue
        keep.append(n)
        if len(keep) >= max_keep:
            break
        near = np.array(tree.query_ball_point(peaks[n, :3], max(3.0, np.sqrt(3) * peaks[n, 3])), dtype=int)
        if mode == "scale" and len(near):
            d = np.linalg.norm(peaks[near, :3] - peaks[n, :3], axis=1)
            limit = np.maximum(3.0, np.sqrt(3) * np.minimum(peaks[near, 3], peaks[n, 3]))
            near = near[d <= limit]
        alive[near] = False
    return peaks[keep]
