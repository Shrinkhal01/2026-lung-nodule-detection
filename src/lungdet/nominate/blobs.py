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


def bundle(peaks, max_keep=3000):
    """Greedy non-maximum suppression: strongest first; drop weaker peaks inside its blob radius (>= 3 mm)."""
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
        radius = max(3.0, np.sqrt(3) * peaks[n, 3])
        alive[tree.query_ball_point(peaks[n, :3], radius)] = False
    return peaks[keep]
