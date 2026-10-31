"""Classical lung mask on a 1 mm HU volume, axes (row, col, slice).

T-R-A-C-E:
  Threshold  air-like voxels (HU < -320)
  Remove     air blobs touching the in-plane border (air outside the body, scanner padding)
  Accept     the 2 biggest remaining blobs (the lungs), if each is >= 10% of the biggest
  Close      with a ball of radius r mm (pulls in nodules stuck to the lung wall)
  Eat holes  slice by slice (vessels and nodules inside the lung become part of the mask)
"""
import numpy as np
from scipy import ndimage


def lung_mask(hu, threshold=-320.0, closing_radius_mm=10.0, keep_ratio=0.1):
    air = hu < threshold
    lab, n = ndimage.label(air)
    if n == 0:
        return np.zeros(hu.shape, dtype=bool)
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(),
                                       lab[:, 0].ravel(), lab[:, -1].ravel()]))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    sizes[border] = 0
    order = np.argsort(sizes)[::-1]
    keep = [l for l in order[:2] if sizes[l] > 0 and sizes[l] >= keep_ratio * sizes[order[0]]]
    mask = np.isin(lab, keep)
    del lab, air
    if closing_radius_mm > 0 and mask.any():
        r = float(closing_radius_mm)
        dilated = ndimage.distance_transform_edt(~mask) <= r   # grow by r ...
        mask = ndimage.distance_transform_edt(dilated) > r    # ... then shrink by r = closing
    for k in range(mask.shape[2]):
        if mask[:, :, k].any():
            mask[:, :, k] = ndimage.binary_fill_holes(mask[:, :, k])
    return mask


def crop_box(mask, margin=10):
    """Bounding box of the mask plus a margin, clipped to the volume. Returns (lo, hi) index arrays."""
    if not mask.any():
        return np.zeros(3, int), np.array(mask.shape)
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(axis=0) - margin, 0)
    hi = np.minimum(idx.max(axis=0) + margin + 1, mask.shape)
    return lo, hi
