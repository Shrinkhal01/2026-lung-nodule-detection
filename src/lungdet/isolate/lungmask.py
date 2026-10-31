"""Classical lung mask on a 1 mm HU volume, axes (row, col, slice).

T-R-A-C-E:
  Threshold  air-like voxels (HU < -320)
  Remove     air touching the image border, SLICE BY SLICE (air outside the body, scanner padding)
  Accept     the 2 biggest remaining 3D blobs (the lungs), if each is >= 10% of the biggest
  Close      with a ball of radius r mm (pulls in nodules stuck to the lung wall)
  Eat holes  slice by slice (vessels and nodules inside the lung become part of the mask)

Why "slice by slice" in R (Phase 1 finding): removing border air in 3D deleted the WHOLE lung in
31 of 897 scans, because the lung touched outside air somewhere through one thin path.
Per slice, such a leak only costs that one slice.
"""
import numpy as np
from scipy import ndimage


def remove_border_air(air, per_slice=True):
    """Drop air components that touch the in-plane image border."""
    air = air.copy()
    if per_slice:
        for k in range(air.shape[2]):
            sl = air[:, :, k]
            if not sl.any():
                continue
            lab, n = ndimage.label(sl)
            edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
            edge = edge[edge > 0]
            if len(edge):
                sl[np.isin(lab, edge)] = False
        return air
    lab, _ = ndimage.label(air)
    edge = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel()]))
    return air & ~np.isin(lab, edge[edge > 0])


def lung_mask(hu, threshold=-320.0, closing_radius_mm=10.0, keep_ratio=0.1, per_slice_border=True):
    air = remove_border_air(hu < threshold, per_slice=per_slice_border)
    lab, n = ndimage.label(air)
    if n == 0:
        return np.zeros(hu.shape, dtype=bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
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
