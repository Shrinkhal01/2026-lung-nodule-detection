import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.isolate.geometry import mm_to_index, resample, resampled_shape, voxel_to_mm
from lungdet.isolate.lungmask import crop_box, lung_mask


def test_shape_of_scan_1():
    # LIDC-IDRI-0001: 512 x 512 x 133 at 0.703 x 0.703 x 2.5 mm
    assert resampled_shape((512, 512, 133), (0.703, 0.703, 2.5)) == (360, 360, 331)


def test_volume_and_point_move_together():
    vol = np.zeros((50, 50, 20), np.float32)
    vol[20, 20, 8] = 1000.0                          # one bright voxel
    spacing = (0.7, 0.7, 2.5)
    out = resample(vol, spacing, 1.0, cval=0.0)
    expected = np.rint(mm_to_index(voxel_to_mm((20, 20, 8), spacing))).astype(int)
    assert tuple(expected) == (14, 14, 20)
    assert np.unravel_index(out.argmax(), out.shape) == (14, 14, 20)


def _phantom():
    """1 mm chest, axes (row, col, slice): air, a body (40 HU), two lungs (-850 HU) 36 mm apart,
    and a 12 mm nodule (30 HU) stuck to the outer wall of the left lung, centre (60, 18, 30)."""
    i, j, k = np.ogrid[0:120, 0:120, 0:60]
    hu = np.full((120, 120, 60), -1000.0, np.float32)
    put = lambda cond, v: hu.__setitem__(np.broadcast_to(cond, hu.shape), v)  # noqa: E731
    put((i - 60) ** 2 + (j - 60) ** 2 <= 55 ** 2, 40.0)
    for cj in (28, 92):
        put(((i - 60) / 30) ** 2 + ((j - cj) / 14) ** 2 + ((k - 30) / 25) ** 2 <= 1, -850.0)
    put((i - 60) ** 2 + (j - 18) ** 2 + (k - 30) ** 2 <= 6 ** 2, 30.0)
    return hu


def test_lung_mask_keeps_lungs_and_wall_nodule():
    mask = lung_mask(_phantom(), threshold=-320, closing_radius_mm=10)
    assert not mask[3, 3, 30]                       # air outside the body is not lung
    assert mask[60, 28, 30] and mask[60, 92, 30]    # both lungs
    assert mask[60, 21, 30]                         # lung-side half of the wall nodule is pulled in (closing)
    assert not mask[60, 60, 30]                     # the 36 mm gap between the lungs stays out


def test_wall_nodule_lost_without_closing():
    mask = lung_mask(_phantom(), threshold=-320, closing_radius_mm=0)
    assert not mask[60, 21, 30]                     # why the closing step exists


def test_crop_box_adds_margin_and_clips():
    m = np.zeros((50, 50, 50), bool)
    m[10:20, 30:45, 0:5] = True
    lo, hi = crop_box(m, margin=10)
    assert tuple(lo) == (0, 20, 0) and tuple(hi) == (30, 50, 15)
