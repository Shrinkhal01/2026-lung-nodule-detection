import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.nominate.blobs import bundle, log_peaks
from lungdet.rate.froc import froc
from lungdet.rate.matching import match


def _lung_with_ball_and_vessel():
    """Lung (-850 HU) with a 10 mm ball at (30, 30, 30) and a 4 mm-wide vessel along k at (30, 60)."""
    i, j, k = np.ogrid[0:64, 0:90, 0:64]
    hu = np.full((64, 90, 64), -850.0, np.float32)
    hu[np.broadcast_to((i - 30) ** 2 + (j - 30) ** 2 + (k - 30) ** 2 <= 25, hu.shape)] = 0.0
    hu[np.broadcast_to((i - 30) ** 2 + (j - 60) ** 2 <= 4, hu.shape)] = 0.0
    return hu, np.ones(hu.shape, bool)


def test_ball_beats_vessel():
    # theory (scale-normalised LoG, matched size): ball ~0.86 x contrast, tube ~0.73 x contrast
    hu, mask = _lung_with_ball_and_vessel()
    c = bundle(log_peaks(hu, mask, sigmas=[1.0, 1.5, 2.5, 4.0], floor=30, min_hu=-700))
    top = c[0]
    assert np.linalg.norm(top[:3] - [30, 30, 30]) <= 1.5          # strongest candidate = the nodule centre
    vessel = c[np.abs(c[:, 1] - 60) <= 3]
    assert len(vessel) > 0 and vessel[:, 4].max() < top[4]        # the vessel beeps too, but weaker


def test_bundle_merges_nearby_peaks():
    peaks = np.array([[10, 10, 10, 2.5, 100.0], [11, 10, 10, 1.5, 80.0], [30, 30, 30, 1.0, 50.0]])
    out = bundle(peaks)
    assert len(out) == 2 and out[0, 4] == 100.0


def test_match_gt_and_ignore():
    hit, ign = match([[0, 0, 0], [0, 0, 4], [20, 0, 0], [50, 50, 50]],
                     nod_xyz=[[0, 0, 0], [20, 0, 1]], nod_diam=[10.0, 6.0], nod_is_gt=[True, False])
    assert list(hit) == [0, 0, -1, -1]             # 4 mm from a 10 mm nodule = inside its 5 mm radius
    assert list(ign) == [False, False, True, False]


def test_froc_toy_cpm():
    # scan A: GT g0 hit at 0.9, FP 0.8 · scan B: GT g1 hit at 0.5, FP 0.95
    r = froc(scores=[0.9, 0.8, 0.5, 0.95], gt_id=[0, -1, 1, -1], ignore=[False] * 4, n_gt=2, n_scans=2)
    assert r["sens_at"][0.5] == 0.5 and r["sens_at"][1] == 1.0 and r["sens_at"][0.25] == 0.0
    assert abs(r["cpm"] - 4.5 / 7) < 1e-9
