import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.discard.features import FEATURES, cube_features
from lungdet.discard.patches import crop_cubes


def test_cube_is_centred_on_the_candidate():
    hu = np.full((40, 50, 60), -1000, np.int16)
    hu[10, 20, 30] = 500
    c = crop_cubes(hu, [[10, 20, 30], [0, 0, 0]])
    assert c.shape == (2, 32, 32, 32)
    assert c[0, 16, 16, 16] == 500 and c[0].sum() == 500 + (-1000) * (32 ** 3 - 1)
    assert c[1, 0, 0, 0] == -1000                       # outside the volume = padded air


def test_features_see_a_ball():
    g = np.indices((32, 32, 32)) - 16
    ball = np.where((g ** 2).sum(0) <= 25, 0, -850).astype(np.int16)    # 10 mm nodule in lung
    f = dict(zip(FEATURES, cube_features(ball[None], sigma=[2.9])[0]))
    assert f["centre_hu"] == 0 and f["ball_mean"] == 0 and f["ball_dense"] == 1
    assert f["contrast"] > 500                          # ball much denser than its shell


def test_cnn_shapes_and_focal_loss():
    torch = pytest.importorskip("torch")
    from lungdet.discard.cnn import Small3DNet, augment, focal_loss
    x = torch.rand(4, 1, 32, 32, 32)
    assert Small3DNet()(augment(x)).shape == (4,)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    easy, hard = torch.tensor([9.0, -9.0, 9.0, -9.0]), torch.tensor([-9.0, 9.0, -9.0, 9.0])
    assert focal_loss(easy, y) < 1e-4 < focal_loss(hard, y)   # focal loss ignores easy cases
