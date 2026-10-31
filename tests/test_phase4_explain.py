import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch = pytest.importorskip("torch")
from lungdet.explain.occlusion import occlusion_map, rank_corr, score_map  # noqa: E402


class CentreModel(torch.nn.Module):
    """Probability = brightness of the 4^3 centre -> hiding the centre must drop it the most."""
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return (x[:, 0, 14:18, 14:18, 14:18].mean((1, 2, 3)) - 0.5) * 10 * self.w


def test_occlusion_finds_the_centre():
    cube = torch.full((32, 32, 32), -850, dtype=torch.int16)
    cube[12:20, 12:20, 12:20] = 400                    # bright centre -> p ~ 0.99
    heat, p = occlusion_map(CentreModel(), cube)
    hit, conc = score_map(heat, (16, 16, 16), radius=4)
    assert p > 0.9 and hit and conc > 0.3


def test_rank_corr():
    a = np.random.default_rng(0).random((8, 8, 8))
    assert abs(rank_corr(a, a) - 1) < 1e-9 and abs(rank_corr(a, -a) + 1) < 1e-9
