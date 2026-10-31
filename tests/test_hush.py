import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch = pytest.importorskip("torch")
from lungdet.discard.cnn import Restorer, Small3DNet, block_mask  # noqa: E402
from lungdet.discard.hessian import hessian_channels, sym3_eigvals  # noqa: E402


def test_closed_form_eigenvalues_match_torch():
    g = torch.Generator().manual_seed(0)
    a, b, c, d, e, f = torch.randn(6, 1000, generator=g)
    mine = torch.stack(sym3_eigvals(a, b, c, d, e, f), dim=1)
    A = torch.stack([torch.stack([a, d, e], 1), torch.stack([d, b, f], 1), torch.stack([e, f, c], 1)], 1)
    assert torch.allclose(mine, torch.linalg.eigvalsh(A), atol=1e-3)


def _cube(kind):
    i, j, k = np.indices((32, 32, 32)) - 16
    x = np.full((32, 32, 32), 0.1, np.float32)
    if kind == "ball":
        x[i ** 2 + j ** 2 + k ** 2 <= 25] = 0.7                    # 10 mm nodule
    else:
        x[i ** 2 + j ** 2 <= 4] = 0.7                              # 4 mm vessel along k
    return torch.from_numpy(x)[None, None]


def test_ball_is_blob_and_tube_is_vessel():
    hb = hessian_channels(_cube("ball"))[0, :, 16, 16, 16]
    ht = hessian_channels(_cube("tube"))[0, :, 16, 16, 16]
    assert hb[1] > 0.1 and hb[1] > hb[2]                           # ball: blobness wins
    assert ht[2] > 0.1 and ht[2] > ht[1]                           # tube: vesselness wins


def test_hush_shapes():
    x = torch.rand(2, 3, 32, 32, 32)
    logit, size = Small3DNet(in_ch=3, size_head=True).forward_both(x)
    assert logit.shape == (2,) and size.shape == (2,)
    assert Restorer(in_ch=3)(x).shape == (2, 1, 32, 32, 32)
    m = block_mask(2, ratio=0.6)
    assert m.shape == (2, 1, 32, 32, 32) and 0.4 < m.mean() < 0.8


def test_phase3_checkpoint_still_loads():
    old = Small3DNet(in_ch=1).state_dict()
    old = {k.replace("cls.", "head.3."): v for k, v in old.items()}  # Phase 3 key names
    Small3DNet(in_ch=1).load_state_dict(old)
