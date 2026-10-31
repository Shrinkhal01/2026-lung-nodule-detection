"""Occlusion sensitivity: cover one 8 mm block of the cube with normal lung (-850 HU) at a time,
re-score, and record how much the nodule probability DROPS. Big drop = the model looked there.

Scored, not just shown (v2 lesson):
  hit           the hottest voxel lies inside the nodule (radius + 2 mm)
  concentration share of the positive heat that falls inside the nodule
  sanity        a RANDOM-weights network must give a different map (low rank correlation)
"""
import numpy as np
import torch

from lungdet.discard.cnn import predict

LUNG_HU = -850


def occlusion_map(model, cube_i16, transform=None, block=8, stride=4, device=None):
    """cube_i16: (32, 32, 32) int16 tensor -> (heat (32,32,32) numpy, base probability)."""
    device = device or next(model.parameters()).device
    size = cube_i16.shape[0]
    starts = list(range(0, size - block + 1, stride))
    coords = [(a, b, c) for a in starts for b in starts for c in starts]
    batch = cube_i16.unsqueeze(0).repeat(len(coords) + 1, 1, 1, 1).clone()
    for n, (a, b, c) in enumerate(coords, 1):
        batch[n, a:a + block, b:b + block, c:c + block] = LUNG_HU
    p = predict(model, batch.to(device), batch=512, transform=transform, device=device)
    base, drops = p[0], p[0] - p[1:]
    heat = np.zeros((size,) * 3, np.float32)
    count = np.zeros((size,) * 3, np.float32)
    for d, (a, b, c) in zip(drops, coords):
        heat[a:a + block, b:b + block, c:c + block] += d
        count[a:a + block, b:b + block, c:c + block] += 1
    return heat / np.maximum(count, 1), float(base)


def score_map(heat, centre, radius):
    """hit (bool) and concentration (0-1) of a heat map for a nodule at `centre` (cube voxel coords)."""
    g = np.indices(heat.shape).transpose(1, 2, 3, 0) - np.asarray(centre, float)
    inside = np.sqrt((g ** 2).sum(-1)) <= radius + 2
    hot = np.unravel_index(np.argmax(heat), heat.shape)
    pos = np.clip(heat, 0, None)
    return bool(inside[hot]), float(pos[inside].sum() / max(pos.sum(), 1e-12))


def rank_corr(a, b):
    """Spearman rank correlation of two maps."""
    ra = np.argsort(np.argsort(a.ravel())).astype(float)
    rb = np.argsort(np.argsort(b.ravel())).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def random_like(model_cls, in_ch, seed=0):
    torch.manual_seed(seed)
    return model_cls(in_ch)
