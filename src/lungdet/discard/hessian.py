"""H in HUSH: physics channels from the Hessian (curvature in 3 directions), computed on the GPU.

For each voxel, sort the 3 Hessian eigenvalues e1 <= e2 <= e3. Bright structures curve DOWN (negative):
  ball  (nodule)  e = (-100, -100, -100)  -> blobness  = -e3 = 100,  vesselness = 0
  tube  (vessel)  e = (-100, -100,    0)  -> blobness  = 0,          vesselness = -e2 = 100
  sheet (wall)    e = (-100,    0,    0)  -> both 0
blobness   = relu(-e3) * sigma^2
vesselness = relu(-e2) * sigma^2 - blobness      (max over a few sizes sigma)
"""
import math

import torch
import torch.nn.functional as F


def _gauss1d(sigma, device):
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum(), r


def blur3d(x, sigma):
    """Separable Gaussian blur of (N, 1, D, H, W), edges replicated."""
    k, r = _gauss1d(sigma, x.device)
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + dim] = len(k)
        pad = [0] * 6
        pad[2 * (2 - dim)] = pad[2 * (2 - dim) + 1] = r      # F.pad lists the LAST dimension first
        x = F.conv3d(F.pad(x, pad, mode="replicate"), k.view(shape))
    return x


def sym3_eigvals(a, b, c, d, e, f):
    """Closed-form eigenvalues of [[a,d,e],[d,b,f],[e,f,c]] per voxel. Returns (e1, e2, e3), e1 <= e2 <= e3."""
    q = (a + b + c) / 3
    p1 = d * d + e * e + f * f
    p = torch.sqrt(((a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2 * p1) / 6).clamp_min(1e-12)
    ba, bb, bc, bd, be, bf = (a - q) / p, (b - q) / p, (c - q) / p, d / p, e / p, f / p
    det = ba * (bb * bc - bf * bf) - bd * (bd * bc - bf * be) + be * (bd * bf - bb * be)
    phi = torch.acos((det / 2).clamp(-1, 1)) / 3
    e3 = q + 2 * p * torch.cos(phi)
    e1 = q + 2 * p * torch.cos(phi + 2 * math.pi / 3)
    return e1, 3 * q - e1 - e3, e3


def hessian_channels(x, sigmas=(1.0, 2.0, 4.0), gain=2.0):
    """x: (N, 1, D, H, W) in [0, 1] -> (N, 3, D, H, W) = [x, blobness, vesselness], channels 2-3 in [0, 1]."""
    x = x.float()
    blob = torch.zeros_like(x)
    vessel = torch.zeros_like(x)
    for s in sigmas:
        g = blur3d(x, s)
        gd, gh, gw = torch.gradient(g, dim=(2, 3, 4))
        hdd, hdh, hdw = torch.gradient(gd, dim=(2, 3, 4))
        hhh, hhw = torch.gradient(gh, dim=(3, 4))
        hww = torch.gradient(gw, dim=4)[0]
        e1, e2, e3 = sym3_eigvals(hdd, hhh, hww, hdh, hdw, hhw)
        b = torch.relu(-e3) * s * s
        blob = torch.maximum(blob, b)
        vessel = torch.maximum(vessel, torch.relu(-e2) * s * s - b)
    return torch.cat([x, (gain * blob).clamp(0, 1), (gain * vessel).clamp(0, 1)], dim=1)
