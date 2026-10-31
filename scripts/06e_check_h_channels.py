"""Phase 3b · verify H: do the blobness / vesselness channels carry signal on REAL cubes, and does R5 use them?

  python scripts/06e_check_h_channels.py          # CPU only, ~1-2 min, safe while GPUs train

1. On 500 nodule + 1,500 non-nodule VAL cubes: channel values at the cube centre and over the whole cube.
2. R5 seed 42 first layer: weight size (L2 norm) per input channel [HU, blobness, vesselness].
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.discard.cnn import to_unit  # noqa: E402
from lungdet.discard.hessian import hessian_channels  # noqa: E402


def main():
    torch.set_num_threads(16)
    cfg = load_config()
    pdir = cfg["paths"]["patches"]
    vm = pd.read_parquet(os.path.join(pdir, "val_meta.parquet"))
    rng = np.random.default_rng(0)
    pos = vm[vm.label == 1].row.to_numpy()
    neg = vm[(vm.label == 0) & ~vm.ignore].row.to_numpy()
    pick = {"nodule": np.sort(rng.choice(pos, min(500, len(pos)), replace=False)),
            "not nodule": np.sort(rng.choice(neg, 1500, replace=False))}
    cubes = np.load(os.path.join(pdir, "val_cubes.npy"), mmap_mode="r")
    print("| cubes | channel | centre mean | centre 90th pct | whole-cube mean | share of voxels > 0.05 |")
    print("|---|---|---|---|---|---|")
    for name, rows in pick.items():
        x = hessian_channels(to_unit(torch.from_numpy(np.array(cubes[rows]))), tuple(cfg["hush"]["hessian_sigmas_mm"]))
        for ch, label in ((1, "blobness"), (2, "vesselness")):
            c = x[:, ch, 16, 16, 16]
            print(f"| {name} (n={len(rows)}) | {label} | {c.mean():.3f} | {c.quantile(0.9):.3f} | "
                  f"{x[:, ch].mean():.3f} | {(x[:, ch] > 0.05).float().mean():.3f} |")
    ck = os.path.join(cfg["paths"]["checkpoints"], f"r5_seed{cfg['discard']['seeds'][0]}.pth")
    w = torch.load(ck, map_location="cpu")["features.0.0.weight"]          # (32, 3, 3, 3, 3)
    norms = w.flatten(2).norm(dim=2).mean(0)
    print(f"\nR5 first-layer weight size per input channel: HU {norms[0]:.3f} | blobness {norms[1]:.3f} | vesselness {norms[2]:.3f}")


if __name__ == "__main__":
    main()
