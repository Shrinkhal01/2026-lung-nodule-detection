"""Write many 32 mm cubes into one int16 .npy memmap, in parallel (one worker per scan)."""
import os
from multiprocessing import get_context

import numpy as np

from lungdet.discard.patches import crop_cubes


def _write(task):
    npz, centers_full, start, path, size = task
    z = np.load(npz)
    cubes = crop_cubes(z["hu"], np.asarray(centers_full) - z["offset"], size=size)
    mm = np.load(path, mmap_mode="r+")
    mm[start:start + len(cubes)] = cubes
    mm.flush()
    return len(cubes)


def build_store(meta, cache_dir, path, size=32, workers=64):
    """meta needs scan_id, series_uid, i, j, k (full 1 mm grid). Returns meta sorted by its new 'row' column."""
    meta = meta.reset_index(drop=True).copy()
    np.lib.format.open_memmap(path, mode="w+", dtype=np.int16, shape=(len(meta), size, size, size)).flush()
    tasks, start = [], 0
    meta["row"] = -1
    for _, g in meta.groupby("scan_id", sort=False):
        npz = os.path.join(cache_dir, f"{g.series_uid.iloc[0]}.npz")
        tasks.append((npz, g[["i", "j", "k"]].to_numpy(), start, path, size))
        meta.loc[g.index, "row"] = np.arange(start, start + len(g))
        start += len(g)
    with get_context("spawn").Pool(workers) as pool:
        n = sum(pool.imap_unordered(_write, tasks))
    assert n == len(meta)
    return meta.sort_values("row").reset_index(drop=True)
