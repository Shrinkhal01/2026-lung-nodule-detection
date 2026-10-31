"""Phase 3 · step 1: cut 32 mm cubes.

  python scripts/05_build_patches.py [--limit 10 --tag _smoke]

TRAIN: a SAMPLE per scan = positive candidates + GT centres (exact + jittered)
       + neg_random random false candidates + neg_hard strongest-blob false candidates.
VAL / TEST: ALL selected candidates (scored later; TEST is not scored until Phase 4).
Writes $SCRATCH/patches<tag>/{split}_cubes.npy (int16 memmap) + {split}_meta.parquet
"""
import argparse
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.discard.patches import crop_cubes  # noqa: E402


def _write(task):
    npz, centers_full, start, path, size = task
    z = np.load(npz)
    cubes = crop_cubes(z["hu"], np.asarray(centers_full) - z["offset"], size=size)
    mm = np.load(path, mmap_mode="r+")
    mm[start:start + len(cubes)] = cubes
    mm.flush()
    return len(cubes)


def train_items(cand_scan, gt_scan, dc, rng):
    pos = cand_scan[(cand_scan.gt_id >= 0)].assign(label=1, kind="cand_pos")
    negs = cand_scan[(cand_scan.gt_id < 0) & ~cand_scan.ignore]
    hard = negs.nsmallest(dc["neg_hard_per_scan"], "global_rank").assign(label=0, kind="neg_hard")
    rest = negs.drop(hard.index)
    rnd = rest.sample(min(dc["neg_random_per_scan"], len(rest)), random_state=int(rng.integers(1 << 31))) \
        .assign(label=0, kind="neg_random")
    jit = []
    for g in gt_scan.itertuples():
        c = np.array([g.mm_i, g.mm_j, g.mm_k])
        offs = np.vstack([np.zeros(3), rng.uniform(-dc["jitter_mm"], dc["jitter_mm"], (dc["gt_jitter"], 3))])
        for o in offs:
            i, j, k = np.rint(c + o).astype(int)
            jit.append({"i": i, "j": j, "k": k, "gt_id": g.gid, "label": 1, "kind": "gt_jitter",
                        "sigma": np.nan, "response": np.nan, "global_rank": -1, "scale_rank": -1, "ignore": False})
    return pd.concat([pos, hard, rnd, pd.DataFrame(jit)], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="scans per split (smoke test)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    cfg = load_config()
    dc, tables = cfg["discard"], cfg["paths"]["tables"]
    out_dir = cfg["paths"]["patches"] + args.tag
    os.makedirs(out_dir, exist_ok=True)
    scans = pd.read_csv(os.path.join(tables, "scans.csv"))
    scans = scans[scans.eligible]
    nod = pd.read_csv(os.path.join(tables, "nodules.csv"))
    nod["gid"] = np.arange(len(nod))
    cand = pd.read_parquet(os.path.join(cfg["paths"]["candidates"], "candidates.parquet"))
    rng = np.random.default_rng(42)
    size = int(cfg["discard"]["cube_mm"])
    t0 = time.time()
    for split in ("train", "val", "test"):
        sc = scans[scans.split == split].sort_values("scan_id")
        if args.limit:
            sc = sc.head(args.limit)
        parts = []
        for s in sc.itertuples():
            cs = cand[cand.scan_id == s.scan_id]
            if split == "train":
                gts = nod[(nod.scan_id == s.scan_id) & (nod.role == "gt")]
                items = train_items(cs, gts, dc, rng)
            else:
                items = cs.assign(label=(cs.gt_id >= 0).astype(int), kind="cand")
            parts.append(items.assign(scan_id=s.scan_id, series_uid=s.series_uid, patient_id=s.patient_id))
        meta = pd.concat(parts, ignore_index=True)
        path = os.path.join(out_dir, f"{split}_cubes.npy")
        np.lib.format.open_memmap(path, mode="w+", dtype=np.int16, shape=(len(meta), size, size, size)).flush()
        tasks, start = [], 0
        for sid, g in meta.groupby("scan_id", sort=False):
            npz = os.path.join(cfg["paths"]["cache"], f"{g.series_uid.iloc[0]}.npz")
            tasks.append((npz, g[["i", "j", "k"]].to_numpy(), start, path, size))
            meta.loc[g.index, "row"] = np.arange(start, start + len(g))
            start += len(g)
        meta["row"] = meta.row.astype(np.int64)
        meta = meta.sort_values("row").reset_index(drop=True)
        with get_context("spawn").Pool(args.workers) as pool:
            n = sum(pool.imap_unordered(_write, tasks))
        meta.to_parquet(os.path.join(out_dir, f"{split}_meta.parquet"))
        k = meta.kind.value_counts().to_dict()
        print(f"{split}: {len(sc)} scans, {n} cubes ({n * size ** 3 * 2 / 2 ** 30:.1f} GiB), kinds {k}, "
              f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
