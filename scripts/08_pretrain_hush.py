"""Phase 3b · U in HUSH: self-supervised pretraining by masked cube restoration (no labels).

  CUDA_VISIBLE_DEVICES=1 python scripts/08_pretrain_hush.py [--tag _smoke --steps 20 --limit 10]

1. Cut pretrain_per_scan unlabelled candidate cubes from every TRAIN scan (once; reused if present).
2. Hide ~60% of the 4^3 blocks of each cube; the encoder + a decoder learn to redraw the hidden HU.
3. Save the ENCODER only -> $CHECKPOINTS/hush_pretrain<tag>.pth (R6 and R7 start from it).
Only TRAIN scans are used, and no labels are read.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.discard.cnn import Restorer, augment, block_mask, to_unit  # noqa: E402
from lungdet.discard.hessian import hessian_channels  # noqa: E402
from lungdet.discard.store import build_store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="TRAIN scans (smoke test)")
    args = ap.parse_args()
    cfg = load_config()
    hc = cfg["hush"]
    steps = args.steps or hc["pretrain_steps"]
    pdir = cfg["paths"]["patches"] + args.tag
    out_dir = os.path.join(cfg["paths"]["results"], "phase3b" + args.tag)
    os.makedirs(out_dir, exist_ok=True)
    path, meta_path = os.path.join(pdir, "pretrain_cubes.npy"), os.path.join(pdir, "pretrain_meta.parquet")
    t0 = time.time()
    if not os.path.exists(meta_path):
        scans = pd.read_csv(os.path.join(cfg["paths"]["tables"], "scans.csv"))
        tr = scans[scans.eligible & (scans.split == "train")].sort_values("scan_id")
        if args.limit:
            tr = tr.head(args.limit)
        cand = pd.read_parquet(os.path.join(cfg["paths"]["candidates"], "candidates.parquet"),
                               columns=["scan_id", "i", "j", "k"])
        cand = cand[cand.scan_id.isin(tr.scan_id)]
        n = hc["pretrain_per_scan"]
        meta = (cand.sample(frac=1.0, random_state=0).groupby("scan_id").head(n)     # up to n random per scan
                .merge(tr[["scan_id", "series_uid"]], on="scan_id"))
        meta = build_store(meta, cfg["paths"]["cache"], path)
        meta.to_parquet(meta_path)
        print(f"pretrain store: {len(meta)} unlabelled TRAIN cubes from {meta.scan_id.nunique()} scans, {time.time() - t0:.0f} s")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, hc["max_gpu_gb"] / total))
    torch.manual_seed(0)
    cubes = torch.from_numpy(np.load(path, mmap_mode="r")[:]).to(dev)
    sig = tuple(hc["hessian_sigmas_mm"])
    model = Restorer(in_ch=3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    bs = hc["pretrain_batch"]
    log = []
    for step in range(1, steps + 1):
        model.train()
        x1 = augment(to_unit(cubes[torch.randint(len(cubes), (bs,), device=dev)]), rot90=True)
        x3 = hessian_channels(x1, sig)
        m = block_mask(bs, hc["mask_ratio"], hc["mask_block"], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pred = model(x3 * (1 - m)).float()
        err = (pred - x1) ** 2
        loss = (err * m).sum() / m.sum().clamp_min(1) + 0.1 * (err * (1 - m)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps:
            log.append({"step": step, "loss": round(loss.item(), 5), "seconds": round(time.time() - t0)})
            print(f"  step {step}/{steps}  restoration loss {loss.item():.5f}  {time.time() - t0:.0f} s")
    ckpt = os.path.join(cfg["paths"]["checkpoints"], f"hush_pretrain{args.tag}.pth")
    torch.save(model.features.state_dict(), ckpt)
    pd.DataFrame(log).to_csv(os.path.join(out_dir, "pretrain_loss.csv"), index=False)
    print(f"encoder saved -> {ckpt}  (final restoration loss {loss.item():.5f}, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
