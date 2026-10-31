"""Phase 3 · R2 / R3: train the 3D CNN false-positive filter on one GPU, then score ALL val candidates.

  CUDA_VISIBLE_DEVICES=1 python scripts/06_train_cnn.py --rung r2 --seed 42 [--tag _smoke --epochs 1 --steps 20]

R2 = BCE loss, random negatives only, flips.
R3 = focal loss + hard negatives (strongest-blob false candidates) + flips + 90 deg rotations + test-time flips.
Cubes live on the GPU as int16 (train ~9 GB, val ~23 GB on a 140 GB H200), so there is no data loader.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.discard.cnn import Small3DNet, augment, focal_loss, predict  # noqa: E402
from lungdet.rate.froc import froc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", choices=["r2", "r3"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="steps per epoch")
    args = ap.parse_args()
    cfg = load_config()
    dc = cfg["discard"]
    epochs, steps, batch = args.epochs or dc["epochs"], args.steps or dc["steps_per_epoch"], dc["batch"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdir = cfg["paths"]["patches"] + args.tag
    out_dir = os.path.join(cfg["paths"]["results"], "phase3" + args.tag)
    os.makedirs(os.path.join(out_dir, "scores"), exist_ok=True)
    os.makedirs(cfg["paths"]["checkpoints"], exist_ok=True)

    meta = pd.read_parquet(os.path.join(pdir, "train_meta.parquet"))
    keep = meta.kind != "neg_hard" if args.rung == "r2" else np.ones(len(meta), bool)
    rows = meta.row[keep].to_numpy()
    cubes = torch.from_numpy(np.load(os.path.join(pdir, "train_cubes.npy"), mmap_mode="r")[rows]).to(dev)
    y = torch.tensor(meta.label[keep].to_numpy(), dtype=torch.float32, device=dev)
    pos, neg = torch.where(y == 1)[0], torch.where(y == 0)[0]
    print(f"{args.rung} seed {args.seed}: {len(pos)} positives, {len(neg)} negatives on {dev}")

    model = Small3DNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=dc["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=dc["lr"], total_steps=epochs * steps)
    n_pos = int(round(batch * dc["pos_frac"]))
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for _ in range(steps):
            idx = torch.cat([pos[torch.randint(len(pos), (n_pos,), device=dev)],
                             neg[torch.randint(len(neg), (batch - n_pos,), device=dev)]])
            x = ((cubes[idx].float().clamp(-1000, 400) + 1000) / 1400).unsqueeze(1)
            x = augment(x, rot90=(args.rung == "r3"))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logit = model(x).float()
            loss = (focal_loss(logit, y[idx]) if args.rung == "r3"
                    else torch.nn.functional.binary_cross_entropy_with_logits(logit, y[idx]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        print(f"  epoch {ep + 1}/{epochs}  loss {tot / steps:.4f}  {time.time() - t0:.0f} s")
    del cubes
    ckpt = os.path.join(cfg["paths"]["checkpoints"], f"{args.rung}_seed{args.seed}{args.tag}.pth")
    torch.save(model.state_dict(), ckpt)

    vmeta = pd.read_parquet(os.path.join(pdir, "val_meta.parquet"))
    vcubes = torch.from_numpy(np.load(os.path.join(pdir, "val_cubes.npy"), mmap_mode="r")[:]).to(dev)
    prob = predict(model, vcubes, tta=(args.rung == "r3"))
    pd.DataFrame({"row": vmeta.row, "prob": prob}).to_parquet(
        os.path.join(out_dir, "scores", f"{args.rung}_seed{args.seed}_val.parquet"))
    tables = cfg["paths"]["tables"]
    nod = pd.read_csv(os.path.join(tables, "nodules.csv"))
    scans = pd.read_csv(os.path.join(tables, "scans.csv"))
    val_scans = scans[(scans.split == "val") & scans.eligible & scans.scan_id.isin(vmeta.scan_id)]
    n_gt = ((nod.role == "gt") & nod.scan_id.isin(val_scans.scan_id)).sum()
    r = froc(prob, vmeta.gt_id, vmeta.ignore, n_gt=n_gt, n_scans=len(val_scans))
    summary = {"rung": args.rung, "seed": args.seed, "val_cpm": r["cpm"], "sens_at": r["sens_at"],
               "train_seconds": round(time.time() - t0), "checkpoint": ckpt}
    json.dump(summary, open(os.path.join(out_dir, "scores", f"{args.rung}_seed{args.seed}.json"), "w"), indent=2)
    print(f"VAL CPM {r['cpm']:.3f}  ({args.rung}, seed {args.seed})")


if __name__ == "__main__":
    main()
