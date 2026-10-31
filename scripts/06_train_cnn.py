"""Phase 3 / 3b: train one 3D CNN rung on one GPU, then score ALL val candidates.

  CUDA_VISIBLE_DEVICES=1 python scripts/06_train_cnn.py --rung r5 --seed 42 [--tag _smoke --epochs 1 --steps 20]

Rungs (one change at a time from R2, the Phase 3 winner):
  r2  1 channel, BCE, random negatives, flips                         (Phase 3 baseline)
  r3  r2 + focal loss + hard negatives + rot90 + 8-flip TTA            (Phase 3, it hurt)
  r5  r2 + H: input = [HU, blobness, vesselness]                       (HUSH)
  r6  r5 + U: encoder starts from masked-cube-restoration pretraining  (HUSH)
  r7  r6 + S: extra head predicts log(nodule diameter) on positives    (HUSH)
Cubes live on the GPU as int16; a memory cap (hush.max_gpu_gb) keeps us polite on the shared server.
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
from lungdet.discard.cnn import Small3DNet, augment, focal_loss, predict, to_unit  # noqa: E402
from lungdet.discard.hessian import hessian_channels  # noqa: E402
from lungdet.rate.froc import froc  # noqa: E402

RUNGS = {
    "r2": dict(in_ch=1, focal=False, hard=False, rot90=False, tta=False, pretrain=False, size=False, out="phase3"),
    "r3": dict(in_ch=1, focal=True, hard=True, rot90=True, tta=True, pretrain=False, size=False, out="phase3"),
    "r5": dict(in_ch=3, focal=False, hard=False, rot90=False, tta=False, pretrain=False, size=False, out="phase3b"),
    "r6": dict(in_ch=3, focal=False, hard=False, rot90=False, tta=False, pretrain=True, size=False, out="phase3b"),
    "r7": dict(in_ch=3, focal=False, hard=False, rot90=False, tta=False, pretrain=True, size=True, out="phase3b"),
}


def gpu_cap(max_gb):
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_gb / total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", choices=list(RUNGS), required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="steps per epoch")
    ap.add_argument("--suffix", default="", help="name variant, e.g. _long (keeps the 10-epoch results intact)")
    ap.add_argument("--out", default=None, help="results sub-folder (default: phase3 / phase3b by rung)")
    ap.add_argument("--eval-every", type=int, default=0, help="also score VAL every N epochs (learning curve)")
    args = ap.parse_args()
    R = RUNGS[args.rung]
    cfg = load_config()
    dc, hc = cfg["discard"], cfg["hush"]
    epochs, steps, batch = args.epochs or dc["epochs"], args.steps or dc["steps_per_epoch"], dc["batch"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_cap(hc["max_gpu_gb"])
    pdir = cfg["paths"]["patches"] + args.tag
    out_dir = os.path.join(cfg["paths"]["results"], (args.out or R["out"]) + args.tag)
    name = f"{args.rung}{args.suffix}"
    os.makedirs(os.path.join(out_dir, "scores"), exist_ok=True)
    os.makedirs(cfg["paths"]["checkpoints"], exist_ok=True)
    sig = tuple(hc["hessian_sigmas_mm"])
    prep = (lambda x: hessian_channels(x, sig)) if R["in_ch"] == 3 else (lambda x: x)

    tables = cfg["paths"]["tables"]
    nod = pd.read_csv(os.path.join(tables, "nodules.csv"))
    nod["gid"] = np.arange(len(nod))
    meta = pd.read_parquet(os.path.join(pdir, "train_meta.parquet"))
    keep = (meta.kind != "neg_hard").to_numpy() if not R["hard"] else np.ones(len(meta), bool)
    m = meta[keep]
    cubes = torch.from_numpy(np.load(os.path.join(pdir, "train_cubes.npy"), mmap_mode="r")[m.row.to_numpy()]).to(dev)
    y = torch.tensor(m.label.to_numpy(), dtype=torch.float32, device=dev)
    diam = nod.set_index("gid").diameter_mm
    size_t = np.where((m.label == 1) & (m.gt_id >= 0), np.log(diam.reindex(m.gt_id).to_numpy()), np.nan)
    size_t = torch.tensor(size_t, dtype=torch.float32, device=dev)
    pos, neg = torch.where(y == 1)[0], torch.where(y == 0)[0]
    print(f"{args.rung} seed {args.seed}: {len(pos)} positives, {len(neg)} negatives, in_ch={R['in_ch']} on {dev}")

    model = Small3DNet(R["in_ch"], size_head=R["size"]).to(dev)
    if R["pretrain"]:
        ckpt = os.path.join(cfg["paths"]["checkpoints"], f"hush_pretrain{args.tag}.pth")
        model.features.load_state_dict(torch.load(ckpt, map_location=dev))
        print(f"  encoder initialised from {ckpt}")
    opt = torch.optim.AdamW(model.parameters(), lr=dc["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=dc["lr"], total_steps=epochs * steps)
    n_pos = int(round(batch * dc["pos_frac"]))
    tables_ = cfg["paths"]["tables"]
    vmeta = pd.read_parquet(os.path.join(pdir, "val_meta.parquet"))
    scans_ = pd.read_csv(os.path.join(tables_, "scans.csv"))
    val_scans = scans_[(scans_.split == "val") & scans_.eligible & scans_.scan_id.isin(vmeta.scan_id)]
    n_gt = ((nod.role == "gt") & nod.scan_id.isin(val_scans.scan_id)).sum()
    ib = 4096 if R["in_ch"] == 1 else hc["infer_batch"]
    vcubes, curve = None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for _ in range(steps):
            idx = torch.cat([pos[torch.randint(len(pos), (n_pos,), device=dev)],
                             neg[torch.randint(len(neg), (batch - n_pos,), device=dev)]])
            x = prep(augment(to_unit(cubes[idx]), rot90=R["rot90"]))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logit, size = model.forward_both(x)
            logit = logit.float()
            loss = focal_loss(logit, y[idx]) if R["focal"] else \
                torch.nn.functional.binary_cross_entropy_with_logits(logit, y[idx])
            if R["size"]:
                t = size_t[idx]
                ok = ~torch.isnan(t)
                if ok.any():
                    loss = loss + hc["size_weight"] * torch.nn.functional.smooth_l1_loss(size.float()[ok], t[ok])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        print(f"  epoch {ep + 1}/{epochs}  loss {tot / steps:.4f}  {time.time() - t0:.0f} s", flush=True)
        if args.eval_every and (ep + 1) % args.eval_every == 0 and ep + 1 < epochs:
            if vcubes is None:
                vcubes = torch.from_numpy(np.load(os.path.join(pdir, "val_cubes.npy")))   # stays in RAM; batches go to the GPU
            p = predict(model, vcubes, batch=ib, tta=R["tta"], transform=prep)
            c = froc(p, vmeta.gt_id, vmeta.ignore, n_gt=n_gt, n_scans=len(val_scans))["cpm"]
            curve.append({"epoch": ep + 1, "val_cpm": round(c, 4)})
            print(f"    VAL CPM at epoch {ep + 1}: {c:.3f}", flush=True)
    # fit check: how well does the model score TRAIN candidates (positives + random negatives)?
    from sklearn.metrics import roc_auc_score
    fit = (m.kind.isin(["cand_pos", "neg_random"])).to_numpy()
    p_tr = predict(model, cubes[torch.from_numpy(np.where(fit)[0]).to(dev)], batch=ib, transform=prep)
    train_auc = float(roc_auc_score(m.label.to_numpy()[fit], p_tr))
    del cubes
    torch.cuda.empty_cache()
    ckpt = os.path.join(cfg["paths"]["checkpoints"], f"{name}_seed{args.seed}{args.tag}.pth")
    torch.save(model.state_dict(), ckpt)

    if vcubes is None:
        vcubes = torch.from_numpy(np.load(os.path.join(pdir, "val_cubes.npy")))   # stays in RAM; batches go to the GPU
    scores = pd.DataFrame({"row": vmeta.row, "prob": predict(model, vcubes, batch=ib, tta=R["tta"], transform=prep)})
    first_seed = args.seed == cfg["discard"]["seeds"][0]
    if R["in_ch"] == 3 and first_seed and not args.suffix:                      # explanation: what does the network rely on?
        def zero(ch):
            def f(x):
                x = prep(x)
                x[:, ch] = 0
                return x
            return f
        scores["prob_no_blob"] = predict(model, vcubes, batch=ib, transform=zero([1]))
        scores["prob_no_vessel"] = predict(model, vcubes, batch=ib, transform=zero([2]))
        scores["prob_hu_only"] = predict(model, vcubes, batch=ib, transform=zero([1, 2]))
    scores.to_parquet(os.path.join(out_dir, "scores", f"{name}_seed{args.seed}_val.parquet"))
    lab = (vmeta.gt_id >= 0).to_numpy()
    keep_v = ~vmeta.ignore.to_numpy()
    val_auc = float(roc_auc_score(lab[keep_v], scores.prob.to_numpy()[keep_v]))
    r = froc(scores.prob, vmeta.gt_id, vmeta.ignore, n_gt=n_gt, n_scans=len(val_scans))
    summary = {"rung": name, "seed": args.seed, "epochs": epochs, "val_cpm": r["cpm"], "sens_at": r["sens_at"],
               "train_auc": train_auc, "val_auc": val_auc, "curve": curve + [{"epoch": epochs, "val_cpm": round(r["cpm"], 4)}],
               "train_seconds": round(time.time() - t0), "checkpoint": ckpt}
    json.dump(summary, open(os.path.join(out_dir, "scores", f"{name}_seed{args.seed}.json"), "w"), indent=2)
    print(f"VAL CPM {r['cpm']:.3f}  train AUC {train_auc:.4f}  val AUC {val_auc:.4f}  ({name}, seed {args.seed})")


if __name__ == "__main__":
    main()
