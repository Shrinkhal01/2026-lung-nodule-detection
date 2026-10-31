"""Phase 4 · S + T: occlusion heat maps (scored + sanity-checked) and the troubles gallery, on TEST.

  CUDA_VISIBLE_DEVICES=1 python scripts/11_explain.py
Uses the main model (FINAL if chosen, else R5) seed 42 and a RANDOM-weights copy for the sanity check.
Writes $RESULTS/phase4/explain/: explain_report.md, gallery_hits.png, gallery_false_alarms.png, gallery_misses.png
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.discard.cnn import Small3DNet  # noqa: E402
from lungdet.discard.hessian import hessian_channels  # noqa: E402
from lungdet.discard.models import load_ensemble, model_specs  # noqa: E402
from lungdet.explain.occlusion import occlusion_map, rank_corr, score_map  # noqa: E402

N_SCORE, N_SHOW = 40, 8


def gallery(items, path, title, blob_fn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 3 if blob_fn else 2
    fig, axes = plt.subplots(len(items), cols, figsize=(3 * cols, 3 * len(items)))
    axes = np.atleast_2d(axes)
    for ax, it in zip(axes, items):
        k = 16
        ax[0].imshow(it["cube"][:, :, k], cmap="gray", vmin=-1000, vmax=400)
        ax[0].set_title(it["label"], fontsize=7)
        ax[1].imshow(it["cube"][:, :, k], cmap="gray", vmin=-1000, vmax=400)
        ax[1].imshow(np.clip(it["heat"][:, :, k], 0, None), cmap="hot", alpha=0.5)
        ax[1].set_title(f"occlusion (p={it['p']:.2f})", fontsize=7)
        if blob_fn:
            ax[2].imshow(it["blob"][:, :, k], cmap="viridis", vmin=0, vmax=1)
            ax[2].set_title("blobness channel", fontsize=7)
        for a in ax:
            a.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    cfg = load_config()
    out = os.path.join(cfg["paths"]["results"], "phase4", "explain")
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs, _ = model_specs(cfg)
    main_name = "FINAL" if "FINAL" in specs else "R5"
    rung, suffix, seeds, in_ch = specs[main_name]
    models, tf = load_ensemble(cfg, (rung, suffix, seeds[:1], in_ch), dev)
    model = models[0]
    torch.manual_seed(0)
    rnd = Small3DNet(in_ch).to(dev).eval()                     # sanity check: same architecture, random weights
    sig = tuple(cfg["hush"]["hessian_sigmas_mm"])

    pdir = cfg["paths"]["patches"]
    tm = pd.read_parquet(os.path.join(pdir, "test_meta.parquet")).set_index("row")
    tm["p"] = pd.read_parquet(os.path.join(cfg["paths"]["results"], "phase4", f"scores_test_{main_name}.parquet")).set_index("row").prob
    cubes = np.load(os.path.join(pdir, "test_cubes.npy"), mmap_mode="r")
    nod = pd.read_csv(os.path.join(cfg["paths"]["tables"], "nodules.csv"))
    nod["gid"] = np.arange(len(nod))
    nod = nod.set_index("gid")

    hits = tm[tm.gt_id >= 0].sort_values("p", ascending=False).drop_duplicates("gt_id").head(N_SCORE)
    fas = tm[(tm.gt_id < 0) & ~tm.ignore].sort_values("p", ascending=False).head(N_SHOW)
    misses = tm[tm.gt_id >= 0].sort_values("p", ascending=False).drop_duplicates("gt_id").sort_values("p").head(N_SHOW)

    def explain(row_id, r, with_score):
        cube = torch.from_numpy(np.array(cubes[row_id]))
        heat, p = occlusion_map(model, cube, transform=tf, device=dev)
        it = {"cube": cube.numpy(), "heat": heat, "p": p}
        if in_ch == 3:
            x = ((cube.float().clamp(-1000, 400) + 1000) / 1400)[None, None]
            it["blob"] = hessian_channels(x, sig)[0, 1].numpy()
        if with_score:
            n = nod.loc[r.gt_id]
            centre = 16 + np.array([n.mm_i - r.i, n.mm_j - r.j, n.mm_k - r.k])
            radius = n.diameter_mm / 2
            heat_r, _ = occlusion_map(rnd, cube, transform=tf, device=dev)
            it.update(zip(["hit", "conc"], score_map(heat, centre, radius)))
            it.update(zip(["hit_rnd", "conc_rnd"], score_map(heat_r, centre, radius)))
            it["corr"] = rank_corr(heat, heat_r)
            it["label"] = f"nodule {n.diameter_mm:.0f} mm, {int(n.n_readers)} readers"
        return it

    scored = [explain(rid, r, True) for rid, r in hits.iterrows()]
    sdf = pd.DataFrame([{k: v for k, v in s.items() if k in ("hit", "conc", "hit_rnd", "conc_rnd", "corr", "p")} for s in scored])
    gallery(scored[:N_SHOW], os.path.join(out, "gallery_hits.png"), f"{main_name}: top-scoring real nodules (TEST)", in_ch == 3)
    fa_items = [explain(rid, r, False) | {"label": f"false alarm, blob {r.sigma:.1f} mm"} for rid, r in fas.iterrows()]
    gallery(fa_items, os.path.join(out, "gallery_false_alarms.png"), f"{main_name}: strongest false alarms (TEST)", in_ch == 3)
    miss_items = []
    for rid, r in misses.iterrows():
        n = nod.loc[r.gt_id]
        miss_items.append(explain(rid, r, False) | {"label": f"MISSED nodule {n.diameter_mm:.0f} mm, {int(n.n_readers)} readers"})
    gallery(miss_items, os.path.join(out, "gallery_misses.png"), f"{main_name}: lowest-scoring real nodules (TEST)", in_ch == 3)

    lines = [f"# Phase 4 · explanations on TEST (`scripts/11_explain.py`, {main_name} seed {seeds[0]})", "",
             f"Occlusion maps on the **{len(sdf)}** top-scoring real nodules (8 mm block of normal lung, stride 4 mm).", "",
             "| | trained model | random-weights model (sanity check) |", "|---|:-:|:-:|",
             f"| hottest voxel on the nodule (hit) | **{100 * sdf.hit.mean():.0f}%** | {100 * sdf.hit_rnd.mean():.0f}% |",
             f"| share of heat on the nodule (concentration) | **{sdf.conc.mean():.2f}** | {sdf.conc_rnd.mean():.2f} |", "",
             f"- rank correlation trained vs random map: median **{sdf.corr.median():.2f}** "
             f"(near 0 = the map depends on what the model learned → sanity check **{'PASSED' if abs(sdf.corr.median()) < 0.3 else 'FAILED'}**)",
             "", "![hits](gallery_hits.png)", "", "![false alarms](gallery_false_alarms.png)", "", "![misses](gallery_misses.png)"]
    open(os.path.join(out, "explain_report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:9]))


if __name__ == "__main__":
    main()
