"""Pictures for predict.py and the Gradio demo (matplotlib only, headless-safe)."""
import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_slice(prep, det, k, title=None, gt=None):
    """Axial slice k of the cropped 1 mm volume, lung outline, a red circle for every finding crossing this slice.
    gt: optional list of (i, j, k, diameter_mm) radiologist nodules in mm, drawn as dashed green circles."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(prep["hu"][:, :, k], cmap="gray", vmin=-1000, vmax=400)
    ax.contour(prep["mask"][:, :, k], levels=[0.5], colors="#2a9d8f", linewidths=0.6)
    for r in det.itertuples():
        c = np.array([r.i, r.j, r.k]) - prep["offset"]
        rad = max(r.diameter_mm_est / 2, 2.0)
        if abs(c[2] - k) <= rad:
            ax.add_patch(plt.Circle((c[1], c[0]), rad + 3, fill=False, color="#e63946", lw=1.4))
            ax.text(c[1] + rad + 4, c[0] - rad - 4, f"#{r.rank}  p={r.prob:.2f}", color="#ffd166", fontsize=8)
    for gi, gj, gk, gd in (gt or []):
        c = np.array([gi, gj, gk]) - prep["offset"]
        if abs(c[2] - k) <= gd / 2:
            ax.add_patch(plt.Circle((c[1], c[0]), gd / 2 + 1, fill=False, color="#80ed99", lw=1.2, ls="--"))
    ax.set_title(title or f"axial slice {k}", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_overview(prep, det, max_n=12, cols=4):
    """One tile per finding: the axial slice through its centre."""
    plt = _plt()
    det = det.head(max_n)
    n = max(len(det), 1)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), det.itertuples()):
        c = np.array([r.i, r.j, r.k]) - prep["offset"]
        k = int(np.clip(c[2], 0, prep["hu"].shape[2] - 1))
        ax.imshow(prep["hu"][:, :, k], cmap="gray", vmin=-1000, vmax=400)
        ax.add_patch(plt.Circle((c[1], c[0]), max(r.diameter_mm_est / 2, 2.0) + 3, fill=False, color="#e63946", lw=1.4))
        ax.set_title(f"#{r.rank}  p={r.prob:.2f}  ~{r.diameter_mm_est:.0f} mm", fontsize=8)
    if len(det) == 0:
        axes[0, 0].text(0.5, 0.5, "no findings at this operating point", ha="center", va="center")
    fig.tight_layout()
    return fig


def explain_one(prep, row, model, transform, device, cube_mm=32):
    """Occlusion map for one finding with ONE ensemble member (as in Phase 4). Returns (figure, base probability)."""
    import torch
    from lungdet.discard.hessian import hessian_channels
    from lungdet.discard.patches import crop_cubes
    from lungdet.explain.occlusion import occlusion_map
    plt = _plt()
    centre = (np.array([row.i, row.j, row.k]) - prep["offset"]).reshape(1, 3)
    cube = torch.from_numpy(crop_cubes(prep["hu"], centre, size=cube_mm)[0])
    heat, p = occlusion_map(model, cube, transform=transform, device=device,
                            batch=512 if str(device).startswith("cuda") else 16)   # 8 GB laptops: small batches
    x = ((cube.float().clamp(-1000, 400) + 1000) / 1400)[None, None]
    blob = hessian_channels(x)[0, 1].numpy()
    k = cube_mm // 2
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    ax[0].imshow(cube.numpy()[:, :, k], cmap="gray", vmin=-1000, vmax=400)
    ax[0].set_title("32 mm cube, centre slice", fontsize=8)
    ax[1].imshow(cube.numpy()[:, :, k], cmap="gray", vmin=-1000, vmax=400)
    ax[1].imshow(np.clip(heat[:, :, k], 0, None), cmap="hot", alpha=0.5)
    ax[1].set_title(f"occlusion: p drops where it looked (p={p:.2f})", fontsize=8)
    ax[2].imshow(blob[:, :, k], cmap="viridis", vmin=0, vmax=1)
    ax[2].set_title("blobness channel", fontsize=8)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    return fig, p
