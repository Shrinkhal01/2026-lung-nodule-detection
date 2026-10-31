"""Phase 5 · PPT figures. Reads the committed result files, writes docs/figures/*.png (1600 x 900 px, slide-ready).

  python scripts/13_make_figures.py            # needs only numpy, pandas, matplotlib (no GPU, no data)
Numbers that live only in a Markdown report are typed below WITH the file they came from.
Colours: blue = R2 baseline, orange = R5 HUSH (the final model), grey = no learning / context. Every mark is labelled directly.
"""
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "docs" / "figures"
BLUE, ORANGE, GREY, INK, INK2, SURF = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.figsize": (16, 9), "figure.dpi": 100, "savefig.dpi": 100, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "font.size": 20, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 26, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.titlepad": 18})
TEST = json.load(open(RES / "phase4" / "test_metrics.json"))


def save(fig, name, title=None, tight=True):
    OUT.mkdir(parents=True, exist_ok=True)
    if title:
        fig.suptitle(title, x=0.02, y=0.975, ha="left", va="top", fontsize=27, fontweight="bold")
    if tight:
        fig.tight_layout(rect=(0, 0, 1, 0.9 if title else 1))
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def fig_pipeline():
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    # numbers: results/phase1/isolate_report.md, results/phase2/nominate_choice.json, results/phase4/test_metrics.json
    boxes = [("F · I", "Fetch + Isolate", "897 CT scans\n1 mm voxels + lung mask\n99.6% of nodules inside"),
             ("N", "Nominate", "blob detector\n~20,860 peaks per scan\n-> 2,643 kept\n96.0% recall (TRAIN)"),
             ("D", "Discard", "3D CNN on 32 mm cubes\n+ blobness / vesselness\n5-seed ensemble (HUSH)"),
             ("E · R", "Explain + Rate", "occlusion maps, checked\nFROC + 95% CI\nTEST opened once")]
    w, gap, x0 = 21.0, 3.6, 1.6
    for n, (tag, title, body) in enumerate(boxes):
        x = x0 + n * (w + gap)
        hero = n == 2
        ax.add_patch(plt.Rectangle((x, 12), w, 30, fc="#fff3ec" if hero else "#f1f0ed", ec=ORANGE if hero else GREY, lw=3))
        ax.text(x + w / 2, 38, tag, ha="center", va="center", fontsize=34, fontweight="bold", color=ORANGE if hero else INK2)
        ax.text(x + w / 2, 31.5, title, ha="center", va="center", fontsize=22, fontweight="bold")
        ax.text(x + w / 2, 21.5, body, ha="center", va="center", fontsize=16, color=INK2, linespacing=1.55)
        if n < 3:
            ax.annotate("", xy=(x + w + gap - 0.4, 27), xytext=(x + w + 0.4, 27),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=3))
    ax.text(1.6, 51, "The pipeline (FINDERS), with the number each stage earned", fontsize=27, fontweight="bold")
    ax.text(1.6, 5, f"Result on 135 unseen test scans:  CPM {TEST['R5']['cpm']:.3f}  [{TEST['R5']['ci'][0]:.3f} to {TEST['R5']['ci'][1]:.3f}]"
                  f"\n{100 * TEST['R5']['sens_at']['1']:.1f}% of nodules found at 1 false alarm per scan",
            fontsize=21, color=INK, linespacing=1.6)
    save(fig, "fig1_pipeline.png", tight=False)


def fig_staircase():
    # VAL, ensemble CPM + seed mean/SD. R0-R4: results/phase3/staircase_val.csv. R5-R7: results/phase3b/phase3b_report.md
    rows = [("R0  blob strength only", 0.088, None, None), ("R1  XGBoost features", 0.275, None, None),
            ("R2  3D CNN (baseline)", 0.778, 0.760, 0.013), ("R3  + focal, hard negatives, TTA", 0.717, 0.711, 0.010),
            ("R4  R2 + XGBoost fused", 0.780, None, None), ("R5  + Hessian channels = HUSH", 0.779, 0.768, 0.012),
            ("R6  + masked pretraining", 0.776, 0.762, 0.008), ("R7  + size head", 0.772, 0.751, 0.018)]
    fig, ax = plt.subplots()
    y = np.arange(len(rows))[::-1]
    for yy, (lab, cpm, m, sd) in zip(y, rows):
        col = ORANGE if lab.startswith("R5") else BLUE if lab.startswith("R2") else GREY
        ax.barh(yy, cpm, color=col, height=0.62)
        ax.text(cpm + 0.012, yy + (0.17 if m else 0), f"{cpm:.3f}", va="center", fontsize=19, fontweight="bold")
        if m:
            ax.errorbar(m, yy - 0.22, xerr=sd, fmt="o", color=INK, ms=7, capsize=5, lw=2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("VAL CPM.  Bar = 5-seed ensemble.  Dot + whisker = single-seed mean +/- SD", fontsize=18)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig2_staircase_val.png", "After R2 (the 3D CNN), no change beats seed noise")


def fig_froc():
    # the 7 measured points CPM is built from (results/phase4/test_metrics.json); lines only join them for the eye
    fig, ax = plt.subplots()
    rates = [float(k) for k in TEST["R5"]["sens_at"]]
    for name, col, lab, dy in (("R0", GREY, "R0  no learning", 0.06), ("R2", BLUE, "R2  3D CNN", -0.15), ("R5", ORANGE, "R5  HUSH (final)", 0.095)):
        sens = list(TEST[name]["sens_at"].values())
        ax.plot(rates, sens, "o-", color=col, lw=4 if name == "R5" else 3, ms=11)
        ax.text(8.6, sens[-1] + dy, f"{lab}   CPM {TEST[name]['cpm']:.3f}", color=col if name != "R0" else INK2,
                ha="right", fontsize=19, fontweight="bold")
    ax.axvline(1, color=INK2, lw=1.5, ls="--")
    ax.text(1.06, 0.3, "1 false alarm per scan", color=INK2, fontsize=17)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0, 1.12)
    ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4, 8], ["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    ax.minorticks_off()
    ax.axhline(TEST["ceiling"], color=INK2, lw=1.2, ls=":")
    ax.text(0.105, TEST["ceiling"] + 0.012, f"ceiling {TEST['ceiling']:.3f}: nodules the blob detector reached", color=INK2, fontsize=16)
    ax.set_xlabel("false alarms per scan (the 7 points CPM averages)")
    ax.set_ylabel("share of nodules found")
    save(fig, "fig3_froc_test.png", "TEST (135 scans, 225 nodules, opened once)")


def fig_cpm_ci():
    fig, ax = plt.subplots()
    for yy, (name, col, lab) in zip((2, 1, 0), (("R0", GREY, "R0  no learning"), ("R2", BLUE, "R2  3D CNN"), ("R5", ORANGE, "R5  HUSH (final)"))):
        m, (lo, hi) = TEST[name]["cpm"], TEST[name]["ci"]
        ax.errorbar(m, yy, xerr=[[m - lo], [hi - m]], fmt="o", color=col, ms=16, capsize=10, lw=4)
        ax.text(hi + 0.015, yy, f"{m:.3f}   [{lo:.3f} to {hi:.3f}]", va="center", fontsize=21, fontweight="bold")
    ax.set_yticks([2, 1, 0], ["R0  no learning", "R2  3D CNN", "R5  HUSH (final)"])
    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.7, 2.6)
    ax.set_xlabel("TEST CPM with 95% CI (1,000 bootstrap resamples of test scans)")
    d = TEST["diff_main_minus_r2"]
    ax.text(0.0, -0.55, f"R5 minus R2 = {d['value']:+.3f}, 95% CI {d['ci'][0]:+.3f} to {d['ci'][1]:+.3f}:  small, weakly supported\n(on VAL they tied)",
            fontsize=18, color=INK2, va="center")
    ax.tick_params(axis="y", length=0)
    save(fig, "fig4_test_cpm_ci.png", "R2 and R5 overlap heavily: read the error bars")


def fig_by_size():
    # results/phase4/test_report.md (R5, TEST): nodules, candidate recall, found at 1 FP/scan
    sizes = ["3-6 mm", "6-10 mm", "10-20 mm", "20-30 mm", "> 30 mm"]
    n = [27, 107, 68, 18, 5]
    cand = [0.889, 0.963, 1.0, 1.0, 1.0]
    found = [0.63, 0.85, 0.897, 0.889, 0.6]
    x = np.arange(5)
    fig, ax = plt.subplots()
    ax.bar(x - 0.2, cand, 0.38, color=GREY, label="reached by the blob detector")
    ax.bar(x + 0.2, found, 0.38, color=ORANGE, label="still found at 1 false alarm per scan")
    for xx, c, f in zip(x, cand, found):
        ax.text(xx - 0.2, c + 0.015, f"{c:.2f}", ha="center", fontsize=17, color=INK2)
        ax.text(xx + 0.2, f + 0.015, f"{f:.2f}", ha="center", fontsize=19, fontweight="bold")
    ax.set_xticks(x, [f"{s}\nn = {k}" for s, k in zip(sizes, n)])
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("share of nodules")
    ax.legend(frameon=False, loc="upper left", ncol=1, fontsize=18)
    save(fig, "fig5_recall_by_size.png", "Small nodules are the weak spot (> 30 mm: only 5 cases)")


def fig_ablation():
    # results/phase3b/phase3b_report.md: R5, 1 seed, VAL CPM when a channel is blanked at test time
    labels = ["all 3 channels", "no vesselness", "no blobness", "HU only"]
    vals = [0.776, 0.695, 0.663, 0.397]
    fig, ax = plt.subplots()
    ax.barh(np.arange(4)[::-1], vals, color=[ORANGE, GREY, GREY, GREY], height=0.6)
    for yy, v in zip(np.arange(4)[::-1], vals):
        ax.text(v + 0.012, yy, f"{v:.3f}", va="center", fontsize=21, fontweight="bold")
    ax.set_yticks(np.arange(4)[::-1], labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("VAL CPM when the channel is blanked at test time (1 seed)\nCaveat: a blanked channel is an input the network never saw in training.", fontsize=17)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig6_ablation.png", "R5 relies on the shape channels, not raw HU alone")


def fig_funnel():
    # ~20,860 stored candidates/scan (docs/DECISIONS.md, Phase 2); 2,643 chosen (results/phase2/nominate_choice.json); 1 false alarm/scan at the end
    labels = ["blob detector peaks", "kept (per-size top-1200)\n96.0% of TRAIN nodules inside", "shown at 1 false alarm / scan\n83.6% of TEST nodules found"]
    vals = [20860, 2643, 1]
    notes = ["", "", ""]
    fig, ax = plt.subplots()
    y = [2, 1, 0]
    ax.barh(y, vals, color=[GREY, GREY, ORANGE], height=0.55)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 400_000)
    for yy, v, nt in zip(y, vals, notes):
        ax.text(v * 1.25, yy, f"{v:,}" + (f"   {nt}" if nt else ""), va="center", fontsize=20, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.set_xlabel("candidates per scan (log scale)")
    ax.tick_params(axis="y", length=0)
    save(fig, "fig7_funnel.png", "From ~20,000 candidates per scan to about one, keeping the nodule")


def copy_galleries():
    OUT.mkdir(parents=True, exist_ok=True)
    for n in ("hits", "false_alarms", "misses"):
        src = RES / "phase4" / "explain" / f"gallery_{n}.png"
        if src.exists():
            shutil.copy(src, OUT / f"fig8_gallery_{n}.png")
            print("copied", src.name)


if __name__ == "__main__":
    for f in (fig_pipeline, fig_staircase, fig_froc, fig_cpm_ci, fig_by_size, fig_ablation, fig_funnel, copy_galleries):
        f()
