#!/usr/bin/env python
"""Gradio demo: pick a scan, move the false-alarm slider, scroll the slices, click a finding to see where it looked.

  python app/app.py                      # on the server, then from the Mac:  ssh -L 7860:localhost:7860 <user>@<server>
  python app/app.py --port 7861 --models 1
Research prototype, not a medical device.
"""
import argparse
import sys
import uuid
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lungdet.common.config import load_config  # noqa: E402
from lungdet.common.device import get_device  # noqa: E402

try:
    import gradio as gr
except ImportError:
    sys.exit("The demo needs Gradio:  pip install gradio")

CFG = load_config()
RATES = [0.125, 0.25, 0.5, 1, 2, 4, 8]
CACHE = OrderedDict()                       # session key -> result (big arrays stay out of gr.State)
ARGS = None
DEVICE = get_device()
BANNER = ("**Research prototype, not a medical device.** It finds what 3 of 4 LIDC radiologists agreed is a nodule "
          "(>= 3 mm); it says nothing about cancer. TEST (135 scans): CPM **0.823** [0.767-0.868]; "
          "**83.6%** of nodules at 1 false alarm per scan.")
COLS = ["rank", "prob", "diameter_mm_est", "row_vox", "col_vox", "slice_vox", "n_merged"]


def tables_dir():
    """The server's tables folder if it exists, else the copy committed in the repo (results/phase1)."""
    for d in (Path(CFG["paths"]["tables"]), ROOT / "results" / "phase1"):
        if (d / "scans.csv").exists():
            return d
    return None


def test_scans():
    """(label, npz path, scan_id) for TEST scans whose cache exists on this machine (server only)."""
    d = tables_dir()
    if d is None:
        return []
    scans = pd.read_csv(d / "scans.csv")
    out = []
    for s in scans[(scans.split == "test") & scans.eligible].sort_values("scan_id").itertuples():
        f = Path(CFG["paths"]["cache"]) / f"{s.series_uid}.npz"
        if f.exists():
            out.append((f"{s.patient_id}", str(f), int(s.scan_id)))
    return out


TEST = test_scans()
LABEL2 = {lab: (path, sid) for lab, path, sid in TEST}


def gt_for_npz(path):
    """Radiologist nodules for a cached scan, looked up by its series UID (file name). [] if unknown."""
    d = tables_dir()
    if d is None or not str(path).endswith(".npz"):
        return []
    sc = pd.read_csv(d / "scans.csv")
    sc = sc[sc.series_uid == Path(path).stem]
    return gt_for(int(sc.scan_id.iloc[0])) if len(sc) else []


def gt_for(scan_id):
    d = tables_dir()
    if d is None:
        return []
    nod = pd.read_csv(d / "nodules.csv")
    g = nod[(nod.scan_id == scan_id) & (nod.role == "gt")]
    return [(r.mm_i, r.mm_j, r.mm_k, r.diameter_mm) for r in g.itertuples()]


def _table(det):
    return det[COLS].round(3) if len(det) else pd.DataFrame(columns=COLS)


def run(choice, path_text, fp, progress=gr.Progress()):
    from lungdet.ship.pipeline import run_scan
    path_text = (path_text or "").strip()
    gt, src = [], path_text
    if not src and choice in LABEL2:
        src, sid = LABEL2[choice]
        gt = gt_for(sid)
    if src and not gt:
        gt = gt_for_npz(src)
    if not src:
        raise gr.Error("Pick a TEST scan or type a path to a DICOM folder / cached .npz")
    progress(0.05, desc="isolating lungs, finding candidates (about 1 min on the server)")
    r = run_scan(src, CFG, device=DEVICE, models_dir=ARGS.models_dir, n_models=ARGS.models, fp_per_scan=fp,
                 progress=lambda f: progress(0.3 + 0.7 * f, desc="scoring candidates"))
    r["gt"], r["source"] = gt, src
    key = uuid.uuid4().hex
    CACHE[key] = r
    while len(CACHE) > 3:
        CACHE.popitem(last=False)
    det = r["det"]
    k0 = int(np.clip(det.k.iloc[0] - r["prep"]["offset"][2], 0, r["prep"]["hu"].shape[2] - 1)) if len(det) \
        else r["prep"]["hu"].shape[2] // 2
    t = r["timings"]
    status = (f"**{len(det)} finding(s)** at threshold {r['threshold']:.3f} ({r['threshold_note']}). "
              f"{len(r['cand'])} candidates scored by {r['n_models']} model(s). "
              f"Times: isolate {t['isolate_s']} s, nominate {t['nominate_s']} s, discard {t['discard_s']} s on {DEVICE}. "
              f"Lung {r['prep']['lung_L']:.1f} L." + (f" Green dashed = {len(gt)} radiologist nodule(s)." if gt else ""))
    return (key, _table(det), gr.Slider(minimum=0, maximum=r["prep"]["hu"].shape[2] - 1, step=1, value=k0, label="axial slice"),
            slice_fig(key, k0), status)


def refilter(key, fp):
    from lungdet.ship.pipeline import finalise
    r = CACHE.get(key)
    if r is None:
        return gr.skip(), gr.skip(), gr.skip()
    c = r["cand"]
    det, thr, note = finalise(r["prep"], c.drop(columns="prob"), c.prob.to_numpy(), CFG, fp_per_scan=fp)
    r["det"], r["threshold"], r["threshold_note"] = det, thr, note
    return _table(det), f"**{len(det)} finding(s)** at threshold {thr:.3f} ({note}).", slice_fig(key, None)


def slice_fig(key, k):
    from lungdet.ship.plots import plot_slice
    r = CACHE.get(key)
    if r is None:
        return None
    if k is None:
        k = int(np.clip(r["det"].k.iloc[0] - r["prep"]["offset"][2], 0, r["prep"]["hu"].shape[2] - 1)) if len(r["det"]) \
            else r["prep"]["hu"].shape[2] // 2
    return plot_slice(r["prep"], r["det"], int(k), gt=r.get("gt"))


def explain(key, rank):
    from lungdet.ship.plots import explain_one
    r = CACHE.get(key)
    if r is None or len(r["det"]) == 0:
        raise gr.Error("Run a scan first (and make sure it has findings).")
    rows = r["det"][r["det"]["rank"] == int(rank)]
    if rows.empty:
        raise gr.Error(f"No finding #{int(rank)}.")
    fig, p = explain_one(r["prep"], next(rows.itertuples()), r["models"][0], r["transform"], DEVICE)
    return fig


def build():
    with gr.Blocks(title="Lung nodule finder (research prototype)") as demo:
        gr.Markdown("# Lung nodule finder: a hybrid, explainable 3D pipeline on LIDC-IDRI\n" + BANNER)
        key = gr.State("")
        with gr.Row():
            with gr.Column(scale=1):
                choice = gr.Dropdown([lab for lab, _, _ in TEST], label="TEST scan (server only)", value=None)
                path = gr.Textbox(label="or: path to a DICOM folder / cached .npz", placeholder="/path/to/scan")
                fp = gr.Dropdown(RATES, value=1, label="false alarms per scan (operating point, picked on VAL)")
                go = gr.Button("Find nodules", variant="primary")
                status = gr.Markdown()
                table = gr.Dataframe(label="findings (voxel coords of the ORIGINAL scan)", interactive=False)
                with gr.Row():
                    rank = gr.Number(value=1, precision=0, label="finding #")
                    ex = gr.Button("Where did it look? (occlusion)")
            with gr.Column(scale=1):
                k = gr.Slider(0, 1, step=1, value=0, label="axial slice")
                view = gr.Plot(label="red = finding, green dashed = radiologists")
                heat = gr.Plot(label="occlusion map (1 model; random-weights check in docs/REALITY_CHECK.md)")
        go.click(run, [choice, path, fp], [key, table, k, view, status])
        fp.change(refilter, [key, fp], [table, status, view])
        k.release(slice_fig, [key, k], view)
        ex.click(explain, [key, rank], heat)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 and use an SSH tunnel on the shared server")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--models", type=int, default=None, help="use only the first N of 5 ensemble members")
    ARGS = ap.parse_args()
    build().queue(max_size=4).launch(server_name=ARGS.host, server_port=ARGS.port)
