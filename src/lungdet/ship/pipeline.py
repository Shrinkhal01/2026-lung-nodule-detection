"""The whole FINDERS chain for ONE scan, with the exact settings the TEST numbers were measured with.

  load      DICOM folder (needs pydicom)  or  cached .npz from scripts/03_isolate.py (already 1 mm)
  I solate  clip HU -> 1 mm voxels -> lung mask -> crop            (= scripts/03_isolate.py)
  N ominate LoG blobs inside the mask, per-size top-q              (= scripts/04_nominate.py)
  D iscard  32 mm cubes -> 3 channels [HU, blobness, vesselness] -> 5-seed 3D CNN ensemble (HUSH = R5)
  select    probability >= threshold of the chosen operating point (false alarms per scan, picked on VAL)
  merge     one finding per blob (NOT in the TEST numbers: TEST counted duplicates on one nodule once)
Torch is imported only where a model is needed, so the classical steps run (and are tested) without it.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from lungdet.discard.patches import crop_cubes
from lungdet.isolate.geometry import resample
from lungdet.isolate.lungmask import crop_box, lung_mask
from lungdet.isolate.padding import padding_to_air
from lungdet.nominate.blobs import bundle, log_peaks

ROOT = Path(__file__).resolve().parents[3]
RATES = (0.125, 0.25, 0.5, 1, 2, 4, 8)


# ---------------------------------------------------------------- I · isolate
def prepare(vol_hu, spacing, cfg):
    """Raw HU volume (rows, cols, slices) + spacing (mm) -> dict(hu int16, mask, offset, ...) in the 1 mm grid."""
    iso = cfg["isolate"]
    lo_hu, hi_hu = iso["clip_hu"]
    vol, pad = padding_to_air(np.asarray(vol_hu, np.float32), air=lo_hu)
    vol = np.clip(vol, lo_hu, hi_hu)
    full = resample(vol, spacing, iso["spacing_mm"], order=1, cval=lo_hu)
    mask = lung_mask(full, iso["lung_threshold_hu"], iso["closing_radius_mm"], iso["keep_ratio"])
    lo, hi = crop_box(mask, int(iso["crop_margin_mm"]))
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    return _pack(np.rint(full[sl]).astype(np.int16), mask[sl], lo, spacing, full.shape, "volume")


def _pack(hu, mask, offset, spacing, shape_1mm, source):
    return {"hu": hu, "mask": mask, "offset": np.asarray(offset, int), "spacing_in": np.asarray(spacing, float),
            "shape_1mm": tuple(int(s) for s in shape_1mm), "source": source,
            "lung_L": float(mask.sum() / 1e6),
            "lung_median_hu": float(np.median(hu[mask])) if mask.any() else float("nan")}


def load_scan(path, cfg):
    """DICOM folder -> prepare();  .npz cache (hu, mask, offset, spacing_in, shape_1mm) -> used as it is."""
    path = Path(path)
    if path.is_dir():
        from lungdet.fetch.dicom_io import load_dicom_series
        vol, spacing, info = load_dicom_series(path)
        prep = prepare(vol, spacing, cfg)
        prep["dicom"] = info
        return prep
    if path.suffix == ".npz":
        z = np.load(path)
        return _pack(z["hu"], z["mask"], z["offset"], z["spacing_in"], z["shape_1mm"], "cache")
    raise ValueError(f"{path}: give a DICOM folder or a cached .npz from scripts/03_isolate.py")


# ---------------------------------------------------------------- N · nominate
def nominate(prep, cfg):
    """LoG candidates, per-size top-q (the Phase 2 choice). Columns i,j,k = mm in the full 1 mm grid."""
    nom = cfg["nominate"]
    cols = ["i", "j", "k", "sigma", "response", "scale_rank"]
    if not prep["mask"].any():
        return pd.DataFrame(columns=cols)
    peaks = log_peaks(prep["hu"], prep["mask"], nom["sigmas_mm"], nom["floor"], nom["min_hu"])
    c = bundle(peaks, nom["max_per_scan"], mode="blob")            # strongest first, as in 04_nominate.py
    if len(c) == 0:
        return pd.DataFrame(columns=cols)
    xyz = (c[:, :3] + prep["offset"]).astype(int)
    df = pd.DataFrame({"i": xyz[:, 0], "j": xyz[:, 1], "k": xyz[:, 2], "sigma": c[:, 3], "response": c[:, 4]})
    df["scale_rank"] = df.groupby("sigma").cumcount()
    return df[df.scale_rank < int(cfg["ship"]["per_size_top"])].reset_index(drop=True)


# ---------------------------------------------------------------- D · discard
def resolve_models_dir(cfg, models_dir=None):
    for cand in (models_dir, os.environ.get("LUNGDET_MODELS"), ROOT / "models", cfg["paths"].get("checkpoints")):
        if cand and (Path(cand) / f"{cfg['ship']['model_rung']}_seed{cfg['ship']['seeds'][0]}.pth").exists():
            return Path(cand)
    raise FileNotFoundError("model weights not found. Pass --models-dir, set LUNGDET_MODELS, or download "
                            "lungdet-v1.0-hush-r5-weights.zip from the GitHub Release into ./models/")


def load_models(cfg, device, models_dir=None, n_models=None):
    """The HUSH (R5) ensemble. n_models < 5 is faster but NOT what was measured on TEST."""
    from lungdet.discard.models import load_ensemble
    seeds = list(cfg["ship"]["seeds"])[: n_models or None]
    paths = {**cfg["paths"], "checkpoints": str(resolve_models_dir(cfg, models_dir))}
    return load_ensemble({**cfg, "paths": paths}, (cfg["ship"]["model_rung"], "", seeds, 3), device)


def score(prep, cand, models, transform, device, cfg, progress=None):
    """Probability per candidate (mean of the ensemble). Cubes are cut once, scored in small batches."""
    import torch
    from lungdet.discard.models import ensemble_predict
    if len(cand) == 0:
        return np.zeros(0, np.float32)
    centres = cand[["i", "j", "k"]].to_numpy() - prep["offset"]
    cubes = torch.from_numpy(crop_cubes(prep["hu"], centres, size=int(cfg["discard"]["cube_mm"])))
    batch = cfg["hush"]["infer_batch"] if str(device).startswith("cuda") else cfg["ship"]["cpu_batch"]
    out = []
    for s in range(0, len(cubes), batch * 4):
        out.append(ensemble_predict(models, transform, cubes[s:s + batch * 4], device, batch=batch))
        if progress:
            progress(min(s + batch * 4, len(cubes)) / len(cubes))
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------- select + merge
def load_operating_points(cfg):
    """Search order: $LUNGDET_OPERATING_POINTS, the committed copy in the repo, then the server's results folder."""
    for f in (os.environ.get("LUNGDET_OPERATING_POINTS"), ROOT / cfg["ship"]["operating_points"],
              Path(cfg["paths"]["results"]) / "phase5" / "operating_points.json"):
        if f and Path(f).exists():
            return json.load(open(f))
    return None


def threshold_for(cfg, fp_per_scan=None, threshold=None):
    """(probability threshold, note). An explicit threshold wins; else VAL-picked thresholds; else 0.5 with a warning."""
    if threshold is not None:
        return float(threshold), "explicit threshold"
    op = load_operating_points(cfg)
    rate = fp_per_scan if fp_per_scan is not None else cfg["ship"]["default_fp_per_scan"]
    if op is None:
        return 0.5, ("results/phase5/operating_points.json is missing -> using 0.5, NOT a measured operating point "
                     "(make it on the server with scripts/12_operating_points.py)")
    key = str(rate if not float(rate).is_integer() else int(rate))
    if key not in op["points"]:
        raise ValueError(f"--fp-per-scan must be one of {sorted(op['points'], key=float)}")
    p = op["points"][key]
    return float(p["threshold"]), f"VAL operating point {key} FP/scan (VAL sensitivity {p['val_sensitivity']:.3f})"


def merge(det, min_mm=3.0):
    """Greedy: strongest first; weaker detections inside max(min_mm, sqrt(3)*sigma) of a kept one are absorbed."""
    if len(det) == 0:
        return det.assign(n_merged=pd.Series(dtype=int))
    det = det.sort_values("prob", ascending=False).reset_index(drop=True)
    xyz = det[["i", "j", "k"]].to_numpy(float)
    tree = cKDTree(xyz)
    alive = np.ones(len(det), bool)
    keep, merged = [], []
    for n in range(len(det)):
        if not alive[n]:
            continue
        near = np.array(tree.query_ball_point(xyz[n], max(min_mm, np.sqrt(3) * det.sigma[n])), int)
        near = near[alive[near]]
        alive[near] = False
        keep.append(n)
        merged.append(len(near) - 1)
    out = det.iloc[keep].copy()
    out["n_merged"] = merged
    return out.reset_index(drop=True)


def finalise(prep, cand, prob, cfg, fp_per_scan=None, threshold=None):
    """Scored candidates -> the findings table at one operating point."""
    thr, note = threshold_for(cfg, fp_per_scan, threshold)
    c = cand.assign(prob=prob)
    det = merge(c[c.prob >= thr], cfg["ship"]["merge_min_mm"])
    sp = prep["spacing_in"]
    if len(det):
        det["diameter_mm_est"] = (3.5 * det.sigma).round(1)           # LoG: diameter ~ 3.5 x sigma (Phase 2)
        vox = det[["i", "j", "k"]].to_numpy() / sp
        det["row_vox"], det["col_vox"], det["slice_vox"] = (np.rint(vox[:, a]).astype(int) for a in range(3))
        det.insert(0, "rank", np.arange(1, len(det) + 1))
    return det, thr, note


def run_scan(path, cfg, device="cpu", models_dir=None, n_models=None, fp_per_scan=None, threshold=None, progress=None):
    """Everything for one scan. Returns a dict with prep, all scored candidates, findings and timings."""
    t = {}
    t0 = time.time()
    prep = load_scan(path, cfg)
    t["isolate_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    cand = nominate(prep, cfg)
    t["nominate_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    models, tf = load_models(cfg, device, models_dir, n_models)
    prob = score(prep, cand, models, tf, device, cfg, progress)
    t["discard_s"] = round(time.time() - t0, 1)
    det, thr, note = finalise(prep, cand, prob, cfg, fp_per_scan, threshold)
    return {"prep": prep, "cand": cand.assign(prob=prob), "det": det, "threshold": thr, "threshold_note": note,
            "n_models": len(models), "timings": t, "models": models, "transform": tf}
