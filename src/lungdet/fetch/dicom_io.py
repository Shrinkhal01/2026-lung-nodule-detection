"""Read a DICOM CT series from a folder -> HU volume (rows, cols, slices) + spacing in mm.

Same axis order and spacing convention as pylidc's Scan.to_volume, so the rest of the pipeline sees no difference.
pydicom is an OPTIONAL dependency: without it we explain what to install instead of crashing.
"""
from pathlib import Path

import numpy as np

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def load_dicom_series(folder):
    """Returns (hu float32 (rows, cols, slices), (row_mm, col_mm, slice_mm), info dict).
    Picks the series with the most slices if the folder holds several."""
    if not HAS_PYDICOM:
        raise ImportError("Reading DICOM needs pydicom:  pip install pydicom   "
                          "(or pass a cached .npz made by scripts/03_isolate.py)")
    folder = Path(folder)
    series = {}
    for f in sorted(p for p in folder.rglob("*") if p.is_file()):
        try:
            ds = pydicom.dcmread(str(f), force=False)
        except Exception:  # noqa: BLE001  (not a DICOM file)
            continue
        if not hasattr(ds, "PixelData") or not hasattr(ds, "ImagePositionPatient"):
            continue
        series.setdefault(str(getattr(ds, "SeriesInstanceUID", "?")), []).append(ds)
    if not series:
        raise ValueError(f"no DICOM CT slices (with PixelData + ImagePositionPatient) found under {folder}")
    uid, slices = max(series.items(), key=lambda kv: len(kv[1]))
    if len(slices) < 10:
        raise ValueError(f"only {len(slices)} slices in the biggest series: not a chest CT volume")
    iop = np.asarray(slices[0].ImageOrientationPatient, float) if hasattr(slices[0], "ImageOrientationPatient") \
        else np.array([1, 0, 0, 0, 1, 0], float)
    normal = np.cross(iop[:3], iop[3:])
    pos = np.array([float(np.dot(normal, np.asarray(s.ImagePositionPatient, float))) for s in slices])
    order = np.argsort(pos)
    slices, pos = [slices[i] for i in order], pos[order]
    gaps = np.diff(pos)
    gaps = gaps[gaps > 1e-3]
    if len(gaps) == 0:
        raise ValueError("all slices share one position")
    slice_mm = float(np.median(gaps))
    hu = np.stack([s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1))
                   + float(getattr(s, "RescaleIntercept", 0)) for s in slices], axis=2)
    ps = [float(v) for v in slices[0].PixelSpacing]
    info = {"series_uid": uid, "n_slices": len(slices), "n_series_in_folder": len(series), "slice_mm": slice_mm,
            "irregular_slices": bool(gaps.max() - gaps.min() > 0.1 * slice_mm)}
    return hu, (ps[0], ps[1], slice_mm), info
