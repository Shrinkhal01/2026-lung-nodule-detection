"""Phase 5 tests: a synthetic chest (body + 2 lungs + one 10 mm nodule) goes through the shipping chain on the CPU."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.ship import pipeline  # noqa: E402
from lungdet.ship.operating import operating_points  # noqa: E402

SPACING = (0.9, 0.9, 2.5)
NODULE_MM = np.array([107.0, 72.0, 75.0])        # physical position of the planted nodule (row, col, slice) in mm


def phantom():
    """160 x 160 x 60 voxels at 0.9 x 0.9 x 2.5 mm: air, a soft-tissue body, two lungs, one 5 mm-radius nodule."""
    n = (160, 160, 60)
    g = np.indices(n).astype(float) * np.array(SPACING).reshape(3, 1, 1, 1)

    def ell(c, a):
        return sum(((g[d] - c[d]) / a[d]) ** 2 for d in range(3)) <= 1

    vol = np.full(n, -1000.0, np.float32)
    vol[ell((72, 72, 75), (65, 55, 120))] = 40
    for dx in (-35, 35):
        vol[ell((72 + dx, 72, 75), (25, 40, 60))] = -850
    vol[((g[0] - NODULE_MM[0]) ** 2 + (g[1] - NODULE_MM[1]) ** 2 + (g[2] - NODULE_MM[2]) ** 2) <= 25] = 20
    return vol


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def prep(cfg):
    return pipeline.prepare(phantom(), SPACING, cfg)


def test_isolate_finds_two_lungs(prep):
    assert prep["hu"].dtype == np.int16 and prep["mask"].any()
    assert 0.2 < prep["lung_L"] < 3.0                    # a small phantom, not a real 5 L chest
    assert -950 < prep["lung_median_hu"] < -700


def test_nominate_finds_the_planted_nodule(prep, cfg):
    cand = pipeline.nominate(prep, cfg)
    assert len(cand) > 0 and (cand.scale_rank < cfg["ship"]["per_size_top"]).all()
    d = np.linalg.norm(cand[["i", "j", "k"]].to_numpy() - NODULE_MM, axis=1)
    assert d.min() <= 5.0, f"closest candidate is {d.min():.1f} mm away"


def test_cache_roundtrip(prep, cfg, tmp_path):
    f = tmp_path / "s.npz"
    np.savez_compressed(f, hu=prep["hu"], mask=prep["mask"], offset=prep["offset"],
                        spacing_in=prep["spacing_in"], shape_1mm=np.array(prep["shape_1mm"]))
    again = pipeline.load_scan(f, cfg)
    assert again["source"] == "cache" and (again["hu"] == prep["hu"]).all()
    with pytest.raises(ValueError):
        pipeline.load_scan(tmp_path / "nothing.txt", cfg)


def test_merge_keeps_strongest_and_counts_absorbed():
    det = pd.DataFrame({"i": [10, 12, 60], "j": [10, 10, 60], "k": [10, 10, 60], "sigma": [2.5, 1.0, 2.5],
                        "prob": [0.9, 0.8, 0.7]})
    out = pipeline.merge(det, 3.0)
    assert list(out.prob) == [0.9, 0.7] and list(out.n_merged) == [1, 0]
    assert len(pipeline.merge(det.iloc[:0], 3.0)) == 0


def test_threshold_picking():
    # 4 scans; false candidates scored 0.9, 0.8, 0.7, 0.6, 0.5 ...; 2 GT nodules scored 0.95 and 0.65
    prob = [0.95, 0.65, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    gt = [0, 1, -1, -1, -1, -1, -1, -1]
    pts = operating_points(prob, gt, [False] * 8, n_gt=2, n_scans=4, rates=(0.5, 1, 2))
    thr1, sens1, fp1 = pts[1]            # 1 FP/scan x 4 scans = 4 false alarms allowed -> threshold = 5th strongest FP = 0.5
    assert thr1 == 0.5 and fp1 == 1.0 and sens1 == 1.0
    thr_half, sens_half, fp_half = pts[0.5]    # 2 allowed -> threshold = 3rd strongest FP = 0.7
    assert thr_half == 0.7 and fp_half == 0.5 and sens_half == 0.5
    assert pts[2][0] == 0.0               # more false alarms allowed than exist -> keep everything


def test_threshold_for_explicit_and_missing_file(cfg, monkeypatch):
    assert pipeline.threshold_for(cfg, threshold=0.42)[0] == 0.42
    monkeypatch.setattr(pipeline, "load_operating_points", lambda c: None)
    thr, note = pipeline.threshold_for(cfg, fp_per_scan=1)
    assert thr == 0.5 and "NOT a measured operating point" in note


def test_threshold_for_reads_operating_points(cfg, monkeypatch):
    op = {"points": {"1": {"threshold": 0.31, "val_sensitivity": 0.82}, "0.5": {"threshold": 0.4, "val_sensitivity": 0.78}}}
    monkeypatch.setattr(pipeline, "load_operating_points", lambda c: op)
    assert pipeline.threshold_for(cfg, fp_per_scan=1)[0] == 0.31
    assert pipeline.threshold_for(cfg, fp_per_scan=0.5)[0] == 0.4
    with pytest.raises(ValueError):
        pipeline.threshold_for(cfg, fp_per_scan=3)


def test_dicom_series_roundtrip(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    from lungdet.fetch.dicom_io import load_dicom_series
    uid = generate_uid()
    rng = np.random.default_rng(0)
    truth = rng.integers(-1000, 400, (16, 16, 12)).astype(np.int16)
    for z in rng.permutation(12):                              # written in random order: the loader must sort by position
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(tmp_path / f"s{z}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SeriesInstanceUID = uid
        ds.ImagePositionPatient = [0, 0, 2.5 * int(z)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [0.7, 0.8]
        ds.Rows = ds.Columns = 16
        ds.SamplesPerPixel, ds.BitsAllocated, ds.BitsStored, ds.HighBit = 1, 16, 16, 15
        ds.PixelRepresentation = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        ds.PixelData = truth[:, :, z].tobytes()
        ds.save_as(str(tmp_path / f"s{z}.dcm"))
    hu, spacing, info = load_dicom_series(tmp_path)
    assert hu.shape == (16, 16, 12) and np.allclose(hu, truth)
    assert spacing == (0.7, 0.8, 2.5) and info["n_slices"] == 12 and not info["irregular_slices"]


def test_full_chain_with_random_weights(cfg, prep, tmp_path):
    torch = pytest.importorskip("torch")
    from lungdet.discard.cnn import Small3DNet
    for s in (1, 2):
        torch.manual_seed(s)
        torch.save(Small3DNet(3).state_dict(), tmp_path / f"r5_seed{s}.pth")
    cfg2 = {**cfg, "ship": {**cfg["ship"], "seeds": [1, 2], "per_size_top": 20}, "hush": {**cfg["hush"]}}
    f = tmp_path / "scan.npz"
    np.savez_compressed(f, hu=prep["hu"], mask=prep["mask"], offset=prep["offset"],
                        spacing_in=prep["spacing_in"], shape_1mm=np.array(prep["shape_1mm"]))
    r = pipeline.run_scan(f, cfg2, device="cpu", models_dir=tmp_path, threshold=0.0)
    det = r["det"]
    assert r["n_models"] == 2 and len(r["cand"]) > 0 and len(det) > 0
    assert det.prob.between(0, 1).all() and det.prob.is_monotonic_decreasing
    assert {"rank", "diameter_mm_est", "row_vox", "col_vox", "slice_vox", "n_merged"} <= set(det.columns)
    assert len(det) <= len(r["cand"])


def test_operating_points_found_through_env_var(cfg, tmp_path, monkeypatch):
    f = tmp_path / "op.json"
    f.write_text(json.dumps({"points": {"1": {"threshold": 0.77, "val_sensitivity": 0.8}}}))
    monkeypatch.setenv("LUNGDET_OPERATING_POINTS", str(f))
    assert pipeline.threshold_for(cfg, fp_per_scan=1)[0] == 0.77
