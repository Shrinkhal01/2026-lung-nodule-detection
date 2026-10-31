"""Phase 0 GATE: can we read one real scan and its radiologist marks?

  python scripts/00_gate_one_scan.py                      # LIDC-IDRI-0001 (downloaded by the speed test)
  python scripts/00_gate_one_scan.py --patient LIDC-IDRI-0005

Passes when: pylidc finds the scan, its DICOM files load into a volume,
air reads about -1000 HU, and the readers' marks group into nodules.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.fetch.pylidc_compat import import_pylidc, write_pylidcrc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="LIDC-IDRI-0001")
    args = ap.parse_args()
    cfg = load_config()
    root = cfg["paths"]["lidc_dicom"]
    print(f"pylidc config -> {write_pylidcrc(root)}  (DICOM root: {root})")

    pl = import_pylidc()
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == args.patient).all()
    print(f"{args.patient}: {len(scans)} scan(s) in pylidc's database")
    scan = scans[0]
    print(f"  slice thickness {scan.slice_thickness} mm, pixel spacing {scan.pixel_spacing:.3f} mm, "
          f"{len(scan.annotations)} reader marks")

    t = time.time()
    vol = scan.to_volume(verbose=False)
    print(f"  volume {vol.shape} loaded in {time.time() - t:.1f} s")
    p1, p50, p99 = np.percentile(vol, [1, 50, 99])
    print(f"  HU percentiles 1/50/99: {p1:.0f} / {p50:.0f} / {p99:.0f}")

    clusters = scan.cluster_annotations()
    print(f"  {len(clusters)} nodule(s) after clustering the readers' marks:")
    for i, anns in enumerate(clusters):
        d = np.mean([a.diameter for a in anns])
        print(f"    nodule {i + 1}: {len(anns)} reader(s), mean diameter {d:.1f} mm")
    n_gt = sum(len(a) >= cfg["ground_truth"]["min_readers"] and
               np.mean([x.diameter for x in a]) >= cfg["ground_truth"]["min_diameter_mm"] for a in clusters)
    print(f"  -> {n_gt} count as ground truth (>= {cfg['ground_truth']['min_readers']} readers, "
          f">= {cfg['ground_truth']['min_diameter_mm']} mm)")

    ok = p1 <= -900 and len(clusters) > 0
    print("\nGATE:", "PASS" if ok else "FAIL (air should be near -1000 HU and at least one nodule should exist)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
