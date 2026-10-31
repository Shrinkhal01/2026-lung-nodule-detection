#!/usr/bin/env python
"""predict.py: one chest CT in -> nodule findings out.

  python predict.py /path/to/dicom_folder                      # 1 false alarm per scan (default)
  python predict.py /path/to/scan.npz --fp-per-scan 0.5 --out out/
  FORCE_CPU=1 python predict.py SCAN --models 1                # slow machine: 1 of 5 models (NOT the measured setup)

Writes <out>/findings.csv, findings.png, summary.json (settings, timings, environment). Research prototype,
not a medical device: it finds what 3 of 4 LIDC radiologists agreed is a nodule (>= 3 mm), nothing about cancer.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.common.device import env_report, get_device, set_seed  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", help="DICOM folder (needs pydicom) or a cached .npz from scripts/03_isolate.py")
    ap.add_argument("--out", default="predict_out", help="output folder (default: predict_out)")
    ap.add_argument("--fp-per-scan", type=float, choices=[0.125, 0.25, 0.5, 1, 2, 4, 8], default=None,
                    help="operating point: false alarms per scan allowed (thresholds picked on VAL); default 1")
    ap.add_argument("--threshold", type=float, default=None, help="probability threshold (overrides --fp-per-scan)")
    ap.add_argument("--models-dir", default=None, help="folder with r5_seed42..46.pth (default: ./models)")
    ap.add_argument("--models", type=int, default=None, help="use only the first N of the 5 ensemble members (faster)")
    ap.add_argument("--explain", type=int, default=0, metavar="N", help="also write occlusion maps for the top N findings")
    args = ap.parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("predict.py needs PyTorch to run the 3D CNN:  pip install torch   (CPU build is fine)")
    from lungdet.ship.pipeline import run_scan
    from lungdet.ship.plots import explain_one, plot_overview

    cfg = load_config()
    set_seed(0)
    device = get_device()
    env = env_report()
    print(f"device: {device} | models: {args.models or len(cfg['ship']['seeds'])} of {len(cfg['ship']['seeds'])}", flush=True)
    t0 = time.time()
    r = run_scan(args.scan, cfg, device=device, models_dir=args.models_dir, n_models=args.models,
                 fp_per_scan=args.fp_per_scan, threshold=args.threshold,
                 progress=lambda f: print(f"  scoring candidates {100 * f:.0f}%", end="\r", flush=True))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    det = r["det"]
    det.to_csv(out / "findings.csv", index=False)
    plot_overview(r["prep"], det).savefig(out / "findings.png", dpi=110)
    if args.explain and len(det):
        for row in det.head(args.explain).itertuples():
            fig, _ = explain_one(r["prep"], row, r["models"][0], r["transform"], device)
            fig.savefig(out / f"explain_{row.rank}.png", dpi=110)
    summary = {"scan": str(args.scan), "device": device, "ensemble_members": r["n_models"],
               "measured_setup": r["n_models"] == len(cfg["ship"]["seeds"]), "threshold": r["threshold"],
               "threshold_source": r["threshold_note"], "candidates_scored": int(len(r["cand"])),
               "findings": int(len(det)), "lung_litres": round(r["prep"]["lung_L"], 2),
               "lung_median_hu": round(r["prep"]["lung_median_hu"], 0), "timings_s": r["timings"],
               "total_s": round(time.time() - t0, 1), "env": env}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(f"\n{len(det)} finding(s) at threshold {r['threshold']:.3f}  ({r['threshold_note']})")
    if len(det):
        print(det[["rank", "prob", "diameter_mm_est", "row_vox", "col_vox", "slice_vox", "n_merged"]].to_string(index=False))
    if not summary["measured_setup"]:
        print("NOTE: fewer than 5 models -> the thresholds were calibrated for the full ensemble.")
    if not (2 <= summary["lung_litres"] <= 10):
        print(f"WARNING: lung volume {summary['lung_litres']} L is outside 2-10 L -> the lung mask may be broken; check findings.png.")
    print(f"files in {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
