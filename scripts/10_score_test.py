"""Phase 4 · T (Test once): score every TEST candidate with ONE model ensemble.

  CUDA_VISIBLE_DEVICES=1 python scripts/10_score_test.py --model R5
Writes $RESULTS/phase4/scores_test_<model>.parquet. tools/phase4.sh records that the test set was opened.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lungdet.common.config import load_config  # noqa: E402
from lungdet.discard.models import ensemble_predict, load_ensemble, model_specs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    cfg = load_config()
    specs, _ = model_specs(cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, cfg["hush"]["max_gpu_gb"] / total))
    pdir = cfg["paths"]["patches"]
    out = os.path.join(cfg["paths"]["results"], "phase4")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    tm = pd.read_parquet(os.path.join(pdir, "test_meta.parquet"))
    cubes = torch.from_numpy(np.load(os.path.join(pdir, "test_cubes.npy")))      # RAM; batches go to the GPU
    models, tf = load_ensemble(cfg, specs[args.model], dev)
    p = ensemble_predict(models, tf, cubes, dev)
    pd.DataFrame({"row": tm.row, "prob": p}).to_parquet(os.path.join(out, f"scores_test_{args.model}.parquet"))
    print(f"{args.model}: {len(models)} models scored {len(tm)} TEST candidates in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
