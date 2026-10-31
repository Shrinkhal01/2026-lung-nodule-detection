#!/usr/bin/env bash
# MAC ONLY. Phase 5 check that never runs anything on the college server (no GPU, no tmux, no jobs).
# The server is only READ twice by rsync: one small parquet and ONE scan cache (~25 MB). Needs LUNG_SERVER (as tools/pull_results.sh).
#   bash tools/phase5.sh            (about 10-15 min on the Mac CPU; plug in the charger)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LUNG_SERVER:?set LUNG_SERVER first (see the top of tools/pull_results.sh)}"
SRV=/scratch/mtech_shrinkhal
SERIES=1.3.6.1.4.1.14519.5.2.1.6279.6001.693480911433291675609148051914     # LIDC-IDRI-0405, TEST split, 2 radiologist nodules (8.4 + 8.4 mm)
mkdir -p data/phase5_local results/phase5
[ -f data/phase5_local/val_meta.parquet ] || rsync -avP --bwlimit=5000 "$LUNG_SERVER:$SRV/patches/val_meta.parquet" data/phase5_local/
[ -f "data/phase5_local/$SERIES.npz" ]    || rsync -avP --bwlimit=5000 "$LUNG_SERVER:$SRV/cache_lungdet/$SERIES.npz" data/phase5_local/
echo "== 1/4 tests";                python -m pytest tests/ -q
echo "== 2/4 operating points (VAL files already on this Mac)"; python scripts/12_operating_points.py | tail -22
echo "== 3/4 predict.py on the real scan (CPU, full 5-model ensemble, timed)"
FORCE_CPU=1 python predict.py "data/phase5_local/$SERIES.npz" --fp-per-scan 1 --explain 2 --out results/phase5/smoke
echo "== 4/4 compare with the radiologists";  python scripts/14_check_scan.py "data/phase5_local/$SERIES.npz" results/phase5/smoke/findings.csv
echo; echo "Look at results/phase5/smoke/findings.png and explain_1.png. Optional demo: python app/app.py  (then type data/phase5_local/$SERIES.npz)"
