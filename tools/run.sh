#!/usr/bin/env bash
# SERVER: pull the newest code, then run one script with the project env.
#   bash tools/run.sh scripts/03_isolate.py --workers 64
set -euo pipefail
cd "$(dirname "$0")/.."
source /scratch/mtech_shrinkhal/ml-projects/venv/bin/activate
# pull first, then restart on the FRESH copy of this script (bash reads a script while running it;
# a pull that changes this file mid-run would mix old and new lines)
if [ "${LUNGDET_PULLED:-0}" != "1" ]; then git pull --ff-only; LUNGDET_PULLED=1 exec bash "$0" "$@"; fi
python "$@"
