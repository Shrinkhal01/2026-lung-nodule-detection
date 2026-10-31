#!/usr/bin/env bash
# SERVER: pull the newest code, then run one script with the project env.
#   bash tools/run.sh scripts/03_isolate.py --workers 64
set -euo pipefail
cd "$(dirname "$0")/.."
source /scratch/mtech_shrinkhal/ml-projects/venv/bin/activate
git pull --ff-only
python "$@"
