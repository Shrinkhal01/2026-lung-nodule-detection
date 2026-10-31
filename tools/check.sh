#!/usr/bin/env bash
# SERVER: length + fit check in one command (2 GPUs by default). Run inside tmux. ~60 min.
#   bash tools/check.sh           # R2 and R5 trained 3x longer (30 epochs), seeds 42-44, + the comparison
#   SMOKE=1 bash tools/check.sh   # 3-minute check of the code path
set -euo pipefail
cd "$(dirname "$0")/.."
source /scratch/mtech_shrinkhal/ml-projects/venv/bin/activate
# pull first, then restart on the FRESH copy of this script
if [ "${LUNGDET_PULLED:-0}" != "1" ]; then git pull --ff-only; LUNGDET_PULLED=1 exec bash "$0" "$@"; fi
source tools/pick_gpus.sh          # asks which GPUs to use
python -m pytest tests/ -q
EP=30; EVAL=10; TAG=""; EXTRA=""
if [ "${SMOKE:-0}" = "1" ]; then EP=2; EVAL=1; TAG="_smoke"; EXTRA="--steps 10"; fi
NG=${#GPU_LIST[@]}
LOG="$SCRATCH/logs/check$TAG"; mkdir -p "$LOG"
exec 9>"$LOG/.lock"; flock -n 9 || { echo "This run is ALREADY running in another window (see: tmux ls). Not starting a second copy."; exit 1; }
jobs=("r2 42" "r2 43" "r5 42" "r5 43" "r2 44" "r5 44")   # balanced: R5 runs are ~2x slower than R2
echo "== 1/2 R2 + R5, ${EP} epochs (3x longer), seeds 42-44, on GPUs ${GPU_LIST[*]}"
for ((g=0; g<NG; g++)); do
  ( for ((n=g; n<${#jobs[@]}; n+=NG)); do
      set -- ${jobs[n]}
      CUDA_VISIBLE_DEVICES=${GPU_LIST[g]} python -u scripts/06_train_cnn.py --rung "$1" --seed "$2" --epochs $EP \
        --eval-every $EVAL --suffix _long --out check --tag "$TAG" $EXTRA > "$LOG/$1_long_$2.log" 2>&1 \
        && echo "   done $1 long seed $2 on GPU ${GPU_LIST[g]}: $(tail -1 "$LOG/$1_long_$2.log")" \
        || echo "   FAILED $1 long seed $2 (see $LOG/$1_long_$2.log)"
    done ) &
done
wait
if [ "${SMOKE:-0}" = "1" ]; then echo "smoke OK (no comparison in smoke mode)"; exit 0; fi
echo "== 2/2 comparison"; CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} python -u scripts/09_check_summary.py
