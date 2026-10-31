#!/usr/bin/env bash
# SERVER: the whole of Phase 3 in one command. Run inside tmux.
#   SMOKE=1 bash tools/phase3.sh     # ~5 min check: 10 scans per split, 1 short epoch, 1 seed
#   bash tools/phase3.sh             # the real run: cubes -> R1 (CPU) + R2/R3 x 5 seeds (GPUs 1-3) -> staircase
set -euo pipefail
cd "$(dirname "$0")/.."
source /scratch/mtech_shrinkhal/ml-projects/venv/bin/activate
git pull --ff-only
python -m pytest tests/ -q
TAG=""; EXTRA=""; SEEDS="42 43 44 45 46"; LIMIT=""
if [ "${SMOKE:-0}" = "1" ]; then TAG="_smoke"; EXTRA="--epochs 1 --steps 20"; SEEDS="42"; LIMIT="--limit 10"; fi
LOG="$SCRATCH/logs/phase3$TAG"; mkdir -p "$LOG"
echo "== 1/4 cubes"; python scripts/05_build_patches.py $LIMIT --tag "$TAG"
echo "== 2/4 R1 XGBoost (CPU, background) -> $LOG/r1.log"
python scripts/06b_xgb.py --tag "$TAG" > "$LOG/r1.log" 2>&1 &
R1=$!
echo "== 3/4 R2 + R3 CNNs on GPUs 1-3 -> $LOG/<rung>_<seed>.log"
jobs=(); for s in $SEEDS; do jobs+=("r2 $s" "r3 $s"); done
for g in 0 1 2; do
  ( for ((n=g; n<${#jobs[@]}; n+=3)); do
      set -- ${jobs[n]}
      CUDA_VISIBLE_DEVICES=$((g+1)) python scripts/06_train_cnn.py --rung "$1" --seed "$2" --tag "$TAG" $EXTRA \
        > "$LOG/$1_$2.log" 2>&1 && echo "   done $1 seed $2 on GPU $((g+1)): $(tail -1 "$LOG/$1_$2.log")" \
        || echo "   FAILED $1 seed $2 (see $LOG/$1_$2.log)"
    done ) &
done
wait
echo "== 4/4 staircase"; python scripts/06c_summarize.py --tag "$TAG"
