#!/usr/bin/env bash
# SERVER: the whole of Phase 3b (HUSH) in one command, on 2 GPUs. Run inside tmux.
#   SMOKE=1 bash tools/phase3b.sh    # ~10 min check (tiny pretraining, 1 short epoch, 1 seed)
#   bash tools/phase3b.sh            # real run (~1.5-2 h): pretraining || R5 x5, then R6 x5 || R7 x5, then staircase
#   GPUS="1 2" is the default (fair use); GPUS="1 2 3" uses 3 GPUs (~25% faster)
set -euo pipefail
cd "$(dirname "$0")/.."
source /scratch/mtech_shrinkhal/ml-projects/venv/bin/activate
# pull first, then restart on the FRESH copy of this script (bash reads a script while running it;
# a pull that changes this file mid-run would mix old and new lines)
if [ "${LUNGDET_PULLED:-0}" != "1" ]; then git pull --ff-only; LUNGDET_PULLED=1 exec bash "$0" "$@"; fi
python -m pytest tests/ -q
TAG=""; EXTRA=""; SEEDS="42 43 44 45 46"; PRE=""
if [ "${SMOKE:-0}" = "1" ]; then TAG="_smoke"; EXTRA="--epochs 1 --steps 20"; SEEDS="42"; PRE="--steps 20 --limit 10"; fi
GPU_LIST=(${GPUS:-1 2}); G1=${GPU_LIST[0]}
LOG="$SCRATCH/logs/phase3b$TAG"; mkdir -p "$LOG"

run() {  # rung seed gpu
  CUDA_VISIBLE_DEVICES=$3 python scripts/06_train_cnn.py --rung "$1" --seed "$2" --tag "$TAG" $EXTRA > "$LOG/$1_$2.log" 2>&1 \
    && echo "   done $1 seed $2 on GPU $3: $(tail -1 "$LOG/$1_$2.log")" \
    || echo "   FAILED $1 seed $2 (see $LOG/$1_$2.log)"
}

NG=${#GPU_LIST[@]}
echo "== 1/3 U: pretraining on GPU $G1   ||   R5 (+H) x seeds on GPUs ${GPU_LIST[*]:1}"
CUDA_VISIBLE_DEVICES=$G1 python scripts/08_pretrain_hush.py --tag "$TAG" $PRE > "$LOG/pretrain.log" 2>&1 &
PRE_PID=$!
R5_GPUS=(${GPU_LIST[@]:1}); [ ${#R5_GPUS[@]} -eq 0 ] && R5_GPUS=($G1)
S=($SEEDS)
for ((g=0; g<${#R5_GPUS[@]}; g++)); do
  ( for ((n=g; n<${#S[@]}; n+=${#R5_GPUS[@]})); do run r5 "${S[n]}" "${R5_GPUS[g]}"; done ) &
done
if ! wait $PRE_PID; then echo "PRETRAINING FAILED: see $LOG/pretrain.log"; tail -5 "$LOG/pretrain.log"; wait; exit 1; fi
echo "   pretraining done: $(tail -1 "$LOG/pretrain.log")"
wait
echo "== 2/3 R6 (+U) and R7 (+S) x seeds, spread over GPUs ${GPU_LIST[*]}"
jobs=(); for s in $SEEDS; do jobs+=("r6 $s" "r7 $s"); done
for ((g=0; g<NG; g++)); do
  ( for ((n=g; n<${#jobs[@]}; n+=NG)); do set -- ${jobs[n]}; run "$1" "$2" "${GPU_LIST[g]}"; done ) &
done
wait
echo "== 3/3 staircase"; python scripts/06d_summarize_hush.py --tag "$TAG"
