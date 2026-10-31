#!/usr/bin/env bash
# SERVER: Phase 4 (TEST) in one command. The TEST set is opened ONCE: a second run refuses unless FORCE=1.
#   bash tools/phase4.sh    (asks which GPUs; ~25-40 min)
set -euo pipefail
cd "$(dirname "$0")/.."
source /scratch/mtech_shrinkhal/ml-projects/venv/bin/activate
# pull first, then restart on the FRESH copy of this script
if [ "${LUNGDET_PULLED:-0}" != "1" ]; then git pull --ff-only; LUNGDET_PULLED=1 exec bash "$0" "$@"; fi
OUT="$SCRATCH/lungdet_results/phase4"; mkdir -p "$OUT"
if [ ! -f "$SCRATCH/lungdet_results/check/final_choice.json" ]; then
  echo "The length check has not chosen a final model yet (no check/final_choice.json). Finish tools/check.sh first."; exit 1
fi
if [ -f "$OUT/TEST_OPENED" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "The TEST set was already opened:"; cat "$OUT/TEST_OPENED"; echo "Re-running would be a second look. Use FORCE=1 only to re-run the SAME models."; exit 1
fi
source tools/pick_gpus.sh          # asks which GPUs to use
LOG="$SCRATCH/logs/phase4"; mkdir -p "$LOG"
exec 9>"$LOG/.lock"; flock -n 9 || { echo "Phase 4 is ALREADY running in another window."; exit 1; }
python -m pytest tests/ -q
MODELS=$(python -c "import sys; sys.path.insert(0,'src'); from lungdet.common.config import load_config; from lungdet.discard.models import model_specs; print(' '.join(model_specs(load_config())[0]))")
echo "TEST opened $(date '+%Y-%m-%d %H:%M') for models: R0 $MODELS (final choice: $(cat $SCRATCH/lungdet_results/check/final_choice.json | tr -d '\n '))" > "$OUT/TEST_OPENED"
echo "== 1/3 scoring TEST with: $MODELS (+ R0) on GPUs ${GPU_LIST[*]}"
M=($MODELS); NG=${#GPU_LIST[@]}
for ((g=0; g<NG; g++)); do
  ( for ((n=g; n<${#M[@]}; n+=NG)); do
      CUDA_VISIBLE_DEVICES=${GPU_LIST[g]} python -u scripts/10_score_test.py --model "${M[n]}" > "$LOG/score_${M[n]}.log" 2>&1 \
        && echo "   $(tail -1 "$LOG/score_${M[n]}.log")" || echo "   FAILED ${M[n]} (see $LOG/score_${M[n]}.log)"
    done ) &
done
wait
echo "== 2/3 TEST numbers + 95% CIs"; python -u scripts/10b_test_report.py
echo "== 3/3 explanations (occlusion + sanity check + galleries)"
CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} python -u scripts/11_explain.py
