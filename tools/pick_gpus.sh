# Sourced by the run scripts: ASKS which GPUs to use (unless GPUS is already set), then sets GPU_LIST.
#   GPUS="1 2 3" bash tools/check.sh   -> no question asked
if [ -z "${GPUS:-}" ]; then
  echo "GPUs right now (index, memory used, utilisation):"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
  if [ -t 0 ]; then
    read -r -p "Which GPUs should this run use? (e.g. '1 2' or '1 2 3'; Enter = 1 2): " GPUS
  fi
fi
GPUS="${GPUS:-1 2}"
GPUS="${GPUS//,/ }"
GPU_LIST=($GPUS)
for g in "${GPU_LIST[@]}"; do
  [[ "$g" =~ ^[0-3]$ ]] || { echo "Not a GPU id (0-3): '$g'"; exit 1; }
done
export GPUS
echo "-> this run uses GPUs: ${GPU_LIST[*]} (${#GPU_LIST[@]} of 4)"
