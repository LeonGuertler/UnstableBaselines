#!/usr/bin/env bash 
set -euo pipefail

PY=(python -m  unstable.utils.eval)
MODELS=(
  "Qwen/Qwen3-4B-Base"
)

fmt_hms () {
  local s=$1
  printf "%02dh:%02dm:%02ds" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

total=${#MODELS[@]}
echo "==> Will run $total model(s)."

t0_all=$(date +%s)

for i in "${!MODELS[@]}"; do
  n=$((i+1))
  model="${MODELS[$i]}"
  echo
  echo "[${n}/${total}] $(date '+%F %T') — Starting: $model"
  t0=$(date +%s)

  "${PY[@]}" --model-name "$model"
done