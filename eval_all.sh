#!/usr/bin/env bash
set -euo pipefail

PY=(python3 eval.py)

MODELS=(
  "Qwen/Qwen3-4B-Base"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000025"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000050"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000075"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000100"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000125"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000150"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000175"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000200"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000225"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000250"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000275"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000300"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000325"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000350"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000375"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000400"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000425"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000450"
  "LeonGuertler/ScalingLaws-Qwen3-4B-Chopsticks-step_000475"
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

  dt=$(( $(date +%s) - t0 ))
  echo "[${n}/${total}] Finished: $model in $(fmt_hms $dt)"
done

echo
echo "All done in $(fmt_hms $(( $(date +%s) - t0_all )))"
