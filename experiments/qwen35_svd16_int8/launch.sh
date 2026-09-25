#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mode=${1:-full}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
unit="sssm-qwen35-int8-${mode}-${stamp}"
out="$repo/experiments/qwen35_svd16_int8/results/$unit"
mkdir -p "$out"
args=(--output "$out")
if [[ "$mode" == smoke ]]; then
  args+=(--smoke --prompts-per-kind 1 --steps 4 --contexts 256)
elif [[ "$mode" != full ]]; then
  echo 'Usage: launch.sh [smoke|full]' >&2
  exit 2
fi
systemd-run --user --unit="$unit" --working-directory="$repo" \
  --property="StandardOutput=append:$out/run.log" \
  --property="StandardError=append:$out/run.log" \
  --setenv=PYTHONUNBUFFERED=1 --setenv=TOKENIZERS_PARALLELISM=false \
  "$repo/.venv/bin/python" "$repo/experiments/qwen35_svd16_int8/run.py" "${args[@]}"
printf '%s\n' "$unit" > "$out/unit.txt"
printf '%s\n' "$out" > "$repo/experiments/qwen35_svd16_int8/latest_${mode}.txt"
printf 'UNIT=%s\nOUTPUT=%s\n' "$unit" "$out"
