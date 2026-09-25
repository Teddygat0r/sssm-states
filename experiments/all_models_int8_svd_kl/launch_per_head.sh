#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
previous=$(cat "$repo/experiments/all_models_int8_svd_kl/latest.txt")
unit="sssm-all-models-kl-per-head-$(date -u +%Y%m%dT%H%M%SZ)"
out="$repo/experiments/all_models_int8_svd_kl/results/$unit"
mkdir -p "$out"
systemd-run --user --unit="$unit" --working-directory="$repo" \
 --property="StandardOutput=append:$out/run.log" --property="StandardError=append:$out/run.log" \
 --setenv=PYTHONUNBUFFERED=1 --setenv=TOKENIZERS_PARALLELISM=false \
 --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
 --setenv=SWEEP_MAX_FULL_TOKENS=65536 \
 "$repo/.venv/bin/python" "$repo/experiments/all_models_int8_svd_kl/parallel_suite.py" \
 --output "$out" --prompts "$previous/prompts.json" --int8-per-head-only
printf '%s\n' "$unit" > "$out/unit.txt"
printf '%s\n' "$out" > "$repo/experiments/all_models_int8_svd_kl/latest_per_head.txt"
printf 'UNIT=%s\nOUTPUT=%s\n' "$unit" "$out"
