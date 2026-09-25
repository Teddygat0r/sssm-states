#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
unit="sssm-all-models-kl-$(date -u +%Y%m%dT%H%M%SZ)"
out="$repo/experiments/all_models_int8_svd_kl/results/$unit"
mkdir -p "$out"
systemd-run --user --unit="$unit" --working-directory="$repo" \
 --property="StandardOutput=append:$out/run.log" --property="StandardError=append:$out/run.log" \
 --setenv=PYTHONUNBUFFERED=1 --setenv=TOKENIZERS_PARALLELISM=false \
 --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
 --setenv=SWEEP_N_PROMPTS=100 --setenv=SWEEP_MIN_CHARS_PER_PROMPT=24000 --setenv=SWEEP_MAX_FULL_TOKENS=65536 \
 "$repo/.venv/bin/python" "$repo/experiments/all_models_int8_svd_kl/suite.py" --output "$out"
printf '%s\n' "$unit" > "$out/unit.txt"
printf '%s\n' "$out" > "$repo/experiments/all_models_int8_svd_kl/latest.txt"
printf 'UNIT=%s\nOUTPUT=%s\n' "$unit" "$out"
