#!/bin/bash
# LongBench-v2: baseline + SVD rank-16 runs over the full dataset (503 items).
#
# Usage:
#   bash benchmarks/run_longbench_v2.sh
#   CUDA_VISIBLE_DEVICES=0,3 bash benchmarks/run_longbench_v2.sh
#
# WARNING: --batch_size 4 at --max_input_len 120000 has not been tested on
# 2 H100s and may OOM. Our working config was batch=12 with 8k input; scaling
# input 15x likely won't fit.


cd "$(dirname "$(readlink -f "$0")")/LongBench-v2"

: "${CUDA_VISIBLE_DEVICES:=2,3}"
export CUDA_VISIBLE_DEVICES
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] === BASELINE start ==="
python pred.py -m qwen3.5-moe \
    --batch_size 4 --max_input_len 120000 --max_gen 32000 \
    --seed 42 -s results
echo "[$(ts)] === BASELINE done (rc=$?) ==="

echo "[$(ts)] === SVD r16 i1024 start ==="
python pred.py -m qwen3.5-moe --svd_rank 16 --svd_interval 1024 \
    --batch_size 4 --max_input_len 120000 --max_gen 32000 \
    --seed 42 -s results
echo "[$(ts)] === SVD done (rc=$?) ==="

echo "[$(ts)] === SCORING ==="
python result.py
cat result.txt
echo "[$(ts)] === ALL DONE ==="
