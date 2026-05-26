#!/usr/bin/env bash
#
# KL-under-state-compression generation sweep across all five recurrent / SSM
# models. For each (model, distribution, prefill length) we prefill the prompt,
# greedily generate KL_GEN_TOKENS tokens, and every KL_EVERY steps measure
# KL(P_true || P_rank-k) for each rank in KL_RANKS by compressing the entire
# recurrent state (one batched forward; attention KV / conv left exact).
#
# Prompts: ShareGPT chat + The-Stack code (svd_sweep/_helpers.load_prompts),
# same two distributions as the reconstruction sweep.
#
# Usage:
#   ./run_all.sh                      # all five models
#   ./run_all.sh mamba2 qwen35        # only the named models
#   KL_EVERY=8 SWEEP_N_PROMPTS=50 ./run_all.sh   # cheaper run
set -euo pipefail

ROOT="/home/joshuaz/sssm-states"
PYTHON="$ROOT/.venv/bin/python"
HERE="$ROOT/kl_generation_experiments"
LOGDIR="$HERE/logs"
mkdir -p "$LOGDIR"

# ---- experiment knobs (override from the environment) ----
export KL_GEN_TOKENS="${KL_GEN_TOKENS:-128}"     # tokens generated per prefill
export KL_EVERY="${KL_EVERY:-4}"                 # measure KL every N tokens
export KL_RANKS="${KL_RANKS:-4,8,16}"            # ranks compared per step
export SWEEP_POSITIONS="${SWEEP_POSITIONS:-256}" # prefill lengths; "full" added
export SWEEP_N_PROMPTS="${SWEEP_N_PROMPTS:-100}" # prompts per distribution
export SWEEP_MIN_CHARS_PER_PROMPT="${SWEEP_MIN_CHARS_PER_PROMPT:-24000}"
export SWEEP_MAX_FULL_TOKENS="${SWEEP_MAX_FULL_TOKENS:-65536}"
export KL_STOP_ON_EOS="${KL_STOP_ON_EOS:-0}"
export KL_OVERSAMPLE="${KL_OVERSAMPLE:-8}"       # rsvd_eigh sketch oversample
export KL_NITER="${KL_NITER:-2}"                 # rsvd_eigh power iterations

# Pin BLAS to one thread (matches the reconstruction sweep; avoids the aarch64
# LAPACK SVD race). The compression SVD here runs on-GPU via svd_lowrank.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(mamba2 qwen35 nemotron deltanet gated_deltanet)
fi

echo "=== KL-generation sweep ==="
echo "models:        ${MODELS[*]}"
echo "gen tokens:    $KL_GEN_TOKENS   every: $KL_EVERY   ranks: $KL_RANKS"
echo "positions:     $SWEEP_POSITIONS (+full)   prompts/dist: $SWEEP_N_PROMPTS"
echo "logs:          $LOGDIR"
echo

for m in "${MODELS[@]}"; do
    ts="$(date +%Y%m%d_%H%M%S)"
    log="$LOGDIR/${m}_${ts}.log"
    echo ">>> [$m] starting ($(date)); logging to $log"
    if "$PYTHON" "$HERE/run_kl_generation.py" --model "$m" > "$log" 2>&1; then
        echo ">>> [$m] DONE  ($(tail -n1 "$log"))"
    else
        echo ">>> [$m] FAILED — see $log"
        tail -n 20 "$log" || true
    fi
    echo
done

echo "=== all requested models finished ==="
