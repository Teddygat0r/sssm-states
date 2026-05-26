#!/usr/bin/env bash
#
# Run the state-level reconstruction-fidelity sweep on all five models.
#
# Metrics per (layer, head, position, prompt): cosine similarity, relative
# Frobenius (MSE) error, per-element abs error, cumulative SVD energy and
# effective-rank thresholds, plus a matched random control.
#
# Prompts: 200 ShareGPT chunks + 200 codeparrot/The-Stack chunks (the loaders
# in svd_sweep/_helpers.py fetch and chunk these automatically).
#
# Usage:
#   ./run_all.sh                      # all five models
#   ./run_all.sh mamba2 deltanet      # only the named models
#
set -euo pipefail

ROOT="/home/joshuaz/sssm-states"
PYTHON="$ROOT/.venv/bin/python"
HERE="$ROOT/reconstruction_experiments"
LOGDIR="$HERE/logs"
mkdir -p "$LOGDIR"

# ---- knobs (override from the environment) ----
export SWEEP_N_PROMPTS="${SWEEP_N_PROMPTS:-200}"          # 200 per distribution
export SWEEP_MIN_CHARS_PER_PROMPT="${SWEEP_MIN_CHARS_PER_PROMPT:-24000}"
export SWEEP_MAX_FULL_TOKENS="${SWEEP_MAX_FULL_TOKENS:-65536}"
export SVD_PARALLEL_WORKERS="${SVD_PARALLEL_WORKERS:-4}"

# Pin BLAS to a single thread: the batched fp64 SVD runs in parallel across the
# worker threads already, and multi-threaded LAPACK SVD has an aarch64 race that
# can abort the process (Fortran STOP 1 in SLASCL).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(qwen35 mamba2 nemotron deltanet gated_deltanet)
fi

echo "=== reconstruction sweep ==="
echo "models:            ${MODELS[*]}"
echo "prompts/dist:      $SWEEP_N_PROMPTS"
echo "max full tokens:   $SWEEP_MAX_FULL_TOKENS"
echo "svd workers:       $SVD_PARALLEL_WORKERS"
echo "logs:              $LOGDIR"
echo

for m in "${MODELS[@]}"; do
    ts="$(date +%Y%m%d_%H%M%S)"
    log="$LOGDIR/${m}_${ts}.log"
    echo ">>> [$m] starting ($(date)); logging to $log"
    if "$PYTHON" "$HERE/run_reconstruction.py" --model "$m" > "$log" 2>&1; then
        echo ">>> [$m] DONE  ($(tail -n1 "$log"))"
    else
        echo ">>> [$m] FAILED — see $log"
        tail -n 20 "$log" || true
    fi
    echo
done

echo "=== all requested models finished ==="
