#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup_zamba2_env.sh
#
# Optional env vars:
#   ENV_NAME=zamba311
#   WORKSPACE_ROOT=$HOME/sssm-states
#   ZAMBA2_ROOT=$HOME/Zamba2
#   MAMBA_SSM_ROOT=$HOME/mamba-ssm
#   CAUSAL_CONV1D_ROOT=$HOME/causal-conv1d
#
# Assumptions:
# - micromamba is installed
# - the caller will customize the torch install if their platform requires it

ENV_NAME="${ENV_NAME:-zamba311}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HOME/sssm-states}"
ZAMBA2_ROOT="${ZAMBA2_ROOT:-$HOME/Zamba2}"
MAMBA_SSM_ROOT="${MAMBA_SSM_ROOT:-$HOME/mamba-ssm}"
CAUSAL_CONV1D_ROOT="${CAUSAL_CONV1D_ROOT:-$HOME/causal-conv1d}"

if ! command -v micromamba >/dev/null 2>&1; then
  echo "micromamba was not found in PATH." >&2
  exit 1
fi

eval "$(micromamba shell hook --shell bash)"

if ! micromamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  micromamba create -y -n "${ENV_NAME}" python=3.11
fi

micromamba activate "${ENV_NAME}"

python -m pip install -U pip "setuptools<82" wheel ninja packaging

# Install PyTorch only if missing. Users may want to replace this with a platform-specific command.
if ! python - <<'PY' >/dev/null 2>&1
import torch
PY
then
  python -m pip install torch torchvision torchaudio
fi

python -m pip install -U \
  transformers \
  huggingface_hub \
  safetensors \
  sentencepiece \
  protobuf \
  accelerate \
  einops \
  matplotlib \
  numpy

mkdir -p "$(dirname "${WORKSPACE_ROOT}")"
mkdir -p "$(dirname "${ZAMBA2_ROOT}")"
mkdir -p "$(dirname "${MAMBA_SSM_ROOT}")"
mkdir -p "$(dirname "${CAUSAL_CONV1D_ROOT}")"

if [ ! -d "${ZAMBA2_ROOT}/.git" ]; then
  git clone https://github.com/Zyphra/Zamba2.git "${ZAMBA2_ROOT}"
fi

if [ ! -d "${MAMBA_SSM_ROOT}/.git" ]; then
  git clone https://github.com/state-spaces/mamba.git "${MAMBA_SSM_ROOT}"
fi

if [ ! -d "${CAUSAL_CONV1D_ROOT}/.git" ]; then
  git clone https://github.com/Dao-AILab/causal-conv1d.git "${CAUSAL_CONV1D_ROOT}"
fi

deps_ok() {
  python - <<'PY'
import mamba_ssm, causal_conv1d
print("deps ok")
PY
}

if ! deps_ok >/dev/null 2>&1; then
  echo "mamba_ssm / causal_conv1d not importable; trying local source installs..."

  if [ -d "${CAUSAL_CONV1D_ROOT}" ]; then
    python -m pip install --no-build-isolation --no-deps "${CAUSAL_CONV1D_ROOT}" || true
  fi

  if [ -d "${MAMBA_SSM_ROOT}" ]; then
    python -m pip install --no-build-isolation --no-deps "${MAMBA_SSM_ROOT}" || true
  fi

  if ! deps_ok >/dev/null 2>&1; then
    cat >&2 <<EOF
Could not make both \`mamba_ssm\` and \`causal_conv1d\` importable.

What the script tried:
- checked \`python -c "import mamba_ssm, causal_conv1d; print('deps ok')"\`
- attempted local source installs from:
  - ${MAMBA_SSM_ROOT}
  - ${CAUSAL_CONV1D_ROOT}

Next step:
- inspect the build output for your platform
- if these packages are already available from another environment path, ensure this micromamba env can import them
EOF
    exit 1
  fi
fi

echo "deps ok"
echo "Environment ready: ${ENV_NAME}"
echo "Workspace root: ${WORKSPACE_ROOT}"
echo "Zamba2 source: ${ZAMBA2_ROOT}"
echo "mamba-ssm source: ${MAMBA_SSM_ROOT}"
echo "causal-conv1d source: ${CAUSAL_CONV1D_ROOT}"
echo
echo "Sanity check:"
echo "  micromamba activate ${ENV_NAME}"
echo "  cd ${WORKSPACE_ROOT}"
echo "  python state_spectrum_sweep/run_experiment_zamba2.py --inspect-cache"
