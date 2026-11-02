#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -n "${VIRTUAL_ENV_ACTIVATE:-}" ]]; then
  if [[ ! -f "$VIRTUAL_ENV_ACTIVATE" ]]; then
    echo "Virtual environment activation script not found: $VIRTUAL_ENV_ACTIVATE" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$VIRTUAL_ENV_ACTIVATE"
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "No CUDA/ROCm device is visible to this PyTorch installation. "
        "Install the matching accelerator build or use _envcheck.sh for CPU checks."
    )

print(f"PyTorch: {torch.__version__}")
print(f"Accelerator: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}; ROCm: {torch.version.hip}")
PY

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${NNDL_RUN_DIR:-$ROOT/runs/gpu-verify-$STAMP}"
mkdir -p "$RUN_DIR"

python -m nndl train mlp --epochs 10 --output-dir "$RUN_DIR/mlp"
python -m nndl train cnn --epochs 10 --output-dir "$RUN_DIR/cnn"
echo "Artifacts: $RUN_DIR"
