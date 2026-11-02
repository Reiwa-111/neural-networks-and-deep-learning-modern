#!/usr/bin/env bash
set -uo pipefail

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
export MPLBACKEND=Agg
status=0

for script in \
  fig/backprop_magnitude_nabla.py \
  fig/false_minima.py \
  fig/misleading_gradient.py \
  fig/misleading_gradient_contours.py \
  fig/pca_limitations.py \
  fig/valley.py \
  fig/valley2.py \
  fig/mnist.py \
  fig/replaced_by_d3/relu.py \
  fig/replaced_by_d3/sigmoid.py \
  fig/replaced_by_d3/step.py \
  fig/replaced_by_d3/tanh.py; do
  echo "=== $script ==="
  python "$script" || status=1
done

echo "status=$status"
exit "$status"
