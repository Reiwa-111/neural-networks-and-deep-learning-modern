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
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${NNDL_RUN_DIR:-$ROOT/runs/envcheck-$STAMP}"
mkdir -p "$RUN_DIR"

status=0
run() {
  echo "=== $* ==="
  "$@" || status=1
}

run python -m nndl train mlp --book --epochs 1 --limit-train 2000 --limit-val 1000 \
  --device cpu --output-dir "$RUN_DIR/chk-book"
run python -m nndl train mlp --loss bce --epochs 1 --limit-train 2000 --limit-val 1000 \
  --device cpu --output-dir "$RUN_DIR/chk-bce"
run python -m nndl train mlp --loss mse --epochs 1 --limit-train 2000 --limit-val 1000 \
  --device cpu --output-dir "$RUN_DIR/chk-mse"
run python -m nndl train cnn --epochs 1 --limit-train 2000 --limit-val 1000 \
  --device cpu --output-dir "$RUN_DIR/chk-cnn"

# The benchmark command writes to ./runs by design. Run it from this unique
# directory so it cannot overwrite historical benchmark artifacts in the repo.
echo "=== python -m nndl benchmark ==="
(
  cd "$RUN_DIR"
  python -m nndl benchmark --epochs 1 --limit-train 500 --limit-val 200 --device cpu
) || status=1

echo "Artifacts: $RUN_DIR"
echo "status=$status"
exit "$status"
