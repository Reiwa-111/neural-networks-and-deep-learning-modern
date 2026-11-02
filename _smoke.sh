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
python -m pytest -q
