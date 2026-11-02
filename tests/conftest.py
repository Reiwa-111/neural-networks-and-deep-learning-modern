from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def mnist_path(repo_root: Path) -> Path:
    path = repo_root / "data" / "mnist.pkl.gz"
    if not path.exists():
        pytest.skip(f"MNIST data not found: {path}")
    return path
