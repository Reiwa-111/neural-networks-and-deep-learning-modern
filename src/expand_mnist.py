"""Create the 250k-image expanded MNIST training set (vectorized)."""

from __future__ import annotations

import argparse
from pathlib import Path

from nndl.augment import expand_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="expand only the first N training images"
    )
    args = parser.parse_args(argv)
    expand_dataset(args.data, args.output, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
