"""SVM baseline on MNIST (modernized with scikit-learn)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nndl.baselines import svm_baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--limit", type=int, default=10000, help="training examples; 0 or negative means all"
    )
    parser.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    result = svm_baseline(
        args.data,
        limit=args.limit if args.limit > 0 else None,
        kernel=args.kernel,
        c=args.c,
        scale=not args.no_scale,
    )
    print("Baseline classifier using an SVM.")
    print(f"{result.correct} of {result.total} values correct ({result.accuracy:.2%}).")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
