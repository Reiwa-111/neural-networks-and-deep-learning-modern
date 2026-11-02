"""Naive average-darkness baseline (modernized)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from nndl.baselines import average_darkness


def avg_darknesses(training_data) -> defaultdict:
    """Average pixel sum for every digit in the training data."""

    digit_counts: defaultdict = defaultdict(int)
    darknesses: defaultdict = defaultdict(float)
    for image, digit in zip(training_data[0], training_data[1]):
        digit_counts[int(digit)] += 1
        darknesses[int(digit)] += float(np.asarray(image).sum())
    return defaultdict(
        float,
        {digit: darknesses[digit] / count for digit, count in digit_counts.items()},
    )


def guess_digit(image, avgs) -> int:
    """Return the digit whose training average is closest to the image."""

    darkness = float(np.asarray(image).sum())
    return min(avgs, key=lambda digit: abs(avgs[digit] - darkness))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    result = average_darkness(args.data)
    print("Baseline classifier using average darkness of image.")
    print(f"{result.correct} of {result.total} values correct ({result.accuracy:.2%}).")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
