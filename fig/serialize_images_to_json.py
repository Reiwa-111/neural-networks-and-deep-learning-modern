"""Serialize part of MNIST to JSON for the JavaScript/d3 visualisations.

Produces the book's ``data_1000.json`` layout::

    {"training": [{"y": [10 one-hot floats], "x": [784 floats]}, ...],
     "validation": [{"y": <int label>, "x": [784 floats]}, ...]}

The conversion is vectorised; nothing is loaded or written at import time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
FIG_DIR = Path(__file__).resolve().parent

from nndl.data import load_mnist, to_one_hot

DEFAULT_TRAINING = 1000
DEFAULT_VALIDATION = 100


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_TRAINING, help="training images to serialize"
    )
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=DEFAULT_VALIDATION,
        help="validation images to serialize",
    )
    parser.add_argument("--output", type=Path, default=FIG_DIR / "data_1000.json")
    parser.add_argument("--data", type=Path, default=None, help="optional MNIST pickle path")
    return parser.parse_args(argv)


def _images_as_lists(images) -> list[list[float]]:
    array = images.detach().cpu().numpy().reshape(images.shape[0], -1)
    return array.astype(float).tolist()


def build_payload(
    training_limit: int, validation_limit: int, data_path: Path | None = None
) -> dict:
    bundle = load_mnist(data_path, flatten=True)
    training_limit = min(training_limit, len(bundle.train_y))
    validation_limit = min(validation_limit, len(bundle.val_y))

    training_x = _images_as_lists(bundle.train_x[:training_limit])
    training_y = to_one_hot(bundle.train_y[:training_limit]).numpy().astype(float).tolist()
    validation_x = _images_as_lists(bundle.val_x[:validation_limit])
    validation_y = bundle.val_y[:validation_limit].numpy().astype(int).tolist()

    return {
        "training": [{"y": y, "x": x} for x, y in zip(training_x, training_y)],
        "validation": [{"y": y, "x": x} for x, y in zip(validation_x, validation_y)],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_payload(args.limit, args.validation_limit, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(
        "Wrote {} training and {} validation examples to {}".format(
            len(payload["training"]), len(payload["validation"]), args.output
        )
    )


if __name__ == "__main__":
    main()
