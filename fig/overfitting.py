"""Illustrate overfitting and L2 regularisation on MNIST.

Trains the book's 784-30-10 network on a small (or full) training set and
writes ``overfitting.json``-style results: test cost, test accuracy, training
cost and training accuracy.  PNG names follow the JSON stem (``overfitting`` ->
``overfitting1.png`` ... ``overfitting4.png``; the ``*_full`` variants produce
the combined accuracy plot).  Existing JSON files are reused unless
``--recompute`` is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
FIG_DIR = Path(__file__).resolve().parent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nndl.config import TrainConfig, set_seed
from nndl.data import MNISTBundle, load_mnist, make_dataloaders, to_one_hot
from nndl.losses import BinaryCrossEntropyLoss
from nndl.models import MLP
from nndl.trainer import Trainer, make_optimizer

KNOWN_FILES = [
    "overfitting.json",
    "overfitting_full.json",
    "regularized.json",
    "regularized_full.json",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--filename",
        default=None,
        help="JSON result file (default: plot all existing known files)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="plot overfitting/regularized and their *_full variants",
    )
    parser.add_argument(
        "--recompute", action="store_true", help="train instead of reusing the JSON file"
    )
    parser.add_argument("--quick", action="store_true", help="5 epochs on 500 examples")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--training-set-size", type=int, default=None)
    parser.add_argument("--lmbda", type=float, default=None)
    parser.add_argument("--training-cost-xmin", type=int, default=None)
    parser.add_argument("--test-accuracy-xmin", type=int, default=None)
    parser.add_argument("--test-cost-xmin", type=int, default=None)
    parser.add_argument("--training-accuracy-xmin", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=12345678)
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args(argv)

    if args.quick:
        args.epochs = 5 if args.epochs is None else args.epochs
        args.training_set_size = 500 if args.training_set_size is None else args.training_set_size
    if args.filename is None and args.recompute and not args.all:
        args.filename = "overfitting.json"
    return args


def infer_training_set_size(path: Path, args: argparse.Namespace) -> int:
    if args.training_set_size is not None:
        return args.training_set_size
    return 50000 if path.stem.endswith("_full") else 1000


def infer_lmbda(path: Path, args: argparse.Namespace) -> float:
    if args.lmbda is not None:
        return args.lmbda
    return 5.0 if path.stem.startswith("regularized") else 0.0


def infer_epochs(path: Path, args: argparse.Namespace, existing: int | None = None) -> int:
    if args.epochs is not None:
        return args.epochs
    if existing is not None:
        return existing
    return 30 if path.stem.endswith("_full") else 400


def one_hot_bundle(bundle: MNISTBundle) -> MNISTBundle:
    """Return a bundle whose labels are one-hot rows for BCE training."""
    return MNISTBundle(
        bundle.train_x,
        to_one_hot(bundle.train_y),
        bundle.val_x,
        to_one_hot(bundle.val_y),
        bundle.test_x,
        to_one_hot(bundle.test_y),
    )


def run_network(filename: Path, args: argparse.Namespace) -> None:
    training_set_size = infer_training_set_size(filename, args)
    lmbda = infer_lmbda(filename, args)
    epochs = infer_epochs(filename, args)

    set_seed(args.seed)
    bundle = one_hot_bundle(load_mnist(flatten=True))
    test_size = len(bundle.test_y)
    subset = bundle.subset(train=training_set_size)
    train_loader, _, test_loader = make_dataloaders(
        subset, batch_size=10, eval_batch_size=1024, seed=args.seed
    )

    model = MLP([784, 30, 10], hidden_activation="sigmoid", output_activation=None)
    model.reset_parameters("large")  # the book's large_weight_initializer()
    config = TrainConfig(
        epochs=epochs,
        batch_size=10,
        lr=0.5,
        weight_decay=lmbda / training_set_size,
        device=args.device,
        seed=args.seed,
    )
    trainer = Trainer(model, BinaryCrossEntropyLoss(), make_optimizer(model, config), config)
    history = trainer.fit(train_loader, test_loader, verbose=True)

    test_cost = [float(value) for value in history.val_loss]
    test_accuracy = [round(value * test_size) for value in history.val_accuracy]
    training_cost = [float(value) for value in history.train_loss]
    training_accuracy = [round(value * training_set_size) for value in history.train_accuracy]

    with filename.open("w", encoding="utf-8") as handle:
        json.dump([test_cost, test_accuracy, training_cost, training_accuracy], handle)


def _resolve_xmins(n: int, args: argparse.Namespace) -> tuple[int, int, int, int]:
    if n <= 50:
        defaults = (0, 0, 0, 0)
    else:
        defaults = (200, 200, 0, 0)
    values = (
        args.training_cost_xmin,
        args.test_accuracy_xmin,
        args.test_cost_xmin,
        args.training_accuracy_xmin,
    )
    resolved = tuple(
        int(default if value is None else value) for default, value in zip(defaults, values)
    )
    upper = max(n - 1, 0)
    return tuple(min(max(0, value), upper) for value in resolved)


def make_plots(filename: Path, args: argparse.Namespace) -> None:
    with filename.open("r", encoding="utf-8") as handle:
        test_cost, test_accuracy, training_cost, training_accuracy = json.load(handle)

    n = min(len(test_cost), len(test_accuracy), len(training_cost), len(training_accuracy))
    if n == 0:
        print(f"Skipping {filename.name}: no epochs recorded")
        return
    training_set_size = infer_training_set_size(filename, args)
    training_cost_xmin, test_accuracy_xmin, test_cost_xmin, training_accuracy_xmin = _resolve_xmins(
        n, args
    )

    stem = filename.stem
    if stem.endswith("_full"):
        plot_overlay(
            test_accuracy,
            training_accuracy,
            training_set_size,
            min(test_accuracy_xmin, training_accuracy_xmin),
            filename.with_suffix(".png"),
        )
    else:
        plot_training_cost(training_cost, training_cost_xmin, filename.with_name(stem + "1.png"))
        plot_test_accuracy(test_accuracy, test_accuracy_xmin, filename.with_name(stem + "2.png"))
        plot_test_cost(test_cost, test_cost_xmin, filename.with_name(stem + "3.png"))
        plot_training_accuracy(
            training_accuracy,
            training_accuracy_xmin,
            training_set_size,
            filename.with_name(stem + "4.png"),
        )


def plot_training_cost(training_cost: list[float], xmin: int, filename: Path) -> None:
    y = np.asarray(training_cost)[xmin:]
    fig, ax = plt.subplots()
    ax.plot(np.arange(xmin, xmin + len(y)), y, color="#2A6EA6")
    ax.set_xlim([xmin, xmin + len(y)])
    ax.grid(True)
    ax.set_xlabel("Epoch")
    ax.set_title("Cost on the training data")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_test_accuracy(test_accuracy: list[int], xmin: int, filename: Path) -> None:
    y = np.asarray(test_accuracy, dtype=float)[xmin:] / 100.0
    fig, ax = plt.subplots()
    ax.plot(np.arange(xmin, xmin + len(y)), y, color="#2A6EA6")
    ax.set_xlim([xmin, xmin + len(y)])
    ax.grid(True)
    ax.set_xlabel("Epoch")
    ax.set_title("Accuracy (%) on the test data")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_test_cost(test_cost: list[float], xmin: int, filename: Path) -> None:
    y = np.asarray(test_cost)[xmin:]
    fig, ax = plt.subplots()
    ax.plot(np.arange(xmin, xmin + len(y)), y, color="#2A6EA6")
    ax.set_xlim([xmin, xmin + len(y)])
    ax.grid(True)
    ax.set_xlabel("Epoch")
    ax.set_title("Cost on the test data")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_training_accuracy(
    training_accuracy: list[int], xmin: int, training_set_size: int, filename: Path
) -> None:
    y = np.asarray(training_accuracy, dtype=float)[xmin:] * 100.0 / training_set_size
    fig, ax = plt.subplots()
    ax.plot(np.arange(xmin, xmin + len(y)), y, color="#2A6EA6")
    ax.set_xlim([xmin, xmin + len(y)])
    ax.grid(True)
    ax.set_xlabel("Epoch")
    ax.set_title("Accuracy (%) on the training data")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_overlay(
    test_accuracy: list[int],
    training_accuracy: list[int],
    training_set_size: int,
    xmin: int,
    filename: Path,
) -> None:
    # Slice both series from the same starting epoch; the original code
    # passed full-length arrays together with a shortened x axis.
    test_y = np.asarray(test_accuracy, dtype=float)[xmin:] / 100.0
    train_y = np.asarray(training_accuracy, dtype=float)[xmin:] * 100.0 / training_set_size
    epochs = np.arange(xmin, xmin + min(len(test_y), len(train_y)))
    fig, ax = plt.subplots()
    ax.plot(epochs, test_y[: len(epochs)], color="#2A6EA6", label="Accuracy on the test data")
    ax.plot(
        epochs,
        train_y[: len(epochs)],
        color="#FFA933",
        label="Accuracy on the training data",
    )
    ax.grid(True)
    ax.set_xlim([xmin, xmin + len(epochs)])
    ax.set_xlabel("Epoch")
    ax.set_ylim([90, 100])
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.filename is not None:
        filenames = [Path(args.filename)]
    elif args.all:
        filenames = [
            Path(name) for name in KNOWN_FILES if args.recompute or (output_dir / name).exists()
        ]
    else:
        filenames = [Path(name) for name in KNOWN_FILES if (output_dir / name).exists()]
        if not filenames:
            filenames = [Path("overfitting.json")]

    for filename in filenames:
        path = filename if filename.is_absolute() else output_dir / filename
        if args.recompute or not path.exists():
            run_network(path, args)
        else:
            print(f"Reusing {path.name} (pass --recompute to retrain).")
        make_plots(path, args)


if __name__ == "__main__":
    main()
