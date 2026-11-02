"""Effect of weight initialisation on MNIST classification accuracy.

Compares the book's large (standard deviation 1) initial weights with the
default 1/sqrt(fan-in) scheme for networks with 100 and 30 hidden neurons.
Results are written to ``weight_initialization_<n>.json`` and the matching
PNG.  Existing JSON files are reused unless ``--recompute`` is passed.
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

HIDDEN_SIZES = [100, 30]
NUM_EPOCHS = 30


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--recompute", action="store_true", help="retrain instead of reusing JSON files"
    )
    parser.add_argument("--quick", action="store_true", help="one epoch on 1000 examples")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=HIDDEN_SIZES)
    parser.add_argument("--eta", type=float, default=3.0)
    parser.add_argument("--lmbda", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=12345678)
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args(argv)
    if args.quick:
        args.epochs = 1 if args.epochs is None else args.epochs
        args.train_size = 1000 if args.train_size is None else args.train_size
    else:
        args.epochs = NUM_EPOCHS if args.epochs is None else args.epochs
        args.train_size = 50000 if args.train_size is None else args.train_size
    return args


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


def train_once(hidden_size: int, initialization: str, args: argparse.Namespace) -> list[int]:
    set_seed(args.seed)
    bundle = one_hot_bundle(load_mnist(flatten=True))
    subset = bundle.subset(train=args.train_size)
    train_loader, val_loader, _ = make_dataloaders(
        subset, batch_size=10, eval_batch_size=1024, seed=args.seed
    )

    model = MLP([784, hidden_size, 10], hidden_activation="sigmoid", output_activation=None)
    if initialization == "large":
        model.reset_parameters("large")
    else:
        model.reset_parameters("default")
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=10,
        lr=args.eta,
        weight_decay=args.lmbda / args.train_size,
        device=args.device,
        seed=args.seed,
    )
    trainer = Trainer(model, BinaryCrossEntropyLoss(), make_optimizer(model, config), config)
    history = trainer.fit(train_loader, val_loader, verbose=True)
    validation_size = len(bundle.val_y)
    return [round(value * validation_size) for value in history.val_accuracy]


def run_network(filename: Path, hidden_size: int, args: argparse.Namespace) -> None:
    print(f"\nHidden layer size {hidden_size}: default starting weights")
    default_accuracy = train_once(hidden_size, "default", args)
    print(f"Hidden layer size {hidden_size}: large starting weights")
    large_accuracy = train_once(hidden_size, "large", args)
    payload = {
        "default_weight_initialization": [[], default_accuracy, [], []],
        "large_weight_initialization": [[], large_accuracy, [], []],
    }
    with filename.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def make_plot(filename: Path) -> None:
    with filename.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    _, default_va, _, _ = results["default_weight_initialization"]
    _, large_va, _, _ = results["large_weight_initialization"]
    # Raw classification counts out of 10,000 -> percentages.
    default_va = np.asarray(default_va, dtype=float) / 100.0
    large_va = np.asarray(large_va, dtype=float) / 100.0
    epochs = min(len(default_va), len(large_va))

    fig, ax = plt.subplots()
    ax.plot(
        np.arange(epochs),
        large_va[:epochs],
        color="#2A6EA6",
        label="Old approach to weight initialization",
    )
    ax.plot(
        np.arange(epochs),
        default_va[:epochs],
        color="#FFA933",
        label="New approach to weight initialization",
    )
    ax.set_xlim([0, epochs])
    ax.set_xlabel("Epoch")
    ax.set_ylim([85, 100])
    ax.set_title("Classification accuracy")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(filename.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for hidden_size in args.hidden_sizes:
        filename = output_dir / f"weight_initialization_{hidden_size}.json"
        if args.recompute or not filename.exists():
            run_network(filename, hidden_size, args)
        else:
            print(f"Reusing {filename.name} (pass --recompute to retrain).")
        make_plot(filename)


if __name__ == "__main__":
    main()
