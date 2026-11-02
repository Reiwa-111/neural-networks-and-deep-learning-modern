"""Compare training with three different learning rates.

The cost curves are stored in ``multiple_eta.json`` (book-compatible: four
lists per run, of which only the training cost is populated) and plotted in
``multiple_eta.png``.  Existing results are reused unless ``--recompute`` is
passed.
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

LEARNING_RATES = [0.025, 0.25, 2.5]
COLORS = ["#2A6EA6", "#FFCD33", "#FF7033"]
NUM_EPOCHS = 30
JSON_FILE = "multiple_eta.json"
PNG_FILE = "multiple_eta.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--recompute", action="store_true", help="retrain instead of reusing the JSON file"
    )
    parser.add_argument("--quick", action="store_true", help="one epoch on 1000 examples")
    parser.add_argument("--epochs", type=int, default=None, help="epochs per learning rate")
    parser.add_argument("--train-size", type=int, default=None, help="training examples")
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


def run_networks(args: argparse.Namespace, filename: Path) -> None:
    bundle = one_hot_bundle(load_mnist(flatten=True))
    subset = bundle.subset(train=args.train_size)
    train_loader, _, _ = make_dataloaders(
        subset, batch_size=10, eval_batch_size=1024, seed=args.seed
    )
    weight_decay = args.lmbda / args.train_size

    results = []
    for eta in LEARNING_RATES:
        print(f"\nTrain a network using eta = {eta}")
        set_seed(args.seed)
        model = MLP([784, 30, 10], hidden_activation="sigmoid", output_activation=None)
        config = TrainConfig(
            epochs=args.epochs,
            batch_size=10,
            lr=eta,
            weight_decay=weight_decay,
            device=args.device,
            seed=args.seed,
        )
        trainer = Trainer(
            model,
            BinaryCrossEntropyLoss(),
            make_optimizer(model, config),
            config,
        )
        history = trainer.fit(train_loader, val_loader=None, verbose=False)
        # Keep network2.SGD's return layout: eval cost, eval accuracy,
        # training cost, training accuracy.  Only the cost curve was recorded.
        results.append([[], [], [float(cost) for cost in history.train_loss], []])

    with filename.open("w", encoding="utf-8") as handle:
        json.dump(results, handle)


def make_plot(filename: Path) -> None:
    with filename.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    fig, ax = plt.subplots()
    for eta, result, color in zip(LEARNING_RATES, results, COLORS):
        training_cost = result[2]
        ax.plot(
            np.arange(len(training_cost)),
            training_cost,
            "o-",
            label=r"$\eta$ = " + str(eta),
            color=color,
        )
    epochs = max((len(result[2]) for result in results), default=NUM_EPOCHS)
    ax.set_xlim([0, epochs])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cost")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(filename.with_name(PNG_FILE), dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / JSON_FILE

    if args.recompute or not filename.exists():
        run_networks(args, filename)
    else:
        print("Reusing precomputed JSON (pass --recompute to retrain).")
    make_plot(filename)


if __name__ == "__main__":
    main()
