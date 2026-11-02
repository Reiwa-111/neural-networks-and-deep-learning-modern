"""Gradient-magnitude experiments from chapter 5.

Compute the average bias gradient (the ``delta`` terms) with
``network2.Network.backprop`` and record how its norm evolves while training
networks with two, three and four hidden layers.

The committed JSON files are reused by default so that running the script
simply regenerates the PNGs.  Pass ``--recompute`` (optionally with
``--quick``) to retrain.
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
from torch.utils.data import TensorDataset

import network2
from nndl.config import set_seed
from nndl.data import load_mnist, to_one_hot

ARCHITECTURES = {
    2: [784, 30, 30, 10],
    3: [784, 30, 30, 30, 10],
    4: [784, 30, 30, 30, 30, 10],
}
COLORS = ["#2A6EA6", "#FFA933", "#FF5555", "#55FF55", "#5555FF"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="retrain the networks instead of reusing the committed JSON files",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="tiny smoke-test training run (1 epoch, 200 examples)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="epochs per network (default: 30, quick: 1)"
    )
    parser.add_argument(
        "--train-size", type=int, default=None, help="training examples (default: 1000, quick: 200)"
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--lmbda", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=12345678)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIG_DIR,
        help="directory for JSON/PNG outputs (default: this script's directory)",
    )
    args = parser.parse_args(argv)
    if args.epochs is None:
        args.epochs = 1 if args.quick else 30
    if args.train_size is None:
        args.train_size = 200 if args.quick else 1000
    return args


def make_dataset(size: int) -> TensorDataset:
    bundle = load_mnist(flatten=True)
    x = bundle.train_x[:size]
    y = to_one_hot(bundle.train_y[:size])
    return TensorDataset(x, y)


def average_bias_gradient(net: network2.Network, data: TensorDataset) -> list[np.ndarray]:
    """Average ``dC/db`` over every example in ``data``."""
    total: list[np.ndarray] | None = None
    for x, y in data:
        nabla_b, _ = net.backprop(x.numpy(), y.numpy())
        if total is None:
            total = [np.zeros_like(g) for g in nabla_b]
        for accumulator, gradient in zip(total, nabla_b):
            accumulator += gradient
    if total is None:
        raise ValueError("cannot average gradients of an empty dataset")
    return [g / len(data) for g in total]


def initial_gradient(net: network2.Network, data: TensorDataset) -> list[list[float]]:
    """Top six components of the hidden-layer bias gradient (book format)."""
    averaged = average_bias_gradient(net, data)
    return [g.reshape(-1)[:6].tolist() for g in averaged[:-1]]


def train_and_record(
    net: network2.Network, data: TensorDataset, args: argparse.Namespace
) -> list[list[float]]:
    norms: list[list[float]] = []
    for epoch in range(args.epochs):
        averaged = average_bias_gradient(net, data)
        norms.append([float(np.linalg.norm(g)) for g in averaged[:-1]])
        print(f"Epoch {epoch}: hidden-layer gradient norms {norms[-1]}")
        net.SGD(data, 1, args.batch_size, args.eta, lmbda=args.lmbda)
    return norms


def compute_all(args: argparse.Namespace, recompute: bool) -> bool:
    """Compute whichever artifacts are missing (or all of them)."""
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    needed = {
        layers: recompute or not (out / f"norms_during_training_{layers}_layers.json").exists()
        for layers in (2, 3, 4)
    }
    need_initial = recompute or not (out / "initial_gradient.json").exists()
    if not need_initial and not any(needed.values()):
        return False

    data = make_dataset(args.train_size)
    set_seed(args.seed)

    # The first 1000 examples are enough to expose the vanishing-gradient
    # pattern; the book used a two-hidden-layer network for this table.
    if need_initial:
        net = network2.Network(ARCHITECTURES[2])
        write_json(out / "initial_gradient.json", initial_gradient(net, data))
        print("Saved the averaged initial gradient for the top six neurons per layer.")
        if needed[2]:
            write_json(
                out / "norms_during_training_2_layers.json",
                train_and_record(net, data, args),
            )
            needed[2] = False

    for layers in (2, 3, 4):
        if not needed[layers]:
            continue
        net = network2.Network(ARCHITECTURES[layers])
        write_json(
            out / f"norms_during_training_{layers}_layers.json",
            train_and_record(net, data, args),
        )
    return True


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def plot_training(filename: Path, num_layers: int) -> None:
    with filename.open("r", encoding="utf-8") as handle:
        norms = json.load(handle)
    if not norms:
        print(f"Skipping {filename.name}: no data")
        return

    epochs = len(norms)
    fig, ax = plt.subplots()
    for layer in range(min(num_layers, len(norms[0]))):
        ax.plot(
            np.arange(epochs),
            [row[layer] for row in norms],
            color=COLORS[layer],
            label=f"Hidden layer {layer + 1}",
        )
    ax.set_xlim([0, epochs])
    ax.grid(True)
    ax.set_xlabel("Number of epochs of training")
    ax.set_title(f"Speed of learning: {num_layers} hidden layers")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(filename.with_name(f"training_speed_{num_layers}_layers.png"), dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out = args.output_dir

    artifacts = [out / "initial_gradient.json"] + [
        out / f"norms_during_training_{layers}_layers.json" for layers in (2, 3, 4)
    ]
    if args.recompute or any(not path.exists() for path in artifacts):
        computed = compute_all(args, args.recompute)
    else:
        computed = False
    if not computed:
        print("Reusing precomputed JSON (pass --recompute to retrain).")

    for layers in (2, 3, 4):
        filename = out / f"norms_during_training_{layers}_layers.json"
        if filename.exists():
            plot_training(filename, layers)
        else:
            print(f"Skipping {filename.name}: file not found")


if __name__ == "__main__":
    main()
