"""How MNIST accuracy improves as the training set grows.

Trains the book's 784-30-10 network for each training-set size and compares it
with an SVM baseline.  Plots ``more_data.png``, ``more_data_log.png`` and
``more_data_comparison.png``.  The committed JSON files are reused unless
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
from nndl.data import MNISTBundle, load_mnist, load_raw, make_dataloaders, to_one_hot
from nndl.losses import BinaryCrossEntropyLoss
from nndl.models import MLP
from nndl.trainer import Trainer, make_optimizer

# The training-set sizes used in the book.  The JSON files must keep nine
# entries so that they stay compatible with the original figures.
SIZES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
NETWORK_FILE = "more_data.json"
SVM_FILE = "more_data_svm.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--recompute", action="store_true", help="retrain instead of reusing the JSON files"
    )
    parser.add_argument(
        "--quick", action="store_true", help="one epoch per size, capped SVM and validation sets"
    )
    parser.add_argument("--epochs", type=int, default=None, help="override epochs per network")
    parser.add_argument(
        "--svm-limit", type=int, default=None, help="max SVM training examples (0 = no limit)"
    )
    parser.add_argument(
        "--validation-limit", type=int, default=None, help="max validation examples for the SVM"
    )
    parser.add_argument("--device", default="auto", help="torch device, e.g. cpu or cuda")
    parser.add_argument("--seed", type=int, default=12345678)
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args(argv)

    if args.quick:
        if args.epochs is None:
            args.epochs = 1
        if args.svm_limit is None:
            args.svm_limit = 1000
        if args.validation_limit is None:
            args.validation_limit = 1000
    else:
        if args.svm_limit is None:
            args.svm_limit = 5000
        if args.validation_limit is None:
            args.validation_limit = 10000
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
    set_seed(args.seed)
    bundle = one_hot_bundle(load_mnist(flatten=True))
    accuracies: list[float] = []
    for size in SIZES:
        epochs = args.epochs or max(1, int(1_500_000 / size))
        print(f"\nTraining network with data set size {size} ({epochs} epochs)")
        subset = bundle.subset(train=size)
        train_loader, val_loader, _ = make_dataloaders(
            subset, batch_size=10, eval_batch_size=1024, seed=args.seed
        )
        model = MLP([784, 30, 10], hidden_activation="sigmoid", output_activation=None)
        model.reset_parameters("large")  # the book's large_weight_initializer()
        config = TrainConfig(
            epochs=epochs,
            batch_size=10,
            lr=0.5,
            weight_decay=0.0001,  # book: lmbda = size * 0.0001, divided by n
            device=args.device,
            seed=args.seed,
        )
        trainer = Trainer(
            model,
            BinaryCrossEntropyLoss(),
            make_optimizer(model, config),
            config,
        )
        trainer.fit(train_loader, val_loader=None, verbose=False)
        _, accuracy = trainer.evaluate(val_loader)
        percent = 100.0 * accuracy
        print(f"Accuracy was {percent:.2f} percent")
        accuracies.append(percent)
    write_json(filename, accuracies)


def run_svms(args: argparse.Namespace, filename: Path) -> None:
    try:
        from sklearn.svm import SVC
    except ImportError:
        print(f"scikit-learn is not installed; leaving {filename.name} untouched.")
        return

    training, validation, _ = load_raw()
    train_x = np.asarray(training[0], dtype=np.float32)
    train_y = np.asarray(training[1], dtype=np.int64)
    val_x = np.asarray(validation[0], dtype=np.float32)
    val_y = np.asarray(validation[1], dtype=np.int64)
    if args.validation_limit:
        val_x = val_x[: args.validation_limit]
        val_y = val_y[: args.validation_limit]

    accuracies: list[float] = []
    for size in SIZES:
        n = min(size, args.svm_limit) if args.svm_limit else size
        print(f"\nTraining SVM with {n} examples (requested size {size})")
        model = SVC()
        model.fit(train_x[:n], train_y[:n])
        predictions = model.predict(val_x)
        percent = 100.0 * float(np.mean(predictions == val_y))
        print(f"Accuracy was {percent:.2f} percent")
        accuracies.append(percent)
    write_json(filename, accuracies)


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_linear_plot(accuracies: list[float], filename: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(SIZES, accuracies, color="#2A6EA6")
    ax.plot(SIZES, accuracies, "o", color="#FFA933")
    ax.set_xlim(0, 50000)
    ax.set_ylim(60, 100)
    ax.grid(True)
    ax.set_xlabel("Training set size")
    ax.set_title("Accuracy (%) on the validation data")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def make_log_plot(accuracies: list[float], filename: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(SIZES, accuracies, color="#2A6EA6")
    ax.plot(SIZES, accuracies, "o", color="#FFA933")
    ax.set_xlim(100, 50000)
    ax.set_ylim(60, 100)
    ax.set_xscale("log")
    ax.grid(True)
    ax.set_xlabel("Training set size")
    ax.set_title("Accuracy (%) on the validation data")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def make_combined_plot(
    accuracies: list[float], svm_accuracies: list[float], filename: Path
) -> None:
    fig, ax = plt.subplots()
    ax.plot(SIZES, accuracies, color="#2A6EA6")
    ax.plot(SIZES, accuracies, "o", color="#2A6EA6", label="Neural network accuracy (%)")
    ax.plot(SIZES, svm_accuracies, color="#FFA933")
    ax.plot(SIZES, svm_accuracies, "o", color="#FFA933", label="SVM accuracy (%)")
    ax.set_xlim(100, 50000)
    ax.set_ylim(25, 100)
    ax.set_xscale("log")
    ax.grid(True)
    ax.set_xlabel("Training set size")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def make_digit_examples(output_dir: Path) -> None:
    """Save the example 5 and a rotated 5 used in the accompanying text."""
    try:
        training, _, _ = load_raw()
    except FileNotFoundError as exc:
        print(f"Skipping digit examples: {exc}")
        return

    images = np.asarray(training[0], dtype=np.float32)
    labels = np.asarray(training[1])
    index = int(np.flatnonzero(labels == 5)[0])
    image = images[index].reshape(28, 28)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray_r", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_dir / "more_data_5.png", dpi=150)
    plt.close(fig)

    try:
        from PIL import Image

        rotated = Image.fromarray(np.uint8(image * 255)).rotate(30, resample=Image.BICUBIC)
        rotated_image = np.asarray(rotated, dtype=np.float32) / 255.0
    except ImportError:
        rotated_image = np.rot90(image)
    fig, ax = plt.subplots()
    ax.imshow(rotated_image, cmap="gray_r", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_dir / "more_data_rotated_5.png", dpi=150)
    plt.close(fig)


def make_plots(output_dir: Path) -> None:
    network_path = output_dir / NETWORK_FILE
    svm_path = output_dir / SVM_FILE
    if not network_path.exists():
        print(f"Missing {network_path}; pass --recompute to train the networks.")
        return

    accuracies = load_json(network_path)
    make_linear_plot(accuracies, output_dir / "more_data.png")
    make_log_plot(accuracies, output_dir / "more_data_log.png")
    if svm_path.exists():
        make_combined_plot(accuracies, load_json(svm_path), output_dir / "more_data_comparison.png")
    else:
        print(f"Missing {svm_path.name}; skipping the comparison plot.")
    make_digit_examples(output_dir)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    network_path = output_dir / NETWORK_FILE
    svm_path = output_dir / SVM_FILE
    if args.recompute or not network_path.exists():
        run_networks(args, network_path)
    if args.recompute or not svm_path.exists():
        run_svms(args, svm_path)
    if not args.recompute and network_path.exists() and svm_path.exists():
        print("Reusing precomputed JSON (pass --recompute to retrain).")
    make_plots(output_dir)


if __name__ == "__main__":
    main()
