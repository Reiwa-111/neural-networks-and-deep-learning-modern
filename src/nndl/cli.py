"""Command line interface for training, evaluation and baselines."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from . import __version__
from .augment import expand_dataset
from .baselines import average_darkness, svm_baseline
from .config import TrainConfig, describe_device, resolve_device, set_seed
from .data import load_mnist, make_dataloaders, to_one_hot
from .losses import BinaryCrossEntropyLoss, QuadraticLoss
from .models import MLP, ConvNet
from .trainer import Trainer, make_optimizer


def _runtime_metadata(device: torch.device) -> dict[str, str | None]:
    """Collect the environment details needed to interpret a training run."""

    is_cuda_device = device.type == "cuda" and torch.cuda.is_available()
    return {
        "project_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "device": describe_device(device),
        "gpu_name": torch.cuda.get_device_name(device) if is_cuda_device else None,
        "cuda_version": torch.version.cuda,
        "rocm_version": getattr(torch.version, "hip", None),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nndl",
        description="Modern PyTorch implementations of the Neural Networks and Deep Learning samples.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train an MLP or CNN on MNIST")
    train.add_argument("model", choices=["mlp", "cnn"])
    train.add_argument(
        "--data", type=Path, default=None, help="path to mnist.pkl.gz or mnist_expanded.npz"
    )
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--l2", type=float, default=None, help="L2 regularisation strength")
    train.add_argument("--momentum", type=float, default=None)
    train.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default=None)
    train.add_argument("--hidden", type=int, nargs="+", default=None, help="MLP hidden layer sizes")
    train.add_argument("--activation", default=None, help="relu, sigmoid, tanh, gelu, ...")
    train.add_argument("--dropout", type=float, default=None)
    train.add_argument("--batchnorm", action="store_true")
    train.add_argument("--loss", choices=["ce", "bce", "mse"], default=None)
    train.add_argument("--channels", type=int, nargs="+", default=None)
    train.add_argument("--kernels", type=int, nargs="+", default=None)
    train.add_argument("--fc", type=int, nargs="+", default=None, help="CNN fully connected sizes")
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--output-dir", type=Path, default=None)
    train.add_argument("--quick", action="store_true", help="1 epoch on a small subset")
    train.add_argument(
        "--book", action="store_true", help="reproduce the book's chapter 3 MLP setup"
    )
    train.add_argument("--limit-train", type=int, default=None)
    train.add_argument("--limit-val", type=int, default=None)
    train.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None)
    train.add_argument(
        "--image-size",
        type=int,
        choices=[28, 32],
        default=None,
        help="input canvas size (default: 32 for modern MLP, 28 for CNN/book MLP)",
    )
    train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    train.add_argument("--deterministic", action="store_true")
    train.add_argument("--early-stopping", type=int, default=None)
    train.add_argument("--grad-clip", type=float, default=None)

    baseline = subparsers.add_parser("baseline", help="run a classical baseline")
    baseline.add_argument("kind", choices=["darkness", "svm"])
    baseline.add_argument("--data", type=Path, default=None)
    baseline.add_argument("--limit", type=int, default=10000)
    baseline.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    baseline.add_argument("--c", type=float, default=1.0)
    baseline.add_argument("--no-scale", action="store_true")
    baseline.add_argument("--json", type=Path, default=None)

    augment = subparsers.add_parser("augment", help="build an expanded MNIST/EMNIST training set")
    augment.add_argument("--data", type=Path, default=None)
    augment.add_argument("--output", type=Path, default=None)
    augment.add_argument(
        "--limit", type=int, default=None, help="limit source MNIST training images"
    )
    augment.add_argument("--image-size", type=int, choices=[28, 32], default=32)
    augment.add_argument(
        "--include-emnist",
        action="store_true",
        help="add 240k EMNIST Digits training images (Hugging Face mirror or torchvision)",
    )
    augment.add_argument(
        "--emnist-root", type=Path, default=None, help="EMNIST download/cache directory"
    )
    augment.add_argument(
        "--emnist-source",
        choices=["huggingface", "torchvision"],
        default="torchvision",
        help="Hugging Face partial mirror (smaller, parallel/resumable) or torchvision/NIST archive",
    )
    augment.add_argument(
        "--emnist-limit",
        type=int,
        default=None,
        help="randomly sample this many EMNIST training images (default: all)",
    )

    benchmark = subparsers.add_parser("benchmark", help="quick throughput/accuracy smoke test")
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--epochs", type=int, default=1)
    benchmark.add_argument("--limit-train", type=int, default=5000)
    benchmark.add_argument("--limit-val", type=int, default=1000)

    return parser


def _train_defaults(args: argparse.Namespace) -> dict:
    is_cnn = args.model == "cnn"
    quick = args.quick

    if args.book and is_cnn:
        raise SystemExit("--book is only supported for the mlp model")

    if args.book:
        defaults = {
            "epochs": 30,
            "batch_size": 10,
            "lr": 0.5,
            "l2": 5.0,
            "momentum": 0.0,
            "optimizer": "sgd",
            "hidden": [30],
            "activation": "sigmoid",
            "dropout": 0.0,
            "loss": "bce",
            "normalize": False,
            "image_size": 28,
        }
    elif is_cnn:
        defaults = {
            "epochs": 10,
            "batch_size": 64,
            "lr": 1e-3,
            "l2": 0.0,
            "momentum": 0.9,
            "optimizer": "adamw",
            "hidden": [100],
            "activation": "relu",
            "dropout": 0.5,
            "loss": "ce",
            "normalize": True,
            "channels": [20, 40],
            "kernels": [5, 5],
            "fc": [100],
            "image_size": 28,
        }
    else:
        defaults = {
            "epochs": 50,
            "batch_size": 128,
            "lr": 1e-3,
            "l2": 1e-4,
            "momentum": 0.0,
            "optimizer": "adamw",
            "hidden": [1024, 768, 512, 256, 128],
            "activation": "relu",
            "dropout": 0.1,
            "loss": "ce",
            "normalize": True,
            "image_size": 32,
        }

    if quick:
        defaults.update(epochs=1, batch_size=128, hidden=defaults["hidden"])
        defaults["limit_train"] = 2000
        defaults["limit_val"] = 1000

    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    if args.limit_train is None:
        args.limit_train = defaults.get("limit_train")
    if args.limit_val is None:
        args.limit_val = defaults.get("limit_val")
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = Path("runs") / f"{args.model}-{stamp}"
    return defaults


def _run_train(args: argparse.Namespace) -> int:
    _train_defaults(args)
    if args.model == "cnn" and args.loss != "ce":
        raise SystemExit("The cnn model currently supports --loss ce only")
    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    print(f"Device: {describe_device(device)}")

    is_cnn = args.model == "cnn"
    bundle = load_mnist(
        args.data, flatten=not is_cnn, normalize=args.normalize, image_size=args.image_size
    )
    bundle = bundle.subset(args.limit_train, args.limit_val)

    if args.loss in {"bce", "mse"}:
        bundle = replace(
            bundle,
            train_y=to_one_hot(bundle.train_y),
            val_y=to_one_hot(bundle.val_y),
            test_y=to_one_hot(bundle.test_y),
        )

    train_loader, val_loader, test_loader = make_dataloaders(
        bundle,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    if is_cnn:
        model = ConvNet(
            channels=tuple(args.channels),
            kernels=tuple(args.kernels),
            fc_sizes=tuple(args.fc),
            activation=args.activation,
            dropout=args.dropout,
            batchnorm=args.batchnorm,
            input_size=args.image_size,
        )
        criterion: nn.Module = nn.CrossEntropyLoss()
    else:
        sizes = [args.image_size * args.image_size, *args.hidden, 10]
        model = MLP(
            sizes,
            hidden_activation=args.activation,
            output_activation="sigmoid" if args.loss == "mse" else None,
            dropout=args.dropout,
            batchnorm=args.batchnorm,
        )
        if args.loss == "ce":
            criterion = nn.CrossEntropyLoss()
        elif args.loss == "bce":
            criterion = BinaryCrossEntropyLoss()
        else:
            criterion = QuadraticLoss()

    # In the book, ``lmbda`` is divided by the training-set size before being
    # applied to the weights.  Modern runs expose the usual PyTorch-style
    # weight decay instead; ``--book`` restores the original semantics.
    weight_decay = args.l2 / len(bundle.train_x) if args.book else args.l2
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=weight_decay,
        momentum=args.momentum,
        optimizer=args.optimizer,
        device=str(device),
        seed=args.seed,
        num_workers=args.num_workers,
        amp=args.amp,
        compile=args.compile,
        deterministic=args.deterministic,
        early_stopping_patience=args.early_stopping,
        grad_clip=args.grad_clip,
    )
    optimizer = make_optimizer(model, config)
    trainer = Trainer(model, criterion, optimizer, config, output_dir=args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "args.json").write_text(
        json.dumps(
            {
                "argv": sys.argv,
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "config": asdict(config),
                "environment": _runtime_metadata(device),
                "dataset": {
                    "source": str(args.data) if args.data is not None else "default MNIST pickle",
                    "train_examples": len(bundle.train_y),
                    "validation_examples": len(bundle.val_y),
                    "test_examples": len(bundle.test_y),
                    "input_shape": list(bundle.train_x.shape[1:]),
                },
                "model": {
                    "class": model.__class__.__name__,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "architecture": str(model),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    history = trainer.fit(train_loader, val_loader)
    best_epoch = trainer.restore_best_weights()
    test_loss, test_accuracy = trainer.evaluate(test_loader)
    selected_checkpoint = "best.pt" if best_epoch is not None else "last.pt"
    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "best_val_accuracy": history.best_val_accuracy,
        "epochs": len(history.train_loss),
        "selected_checkpoint": selected_checkpoint,
        "selected_epoch": best_epoch + 1 if best_epoch is not None else None,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"Test accuracy: {test_accuracy:.2%} "
        f"(best validation: {history.best_val_accuracy:.2%}; "
        f"checkpoint: {selected_checkpoint}"
    )
    print(f"Artifacts: {args.output_dir}")
    return 0


def _run_baseline(args: argparse.Namespace) -> int:
    if args.kind == "darkness":
        result = average_darkness(args.data)
    else:
        result = svm_baseline(
            args.data,
            limit=args.limit,
            kernel=args.kernel,
            c=args.c,
            scale=not args.no_scale,
        )
    print(f"{result.name}: {result.accuracy:.2%} ({result.correct}/{result.total})")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return 0


def _run_augment(args: argparse.Namespace) -> int:
    expand_dataset(
        args.data,
        args.output,
        limit=args.limit,
        image_size=args.image_size,
        include_emnist=args.include_emnist,
        emnist_root=args.emnist_root,
        emnist_limit=args.emnist_limit,
        emnist_source=args.emnist_source,
    )
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    common = {
        "data": None,
        "epochs": args.epochs,
        "device": args.device,
        "quick": False,
        "book": False,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "seed": 42,
        "num_workers": 0,
        "output_dir": Path("runs") / "benchmark",
        "amp": True,
        "compile": False,
        "deterministic": False,
        "early_stopping": None,
        "grad_clip": None,
        "normalize": None,
        "activation": None,
        "dropout": None,
        "hidden": None,
        "channels": None,
        "kernels": None,
        "fc": None,
        "loss": None,
        "l2": None,
        "lr": None,
        "momentum": None,
        "optimizer": None,
        "batch_size": None,
        "batchnorm": False,
    }
    for model_name in ("mlp", "cnn"):
        args.model = model_name
        run_args = dict(common, output_dir=Path("runs") / f"benchmark-{model_name}")
        _run_train(argparse.Namespace(**run_args, model=model_name))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        return _run_train(args)
    if args.command == "baseline":
        return _run_baseline(args)
    if args.command == "augment":
        return _run_augment(args)
    if args.command == "benchmark":
        return _run_benchmark(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
