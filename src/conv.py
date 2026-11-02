"""Chapter 6 experiments: convolutional networks on MNIST.

This is the modern replacement for the original Theano-based ``conv.py``.
Every experiment uses :class:`nndl.models.ConvNet` and the shared training
loop, so it runs on CPU, NVIDIA CUDA and AMD ROCm without code changes.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nndl.config import TrainConfig, set_seed
from nndl.data import MNISTBundle, load_mnist, make_dataloaders
from nndl.models import MLP, ConvNet
from nndl.trainer import History, Trainer, make_optimizer

DEFAULT_DATA = Path("data/mnist_expanded.npz")


@dataclass
class ExperimentResult:
    name: str
    trainer: Trainer
    test_accuracy: float
    history: History
    expanded: bool = False


def _bundle(*, expanded: bool = False, quick: bool = False) -> MNISTBundle:
    path = DEFAULT_DATA if expanded else None
    if expanded and not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/expand_mnist.py` or `nndl augment` first."
        )
    bundle = load_mnist(path, normalize=True)
    if quick:
        bundle = bundle.subset(2000, 1000)
    return bundle


def _run_model(
    name: str,
    model: nn.Module,
    bundle: MNISTBundle,
    *,
    epochs: int,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer_name: str = "adamw",
    device: str = "auto",
    amp: bool = True,
    seed: int = 42,
    expanded: bool = False,
) -> ExperimentResult:
    set_seed(seed)
    train_loader, val_loader, test_loader = make_dataloaders(
        bundle, batch_size=batch_size, seed=seed
    )
    config = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        optimizer=optimizer_name,
        device=device,
        seed=seed,
        amp=amp,
    )
    optimizer = make_optimizer(model, config)
    trainer = Trainer(
        model,
        nn.CrossEntropyLoss(),
        optimizer,
        config,
        output_dir=Path("runs") / "conv" / name,
    )
    history = trainer.fit(train_loader, val_loader)
    _, test_accuracy = trainer.evaluate(test_loader)
    print(f"{name}: test accuracy {test_accuracy:.2%}")
    return ExperimentResult(name, trainer, test_accuracy, history, expanded)


def shallow(n: int = 3, epochs: int = 60, *, quick: bool = False, device: str = "auto"):
    """A shallow 784-100-10 network used as a baseline."""

    results = []
    bundle = _bundle(quick=quick)
    for j in range(n):
        model = MLP([784, 100, 10], hidden_activation="relu")
        results.append(
            _run_model(
                f"shallow-{j}",
                model,
                bundle,
                epochs=epochs,
                device=device,
                expanded=False,
            )
        )
    return results


def basic_conv(n: int = 3, epochs: int = 60, *, quick: bool = False, device: str = "auto"):
    """One convolution/pooling stage followed by a fully connected layer."""

    results = []
    bundle = _bundle(quick=quick)
    for j in range(n):
        model = ConvNet(
            channels=(20,),
            kernels=(5,),
            fc_sizes=(100,),
            activation="relu",
        )
        results.append(_run_model(f"basic-conv-{j}", model, bundle, epochs=epochs, device=device))
    return results


def omit_FC(n: int = 3, epochs: int = 60, *, quick: bool = False, device: str = "auto"):
    """Convolution/pooling followed directly by softmax."""

    results = []
    bundle = _bundle(quick=quick)
    for j in range(n):
        model = ConvNet(
            channels=(20,),
            kernels=(5,),
            fc_sizes=(),
            activation="relu",
        )
        results.append(_run_model(f"omit-fc-{j}", model, bundle, epochs=epochs, device=device))
    return results


def dbl_conv(
    activation_fn: str = "sigmoid",
    n: int = 3,
    epochs: int = 60,
    *,
    quick: bool = False,
    device: str = "auto",
):
    """Two convolution/pooling stages plus one fully connected layer."""

    results = []
    bundle = _bundle(quick=quick)
    for j in range(n):
        model = ConvNet(
            channels=(20, 40),
            kernels=(5, 5),
            fc_sizes=(100,),
            activation=activation_fn,
        )
        results.append(
            _run_model(
                f"dbl-conv-{activation_fn}-{j}",
                model,
                bundle,
                epochs=epochs,
                device=device,
            )
        )
    return results


def regularized_dbl_conv(
    lambdas: Sequence[float] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    n: int = 1,
    epochs: int = 60,
    *,
    quick: bool = False,
    device: str = "auto",
):
    results = []
    bundle = _bundle(quick=quick)
    for lmbda in lambdas:
        for j in range(n):
            model = ConvNet(
                channels=(20, 40),
                kernels=(5, 5),
                fc_sizes=(100,),
                activation="sigmoid",
            )
            results.append(
                _run_model(
                    f"regularized-{lmbda:g}-{j}",
                    model,
                    bundle,
                    epochs=epochs,
                    weight_decay=lmbda,
                    device=device,
                )
            )
    return results


def dbl_conv_relu(n: int = 3, epochs: int = 60, *, quick: bool = False, device: str = "auto"):
    return dbl_conv("relu", n=n, epochs=epochs, quick=quick, device=device)


def expanded_data(
    n_fc: int = 100, n: int = 3, epochs: int = 60, *, quick: bool = False, device: str = "auto"
):
    """Train on the 250k-image expanded dataset."""

    results = []
    bundle = _bundle(expanded=True, quick=quick)
    for j in range(n):
        model = ConvNet(
            channels=(20, 40),
            kernels=(5, 5),
            fc_sizes=(n_fc,),
            activation="relu",
        )
        results.append(
            _run_model(
                f"expanded-{n_fc}-{j}",
                model,
                bundle,
                epochs=epochs,
                device=device,
                expanded=True,
            )
        )
    return results


def expanded_data_double_fc(
    n_fc: int = 100, n: int = 3, epochs: int = 60, *, quick: bool = False, device: str = "auto"
):
    results = []
    bundle = _bundle(expanded=True, quick=quick)
    for j in range(n):
        model = ConvNet(
            channels=(20, 40),
            kernels=(5, 5),
            fc_sizes=(n_fc, n_fc),
            activation="relu",
        )
        results.append(
            _run_model(
                f"expanded-double-{n_fc}-{j}",
                model,
                bundle,
                epochs=epochs,
                device=device,
                expanded=True,
            )
        )
    return results


def double_fc_dropout(
    p0: float,
    p1: float,
    p2: float,
    repetitions: int,
    *,
    epochs: int = 40,
    quick: bool = False,
    device: str = "auto",
):
    results = []
    bundle = _bundle(expanded=True, quick=quick)
    for j in range(repetitions):
        print(f"Training dropout network {j + 1}/{repetitions}: p=({p0}, {p1}, {p2})")
        model = ConvNet(
            channels=(20, 40),
            kernels=(5, 5),
            fc_sizes=(1000, 1000),
            activation="relu",
            dropout=(p0, p1),
            output_dropout=p2,
        )
        results.append(
            _run_model(
                f"dropout-{p0}-{p1}-{p2}-{j}",
                model,
                bundle,
                epochs=epochs,
                device=device,
                expanded=True,
            )
        )
    return results


def ensemble(results: Sequence[ExperimentResult]) -> tuple[list[int], list[int]]:
    """Majority vote over several trained networks."""

    if not results:
        raise ValueError("ensemble requires at least one trained model")
    bundle = load_mnist(normalize=True)
    loader = DataLoader(
        TensorDataset(bundle.test_x, bundle.test_y),
        batch_size=1024,
        shuffle=False,
    )
    predictions = torch.stack([result.trainer.predict(loader) for result in results])
    majority = torch.mode(predictions, dim=0).values
    errors = (majority != bundle.test_y).nonzero(as_tuple=False).flatten()
    print(f"Ensemble accuracy: {1 - len(errors) / len(bundle.test_y):.2%}")
    return errors.tolist(), majority[errors].tolist()


def plot_errors(
    error_locations: Sequence[int],
    erroneous_predictions: Sequence[int] | None = None,
    *,
    output: str | Path = "ensemble_errors.png",
) -> Path:
    bundle = load_mnist(normalize=True)
    n = min(40, len(error_locations))
    fig, axes = plt.subplots(5, 8, figsize=(10, 6.5))
    axes = axes.ravel()
    for axis in axes:
        axis.axis("off")
    for j in range(n):
        index = error_locations[j]
        axes[j].imshow(bundle.test_x[index, 0].numpy(), cmap="gray")
        axes[j].set_title(str(int(bundle.test_y[index])), fontsize=8)
        if erroneous_predictions is not None:
            axes[j].set_xlabel(str(int(erroneous_predictions[j])), fontsize=8, color="red")
    fig.tight_layout()
    output = Path(output)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_filters(
    net,
    layer: int,
    x: int,
    y: int,
    *,
    output: str | Path = "net_filters.png",
) -> Path:
    model = net.trainer.unwrap_model() if isinstance(net, ExperimentResult) else net
    filters = model.conv_layers[layer].weight.detach().cpu()
    fig, axes = plt.subplots(y, x, figsize=(x * 1.5, y * 1.5))
    axes = np.asarray(axes).ravel()
    for axis in axes:
        axis.axis("off")
    for j in range(min(len(filters), len(axes))):
        axes[j].imshow(filters[j, 0].numpy(), cmap="gray")
    fig.tight_layout()
    output = Path(output)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def run_experiments(*, quick: bool = False, device: str = "auto") -> None:
    n = 1 if quick else 3
    epochs = 2 if quick else 60
    shallow(n=n, epochs=epochs, quick=quick, device=device)
    basic_conv(n=n, epochs=epochs, quick=quick, device=device)
    omit_FC(n=n, epochs=epochs, quick=quick, device=device)
    dbl_conv(n=n, epochs=epochs, quick=quick, device=device)
    dbl_conv_relu(n=n, epochs=epochs, quick=quick, device=device)
    try:
        expanded_data(n_fc=100, n=n, epochs=epochs, quick=quick, device=device)
        expanded_data_double_fc(n_fc=100, n=n, epochs=epochs, quick=quick, device=device)
        nets = double_fc_dropout(
            0.5, 0.5, 0.5, repetitions=n, epochs=epochs, quick=quick, device=device
        )
        errors, predictions = ensemble(nets)
        print(plot_errors(errors, predictions))
        print(plot_filters(nets[0], 0, 5, 4, output="net_full_layer_0.png"))
        print(plot_filters(nets[0], 1, 8, 5, output="net_full_layer_1.png"))
    except FileNotFoundError as exc:
        print(f"Skipping expanded-data experiments: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        nargs="?",
        default="all",
        choices=[
            "all",
            "shallow",
            "basic_conv",
            "omit_fc",
            "dbl_conv",
            "dbl_conv_relu",
            "regularized",
            "expanded",
            "expanded_double_fc",
            "dropout",
        ],
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    set_seed(args.seed)

    quick = args.quick
    epochs = args.epochs if args.epochs is not None else (2 if quick else 60)
    n = 1 if quick else 3

    if args.experiment == "all":
        run_experiments(quick=quick, device=args.device)
    elif args.experiment == "shallow":
        shallow(n=n, epochs=epochs, quick=quick, device=args.device)
    elif args.experiment == "basic_conv":
        basic_conv(n=n, epochs=epochs, quick=quick, device=args.device)
    elif args.experiment == "omit_fc":
        omit_FC(n=n, epochs=epochs, quick=quick, device=args.device)
    elif args.experiment == "dbl_conv":
        dbl_conv(n=n, epochs=epochs, quick=quick, device=args.device)
    elif args.experiment == "dbl_conv_relu":
        dbl_conv_relu(n=n, epochs=epochs, quick=quick, device=args.device)
    elif args.experiment == "regularized":
        regularized_dbl_conv(n=n, epochs=epochs, quick=quick, device=args.device)
    elif args.experiment == "expanded":
        expanded_data(n_fc=100, n=n, epochs=epochs, quick=quick, device=args.device)
    elif args.experiment == "expanded_double_fc":
        expanded_data_double_fc(n_fc=100, n=n, epochs=epochs, quick=quick, device=args.device)
    elif args.experiment == "dropout":
        nets = double_fc_dropout(
            0.5, 0.5, 0.5, repetitions=n, epochs=epochs, quick=quick, device=args.device
        )
        errors, predictions = ensemble(nets)
        print(plot_errors(errors, predictions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
