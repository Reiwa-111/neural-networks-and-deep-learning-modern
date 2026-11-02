"""Chapter 6 convolutional network with a PyTorch backend.

The original Theano implementation exposed ``Network`` plus three layer
descriptors (``ConvPoolLayer``, ``FullyConnectedLayer``, ``SoftmaxLayer``).
Those names and constructor arguments are preserved, but training now uses
PyTorch autograd and the GPU.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nndl.config import TrainConfig, describe_device, resolve_device
from nndl.data import load_expanded, load_mnist
from nndl.losses import LogSoftmaxNLLLoss
from nndl.models import ConvPoolStack, resolve_activation
from nndl.trainer import Trainer, make_optimizer

__all__ = [
    "GPU",
    "ConvPoolLayer",
    "FullyConnectedLayer",
    "Network",
    "ReLU",
    "SoftmaxLayer",
    "dropout_layer",
    "linear",
    "load_data_shared",
    "sigmoid",
    "size",
    "tanh",
]

GPU = True


def linear(z):
    return z


def sigmoid(z):
    return torch.sigmoid(z) if isinstance(z, torch.Tensor) else 1.0 / (1.0 + np.exp(-z))


def tanh(z):
    return torch.tanh(z) if isinstance(z, torch.Tensor) else np.tanh(z)


def ReLU(z):
    return torch.relu(z) if isinstance(z, torch.Tensor) else np.maximum(0, z)


def load_data_shared(filename: str | Path | None = None):
    """Return ``[(x, y), (x, y), (x, y)]`` tensors for train/validation/test.

    The original function returned Theano shared variables.  PyTorch tensors
    are the modern equivalent: they are moved to the GPU by the training loop.
    """

    path = Path(filename) if filename else None
    if path is not None and path.suffix == ".npz":
        bundle = load_expanded(path)
    else:
        bundle = load_mnist(path)
    return [
        (bundle.train_x, bundle.train_y),
        (bundle.val_x, bundle.val_y),
        (bundle.test_x, bundle.test_y),
    ]


@dataclass
class ConvPoolLayer:
    filter_shape: tuple[int, int, int, int]
    image_shape: tuple[int, int, int, int]
    poolsize: tuple[int, int] = (2, 2)
    activation_fn: object = sigmoid

    def build(self) -> list[nn.Module]:
        out_channels, in_channels, kernel_h, kernel_w = self.filter_shape
        if kernel_h != kernel_w:
            raise ValueError("Only square kernels are supported")
        return [
            ConvPoolStack(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_h,
                pool_size=self.poolsize,
                activation=self.activation_fn,
            )
        ]


@dataclass
class FullyConnectedLayer:
    n_in: int
    n_out: int
    activation_fn: object = sigmoid
    p_dropout: float = 0.0

    def build(self) -> list[nn.Module]:
        modules: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(self.n_in, self.n_out),
            resolve_activation(self.activation_fn),
        ]
        if self.p_dropout > 0:
            modules.append(nn.Dropout(self.p_dropout))
        return modules


@dataclass
class SoftmaxLayer:
    n_in: int
    n_out: int
    p_dropout: float = 0.0

    def build(self) -> list[nn.Module]:
        modules: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(self.n_in, self.n_out),
        ]
        if self.p_dropout > 0:
            modules.append(nn.Dropout(self.p_dropout))
        modules.append(nn.LogSoftmax(dim=1))
        return modules


class Network:
    """Convolutional network built from the layer descriptors above."""

    def __init__(
        self,
        layers: Sequence[object],
        mini_batch_size: int,
        *,
        device: str | torch.device = "auto",
    ) -> None:
        self.layers = list(layers)
        self.mini_batch_size = int(mini_batch_size)
        modules: list[nn.Module] = []
        for layer in self.layers:
            if not hasattr(layer, "build"):
                raise TypeError(f"Unsupported layer descriptor: {layer!r}")
            modules.extend(layer.build())
        self.model = nn.Sequential(*modules)
        self.device = resolve_device(device)
        self.model.to(self.device)
        self.test_mb_predictions = None
        self._test_data = None

    def SGD(
        self,
        training_data,
        epochs: int,
        mini_batch_size: int,
        eta: float,
        validation_data,
        test_data,
        lmbda: float = 0.0,
    ) -> None:
        """Train with mini-batch SGD and report validation/test accuracy."""

        train_x, train_y = _as_xy(training_data)
        val_x, val_y = _as_xy(validation_data)
        test_x, test_y = _as_xy(test_data)
        self._test_data = (test_x, test_y)

        num_training_batches = len(train_x) // mini_batch_size
        if num_training_batches == 0:
            raise ValueError("mini_batch_size is larger than the training set")

        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=mini_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            TensorDataset(val_x, val_y),
            batch_size=max(mini_batch_size, 256),
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            TensorDataset(test_x, test_y),
            batch_size=max(mini_batch_size, 256),
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        config = TrainConfig(
            epochs=epochs,
            batch_size=mini_batch_size,
            lr=eta,
            weight_decay=lmbda / num_training_batches,
            amp=True,
            device=str(self.device),
        )
        optimizer = make_optimizer(self.model, config)
        trainer = Trainer(
            self.model,
            LogSoftmaxNLLLoss(),
            optimizer,
            config,
        )

        print(f"Device: {describe_device(self.device)}")
        best_validation_accuracy = 0.0
        best_iteration = 0
        test_accuracy = 0.0
        for epoch in range(epochs):
            started = time.perf_counter()
            trainer.train_epoch(train_loader)
            val_loss, validation_accuracy = trainer.evaluate(val_loader)
            elapsed = time.perf_counter() - started
            print(
                f"Epoch {epoch}: validation accuracy {validation_accuracy:.2%} "
                f"(loss {val_loss:.4f}, {elapsed:.2f}s)"
            )
            if validation_accuracy >= best_validation_accuracy:
                print("This is the best validation accuracy to date.")
                best_validation_accuracy = validation_accuracy
                best_iteration = epoch
                _, test_accuracy = trainer.evaluate(test_loader)
                print(f"The corresponding test accuracy is {test_accuracy:.2%}")

        print("Finished training network.")
        print(
            f"Best validation accuracy of {best_validation_accuracy:.2%} obtained at iteration {best_iteration}"
        )
        print(f"Corresponding test accuracy of {test_accuracy:.2%}")

    def predict(self, data) -> np.ndarray:
        """Predict labels for a ``(x, y)`` pair or a tensor of images."""

        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
        x = _as_x(x)
        loader = DataLoader(
            TensorDataset(x, torch.zeros(len(x), dtype=torch.long)),
            batch_size=1024,
            shuffle=False,
        )
        return (
            Trainer(
                self.model,
                LogSoftmaxNLLLoss(),
                torch.optim.SGD(self.model.parameters(), lr=0.0),
                TrainConfig(
                    epochs=0, batch_size=1024, lr=1e-12, amp=False, device=str(self.device)
                ),
            )
            .predict(loader)
            .numpy()
        )


def _as_x(x) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    else:
        x = x.float()
    if x.dim() == 2 and x.shape[1] == 784:
        x = x.reshape(-1, 1, 28, 28)
    return x


def _as_xy(data) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = data
    x = _as_x(x)
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(np.asarray(y), dtype=torch.long)
    return x, y.reshape(-1).long()


def size(data) -> int:
    """Return the number of examples in a ``(x, y)`` pair."""

    return len(data[0])


def dropout_layer(layer, p_dropout: float):
    """Compatibility shim; modern modules apply dropout internally."""

    return layer
