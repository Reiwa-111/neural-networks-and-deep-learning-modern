"""Chapter 1 network: a minimal fully connected classifier.

The public API is unchanged (``Network(sizes).SGD(...)``), but gradients are
now computed by PyTorch and the model runs on the best available device.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nndl.config import TrainConfig, resolve_device
from nndl.losses import QuadraticLoss, ZeroLoss
from nndl.models import MLP
from nndl.trainer import Trainer

__all__ = ["Network", "sigmoid", "sigmoid_prime"]


def _as_input(value) -> torch.Tensor:
    array = np.asarray(value, dtype=np.float32)
    return torch.from_numpy(array)


class Network:
    """Feedforward network with sigmoid units and a quadratic cost."""

    def __init__(self, sizes: Sequence[int]) -> None:
        self.sizes = list(sizes)
        self.device = resolve_device("auto")
        self.model = MLP(
            self.sizes,
            hidden_activation="sigmoid",
            output_activation="sigmoid",
        ).to(self.device)

    # ------------------------------------------------------------------ inference
    def feedforward(self, a) -> np.ndarray:
        """Return the network output for a single input (or a batch)."""

        tensor = _as_input(a)
        single = tensor.dim() == 1 or (tensor.dim() == 2 and tensor.shape[1] == 1)
        if single:
            tensor = tensor.reshape(1, -1)
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            output = self.model(tensor).cpu().numpy()
        return output.reshape(-1, 1) if single else output

    def evaluate(self, test_data) -> int:
        """Number of correctly classified examples."""

        x, y = _split_pairs(test_data, one_hot=False)
        loader = DataLoader(
            TensorDataset(x, y),
            batch_size=1024,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        trainer = Trainer(
            self.model,
            ZeroLoss(),
            torch.optim.SGD(self.model.parameters(), lr=0.0),
            TrainConfig(epochs=0, batch_size=1024, lr=1e-12, amp=False, device=str(self.device)),
        )
        _, accuracy = trainer.evaluate(loader)
        return round(accuracy * len(y))

    # ------------------------------------------------------------------ training
    def SGD(
        self,
        training_data,
        epochs: int,
        mini_batch_size: int,
        eta: float,
        test_data=None,
    ) -> None:
        """Train with mini-batch stochastic gradient descent."""

        x, y = _split_pairs(training_data, one_hot=True)
        loader = DataLoader(
            TensorDataset(x, y),
            batch_size=mini_batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )
        optimizer = torch.optim.SGD(self.model.parameters(), lr=eta)
        trainer = Trainer(
            self.model,
            QuadraticLoss(),
            optimizer,
            TrainConfig(
                epochs=epochs,
                batch_size=mini_batch_size,
                lr=eta,
                amp=False,
                device=str(self.device),
            ),
        )

        test_loader = None
        if test_data:
            test_x, test_y = _split_pairs(test_data, one_hot=False)
            test_loader = DataLoader(
                TensorDataset(test_x, test_y),
                batch_size=1024,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )
            # The training objective is quadratic on one-hot targets; test
            # labels are integers, so accuracy is measured with ZeroLoss.
            test_trainer = Trainer(
                self.model,
                ZeroLoss(),
                torch.optim.SGD(self.model.parameters(), lr=0.0),
                TrainConfig(
                    epochs=0, batch_size=1024, lr=1e-12, amp=False, device=str(self.device)
                ),
            )

        n_test = len(test_data) if test_data else 0
        for epoch in range(epochs):
            started = time.perf_counter()
            trainer.train_epoch(loader)
            elapsed = time.perf_counter() - started
            if test_loader is not None:
                _, accuracy = test_trainer.evaluate(test_loader)
                correct = round(accuracy * n_test)
                print(f"Epoch {epoch}: {correct} / {n_test}, took {elapsed:.2f} seconds")
            else:
                print(f"Epoch {epoch} complete in {elapsed:.2f} seconds")

    # ------------------------------------------------------------------ persistence
    def save(self, filename: str | Path) -> None:
        torch.save(
            {"sizes": self.sizes, "state_dict": self.model.state_dict()},
            filename,
        )

    @classmethod
    def load(cls, filename: str | Path) -> Network:
        checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
        network = cls(checkpoint["sizes"])
        network.model.load_state_dict(checkpoint["state_dict"])
        return network

    # ------------------------------------------------------------------ compatibility
    def cost_derivative(self, output_activations, y):
        return np.asarray(output_activations) - np.asarray(y)


def _split_pairs(data: Iterable, *, one_hot: bool):
    """Convert the book's list-of-pairs into flattened tensors."""

    xs = []
    ys = []
    labels = []
    for x, y in data:
        xs.append(np.asarray(x, dtype=np.float32).reshape(-1))
        ys.append(np.asarray(y, dtype=np.float32).reshape(-1))
        labels.append(int(np.asarray(y).reshape(-1)[0]))
    x_tensor = torch.from_numpy(np.stack(xs))
    if one_hot:
        y_tensor = torch.from_numpy(np.stack(ys))
    else:
        y_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    return x_tensor, y_tensor


def sigmoid(z):
    """The sigmoid function (NumPy compatible)."""

    return 1.0 / (1.0 + np.exp(-np.asarray(z)))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""

    s = sigmoid(z)
    return s * (1 - s)
