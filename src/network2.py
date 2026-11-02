"""Chapter 3 network: cross-entropy, L2 regularisation and better initialisation.

The original class and method signatures are preserved.  Internally the model
is a PyTorch ``MLP`` and gradients come from autograd, which makes the code
both faster and less error-prone.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nndl.config import TrainConfig, resolve_device
from nndl.losses import BinaryCrossEntropyLoss, QuadraticLoss
from nndl.models import MLP
from nndl.trainer import Trainer

__all__ = [
    "CrossEntropyCost",
    "Network",
    "QuadraticCost",
    "load",
    "sigmoid",
    "sigmoid_prime",
    "vectorized_result",
]


class QuadraticCost:
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(np.asarray(a) - np.asarray(y)) ** 2

    @staticmethod
    def delta(z, a, y):
        return (np.asarray(a) - np.asarray(y)) * sigmoid_prime(z)


class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        a = np.asarray(a)
        y = np.asarray(y)
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return np.asarray(a) - np.asarray(y)


class Network:
    """Fully connected network with the book's chapter 3 features."""

    def __init__(self, sizes: Sequence[int], cost=CrossEntropyCost) -> None:
        self.sizes = list(sizes)
        self.cost = cost if isinstance(cost, type) else type(cost)
        self.device = resolve_device("auto")
        self.model = MLP(
            self.sizes,
            hidden_activation="sigmoid",
            output_activation=None if self.cost is CrossEntropyCost else "sigmoid",
        ).to(self.device)

    # ------------------------------------------------------------------ init
    def default_weight_initializer(self) -> None:
        self.model.reset_parameters("default")

    def large_weight_initializer(self) -> None:
        self.model.reset_parameters("large")

    # ------------------------------------------------------------------ inference
    def feedforward(self, a) -> np.ndarray:
        tensor = torch.as_tensor(np.asarray(a), dtype=torch.float32)
        single = tensor.dim() == 1 or (tensor.dim() == 2 and tensor.shape[1] == 1)
        if single:
            tensor = tensor.reshape(1, -1)
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            # The original API always returns sigmoid activations, regardless
            # of whether the cost is quadratic or cross-entropy.
            output = torch.sigmoid(logits)
        array = output.cpu().numpy()
        return array.reshape(-1, 1) if single else array

    def accuracy(self, data, convert: bool = False) -> int:
        x, y = _pairs_to_tensors(data, one_hot=convert)
        x = x.to(self.device)
        y = y.to(self.device)
        correct = 0
        with torch.inference_mode():
            for start in range(0, len(x), 1024):
                logits = self.model(x[start : start + 1024])
                predictions = logits.argmax(dim=1)
                targets = y[start : start + 1024]
                if targets.dim() > 1:
                    targets = targets.argmax(dim=1)
                correct += int((predictions == targets).sum().item())
        return correct

    def total_cost(self, data, lmbda: float, convert: bool = False) -> float:
        x, y = _pairs_to_tensors(data, one_hot=True if convert else None)
        x = x.to(self.device)
        y = y.to(self.device).to(torch.float32)
        with torch.inference_mode():
            logits = self.model(x)
            if self.cost is CrossEntropyCost:
                data_cost = F.binary_cross_entropy_with_logits(
                    logits, y, reduction="sum"
                ).item() / len(data)
            else:
                data_cost = 0.5 * ((torch.sigmoid(logits) - y) ** 2).sum().item() / len(data)
        l2 = sum(float((linear.weight**2).sum()) for linear in self.model.linears)
        return data_cost + 0.5 * (lmbda / len(data)) * l2

    # ------------------------------------------------------------------ training
    def SGD(
        self,
        training_data,
        epochs: int,
        mini_batch_size: int,
        eta: float,
        lmbda: float = 0.0,
        evaluation_data=None,
        monitor_evaluation_cost: bool = False,
        monitor_evaluation_accuracy: bool = False,
        monitor_training_cost: bool = False,
        monitor_training_accuracy: bool = False,
    ):
        x, y = _pairs_to_tensors(training_data, one_hot=True)
        loader = DataLoader(
            TensorDataset(x, y),
            batch_size=mini_batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )
        n = len(training_data)
        criterion = BinaryCrossEntropyLoss() if self.cost is CrossEntropyCost else QuadraticLoss()
        optimizer = torch.optim.SGD(
            _param_groups(self.model, lmbda, n),
            lr=eta,
        )
        trainer = Trainer(
            self.model,
            criterion,
            optimizer,
            TrainConfig(
                epochs=epochs,
                batch_size=mini_batch_size,
                lr=eta,
                amp=False,
                device=str(self.device),
            ),
        )

        evaluation_cost: list[float] = []
        evaluation_accuracy: list[float] = []
        training_cost: list[float] = []
        training_accuracy: list[float] = []

        for epoch in range(epochs):
            trainer.train_epoch(loader)
            print(f"Epoch {epoch} training complete")
            if monitor_training_cost:
                value = self.total_cost(training_data, lmbda)
                training_cost.append(value)
                print(f"Cost on training data: {value}")
            if monitor_training_accuracy:
                value = self.accuracy(training_data, convert=True)
                training_accuracy.append(value)
                print(f"Accuracy on training data: {value} / {n}")
            if monitor_evaluation_cost:
                assert evaluation_data is not None
                value = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(value)
                print(f"Cost on evaluation data: {value}")
            if monitor_evaluation_accuracy:
                assert evaluation_data is not None
                value = self.accuracy(evaluation_data)
                evaluation_accuracy.append(value)
                print(f"Accuracy on evaluation data: {value} / {len(evaluation_data)}")
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    # ------------------------------------------------------------------ low-level API
    def backprop(self, x, y):
        """Return ``(nabla_b, nabla_w)`` for a single example.

        Kept for compatibility with the book and with scripts that inspect
        gradients.  The computation is a standard PyTorch autograd pass.
        """

        self.model.zero_grad(set_to_none=True)
        x_tensor = (
            torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(1, -1).to(self.device)
        )
        y_tensor = (
            torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(1, -1).to(self.device)
        )
        logits = self.model(x_tensor)
        if self.cost is CrossEntropyCost:
            loss = F.binary_cross_entropy_with_logits(logits, y_tensor, reduction="sum")
        else:
            loss = 0.5 * ((torch.sigmoid(logits) - y_tensor) ** 2).sum()
        loss.backward()
        nabla_w = []
        nabla_b = []
        for linear in self.model.linears:
            nabla_w.append(linear.weight.grad.detach().cpu().numpy().copy())
            nabla_b.append(linear.bias.grad.detach().cpu().numpy().reshape(-1, 1).copy())
        self.model.zero_grad(set_to_none=True)
        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, eta: float, lmbda: float, n: int) -> None:
        nabla_b = [np.zeros((l.bias.shape[0], 1)) for l in self.model.linears]
        nabla_w = [np.zeros(l.weight.shape) for l in self.model.linears]
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_w)]
        with torch.no_grad():
            for linear, nw, nb in zip(self.model.linears, nabla_w, nabla_b):
                linear.weight -= (eta / len(mini_batch)) * torch.as_tensor(
                    nw, dtype=linear.weight.dtype, device=linear.weight.device
                )
                linear.weight -= (eta * lmbda / n) * linear.weight
                if linear.bias is not None:
                    linear.bias -= (eta / len(mini_batch)) * torch.as_tensor(
                        nb, dtype=linear.bias.dtype, device=linear.bias.device
                    ).reshape_as(linear.bias)

    # ------------------------------------------------------------------ persistence
    @property
    def weights(self) -> list[np.ndarray]:
        return [linear.weight.detach().cpu().numpy() for linear in self.model.linears]

    @property
    def biases(self) -> list[np.ndarray]:
        return [linear.bias.detach().cpu().numpy() for linear in self.model.linears]

    def save(self, filename: str | Path) -> None:
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": self.cost.__name__,
        }
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

    def state_dict(self):
        return self.model.state_dict()


def load(filename: str | Path) -> Network:
    with open(filename, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    cost = getattr(sys.modules[__name__], data["cost"])
    network = Network(data["sizes"], cost=cost)
    with torch.no_grad():
        for linear, weights, biases in zip(network.model.linears, data["weights"], data["biases"]):
            linear.weight.copy_(torch.as_tensor(weights, dtype=linear.weight.dtype))
            if linear.bias is not None:
                linear.bias.copy_(torch.as_tensor(biases, dtype=linear.bias.dtype).reshape(-1))
    return network


def _pairs_to_tensors(data: Iterable, *, one_hot: bool | None = None):
    xs = []
    ys = []
    for x, y in data:
        xs.append(np.asarray(x, dtype=np.float32).reshape(-1))
        ys.append(np.asarray(y, dtype=np.float32).reshape(-1))
    x_tensor = torch.from_numpy(np.stack(xs))
    if one_hot is True:
        y_tensor = torch.from_numpy(np.stack(ys))
    elif one_hot is False:
        y_tensor = torch.from_numpy(
            np.asarray([int(np.asarray(y).reshape(-1)[0]) for _, y in data], dtype=np.int64)
        )
    else:
        # Preserve whatever shape the caller supplied; used by total_cost(convert=False).
        y_tensor = torch.from_numpy(np.stack(ys))
    return x_tensor, y_tensor


def _param_groups(model: MLP, lmbda: float, n: int):
    weights = [linear.weight for linear in model.linears]
    biases = [linear.bias for linear in model.linears if linear.bias is not None]
    return [
        {"params": weights, "weight_decay": lmbda / n},
        {"params": biases, "weight_decay": 0.0},
    ]


def vectorized_result(j: int) -> np.ndarray:
    e = np.zeros((10, 1), dtype=np.float32)
    e[int(j)] = 1.0
    return e


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z)))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)
