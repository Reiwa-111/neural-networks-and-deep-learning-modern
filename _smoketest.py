"""Quick smoke test for the modernized implementation.

Run from the repository root:

    python _smoketest.py

It exercises the data loader, one MLP epoch and a forward pass of the CNN.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch

from nndl.config import TrainConfig
from nndl.data import load_mnist, make_dataloaders
from nndl.models import MLP, ConvNet
from nndl.trainer import Trainer, make_optimizer


def main() -> int:
    bundle = load_mnist().subset(2000, 500)
    print("Train images:", bundle.train_x.shape, "labels:", bundle.train_y.shape)
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("Labels:", torch.bincount(bundle.train_y, minlength=10).tolist())

    train, val, test = make_dataloaders(bundle, batch_size=128)
    model = MLP([784, 30, 10], hidden_activation="relu")
    config = TrainConfig(epochs=1, batch_size=128, lr=1e-3, optimizer="adamw", device="auto")
    trainer = Trainer(model, torch.nn.CrossEntropyLoss(), make_optimizer(model, config), config)
    history = trainer.fit(train, val, verbose=False)
    _, test_accuracy = trainer.evaluate(test)
    print(f"MLP one epoch: val={history.val_accuracy[-1]:.2%}, test={test_accuracy:.2%}")

    cnn = ConvNet(channels=(8,), kernels=(5,), fc_sizes=(32,))
    assert cnn(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
    print("CNN forward: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
