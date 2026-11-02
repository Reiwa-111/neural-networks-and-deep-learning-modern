"""Runtime configuration: device selection, reproducibility and training knobs."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to the best available accelerator.

    ROCm builds of PyTorch expose the GPU through the ``cuda`` device namespace,
    so the same code works on NVIDIA, AMD ROCm and Apple MPS hardware.
    """

    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch.

    ``deterministic=True`` is slower but gives bit-reproducible results on most
    hardware.  The default favours throughput (cuDNN benchmark mode).
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True


def describe_device(device: torch.device) -> str:
    """Human-readable device description used by the CLI and training logs."""

    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index or 0
        name = torch.cuda.get_device_name(index)
        hip = getattr(torch.version, "hip", None)
        backend = f"ROCm {hip}" if hip else f"CUDA {torch.version.cuda}"
        return f"{device} ({name}, {backend})"
    return str(device)


@dataclass(slots=True)
class TrainConfig:
    """Hyper-parameters shared by all training entry points.

    Defaults mirror the book's chapter 3 setup: plain SGD, mini-batches of 10,
    learning rate 0.5 and no weight decay.  Modern entry points override these
    defaults with better values where appropriate.
    """

    epochs: int = 30
    batch_size: int = 10
    lr: float = 0.5
    weight_decay: float = 0.0
    momentum: float = 0.0
    optimizer: str = "sgd"
    device: str = "auto"
    seed: int = 42
    num_workers: int = 0
    amp: bool = True
    compile: bool = False
    deterministic: bool = False
    early_stopping_patience: int | None = None
    grad_clip: float | None = None

    def __post_init__(self) -> None:
        if self.epochs < 0:
            raise ValueError("epochs must be non-negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.momentum) or self.momentum < 0:
            raise ValueError("momentum must be finite and non-negative")
        if self.optimizer not in {"sgd", "adam", "adamw"}:
            raise ValueError("optimizer must be one of: sgd, adam, adamw")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if self.grad_clip is not None and (
            not math.isfinite(self.grad_clip) or self.grad_clip <= 0
        ):
            raise ValueError("grad_clip must be finite and positive")
