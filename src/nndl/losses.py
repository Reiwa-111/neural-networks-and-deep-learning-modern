"""Loss functions that keep the original book's semantics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class QuadraticLoss(nn.Module):
    """``0.5 * ||a - y||^2`` averaged over the mini-batch.

    This matches the gradient used by ``network.py``: the derivative with
    respect to the output activation is ``(a - y) / batch_size``.
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(input.dtype)
        return 0.5 * ((input - target) ** 2).sum() / input.size(0)


class BinaryCrossEntropyLoss(nn.Module):
    """Binary cross-entropy on logits, averaged over the mini-batch.

    The original ``network2.py`` applies a sigmoid to every output unit and
    sums ``-y log(a) - (1-y) log(1-a)``.  ``binary_cross_entropy_with_logits``
    is the numerically stable equivalent; dividing by the batch size gives the
    same per-example gradient as the book's update rule.
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(input.dtype)
        return F.binary_cross_entropy_with_logits(input, target, reduction="sum") / input.size(0)


class LogSoftmaxNLLLoss(nn.Module):
    """NLL loss for models whose final layer already applies ``log_softmax``."""

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.NLLLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(input, target)


class ZeroLoss(nn.Module):
    """Accuracy-only criterion used when labels are integers but the model
    was trained with a one-hot objective."""

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), dtype=input.dtype, device=input.device)
