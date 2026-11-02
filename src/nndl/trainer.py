"""Reusable training loop with AMP, checkpointing and early stopping."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig, describe_device, resolve_device

Batch = tuple[torch.Tensor, torch.Tensor]


def _labels(target: torch.Tensor) -> torch.Tensor:
    """Accept integer labels or one-hot rows and return integer labels."""

    if target.dim() > 1:
        return target.argmax(dim=1)
    return target


@dataclass(slots=True)
class History:
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)

    def append(
        self,
        *,
        train_loss: float,
        train_accuracy: float,
        val_loss: float | None = None,
        val_accuracy: float | None = None,
        seconds: float | None = None,
    ) -> None:
        self.train_loss.append(float(train_loss))
        self.train_accuracy.append(float(train_accuracy))
        if val_loss is not None:
            self.val_loss.append(float(val_loss))
            self.val_accuracy.append(float(val_accuracy))
        if seconds is not None:
            self.epoch_seconds.append(float(seconds))

    @property
    def best_val_accuracy(self) -> float:
        return max(self.val_accuracy) if self.val_accuracy else 0.0

    def to_dict(self) -> dict[str, list[float]]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> History:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{key: list(data.get(key, [])) for key in cls.__dataclass_fields__})


class Trainer:
    """A compact, explicit training loop.

    The class intentionally exposes ``train_epoch`` and ``evaluate`` so the
    compatibility modules (``network2``/``network3``) can reproduce the book's
    per-epoch monitoring without reimplementing the loop.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainConfig,
        *,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.model = model.to(self.device)
        if config.compile and hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model)
            except Exception as exc:  # noqa: BLE001  # pragma: no cover - hardware dependent
                print(f"torch.compile unavailable, continuing eagerly: {exc}")
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
            if config.optimizer in {"adam", "adamw"} and config.epochs > 1
            else None
        )
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.amp = bool(config.amp and self.device.type == "cuda")
        if self.amp:
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.use_scaler = self.amp_dtype == torch.float16
        else:
            self.amp_dtype = torch.float32
            self.use_scaler = False
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)

        self.history = History()
        self.start_epoch = 0
        self.best_val_accuracy = 0.0

    # ------------------------------------------------------------------ helpers
    def _autocast(self):
        if self.amp:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return nullcontext()

    def unwrap_model(self) -> nn.Module:
        model = self.model
        if hasattr(model, "_orig_mod"):
            return model._orig_mod
        return model

    def _batch_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(output, target)

    # ------------------------------------------------------------------ loops
    def train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                output = self.model(x)
                loss = self._batch_loss(output, y)
            if self.use_scaler:
                self.scaler.scale(loss).backward()
                if self.config.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            batch = y.size(0)
            total_loss += float(loss.detach()) * batch
            total_correct += int((output.argmax(dim=1) == _labels(y)).sum().item())
            total += batch
        return total_loss / max(total, 1), total_correct / max(total, 1)

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with self._autocast():
                output = self.model(x)
                loss = self._batch_loss(output, y)
            batch = y.size(0)
            total_loss += float(loss) * batch
            total_correct += int((output.argmax(dim=1) == _labels(y)).sum().item())
            total += batch
        return total_loss / max(total, 1), total_correct / max(total, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        *,
        epochs: int | None = None,
        verbose: bool = True,
    ) -> History:
        epochs = self.config.epochs if epochs is None else epochs
        patience = self.config.early_stopping_patience
        stale_epochs = 0

        if verbose:
            print(f"Device: {describe_device(self.device)}")
            print(
                f"Epochs: {epochs}, batch size: {self.config.batch_size}, "
                f"lr: {self.config.lr}, optimizer: {self.config.optimizer}, "
                f"amp: {self.amp}"
            )

        for epoch in range(self.start_epoch, epochs):
            started = time.perf_counter()
            train_loss, train_accuracy = self.train_epoch(train_loader)
            elapsed = time.perf_counter() - started
            if val_loader is not None:
                val_loss, val_accuracy = self.evaluate(val_loader)
            else:
                val_loss, val_accuracy = math.nan, math.nan

            self.history.append(
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=None if math.isnan(val_loss) else val_loss,
                val_accuracy=None if math.isnan(val_accuracy) else val_accuracy,
                seconds=elapsed,
            )

            if verbose:
                message = (
                    f"Epoch {epoch:3d}: train loss {train_loss:.4f}, train acc {train_accuracy:.2%}"
                )
                if val_loader is not None:
                    message += f", val loss {val_loss:.4f}, val acc {val_accuracy:.2%}"
                message += f" ({elapsed:.2f}s)"
                print(message)

            if self.scheduler is not None:
                self.scheduler.step()

            if val_loader is not None:
                improved = val_accuracy > self.best_val_accuracy
                if improved:
                    self.best_val_accuracy = val_accuracy
                    stale_epochs = 0
                    if self.output_dir:
                        self.save(self.output_dir / "best.pt", epoch=epoch)
                else:
                    stale_epochs += 1

                if self.output_dir:
                    self.save(self.output_dir / "last.pt", epoch=epoch)
                    self.history.save(self.output_dir / "history.json")

                if patience is not None and stale_epochs >= patience:
                    if verbose:
                        print(f"Early stopping after {patience} epochs without improvement.")
                    break

        if self.output_dir:
            self.history.save(self.output_dir / "history.json")
        return self.history

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def predict(self, loader: DataLoader) -> torch.Tensor:
        self.model.eval()
        predictions: list[torch.Tensor] = []
        for x, _ in loader:
            x = x.to(self.device, non_blocking=True)
            predictions.append(self.model(x).argmax(dim=1).cpu())
        return torch.cat(predictions) if predictions else torch.empty(0, dtype=torch.long)

    # ------------------------------------------------------------------ checkpoints
    def save(self, path: str | Path, *, epoch: int | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.unwrap_model().state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "history": self.history.to_dict(),
                "config": asdict(self.config),
                "epoch": epoch,
                "best_val_accuracy": self.best_val_accuracy,
            },
            path,
        )

    def load(self, path: str | Path, *, load_optimizer: bool = True) -> dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.unwrap_model().load_state_dict(checkpoint["model_state"])
        if load_optimizer and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "history" in checkpoint:
            self.history = History(**{k: list(v) for k, v in checkpoint["history"].items()})
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        self.best_val_accuracy = float(checkpoint.get("best_val_accuracy", 0.0))
        return checkpoint

    def restore_best_weights(self) -> int | None:
        """Restore the weights selected by validation, without rewinding history.

        Returns the zero-based epoch stored in ``best.pt``.  If this trainer has
        no output directory or no best checkpoint, the current weights remain
        unchanged and ``None`` is returned.
        """

        if self.output_dir is None:
            return None
        path = self.output_dir / "best.pt"
        if not path.is_file():
            return None
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.unwrap_model().load_state_dict(checkpoint["model_state"])
        return int(checkpoint["epoch"]) if checkpoint.get("epoch") is not None else None


def make_optimizer(
    model: nn.Module,
    config: TrainConfig,
    *,
    no_decay_on_bias: bool = True,
) -> torch.optim.Optimizer:
    """Create an optimizer, optionally excluding biases from weight decay.

    The book applies L2 regularisation only to weights, never to biases.  The
    two parameter groups below preserve that behaviour for the compatibility
    modules and for modern training runs.
    """

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if no_decay_on_bias and (name.endswith("bias") or "norm" in name.lower()):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    groups = [{"params": decay, "weight_decay": config.weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})

    if config.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=config.lr, momentum=config.momentum)
    if config.optimizer == "adam":
        return torch.optim.Adam(groups, lr=config.lr)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=config.lr)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str = "auto",
) -> tuple[float, float]:
    """Stateless evaluation helper used by scripts and tests."""

    trainer = Trainer.__new__(Trainer)
    trainer.config = TrainConfig()
    trainer.device = resolve_device(device)
    trainer.model = model.to(trainer.device)
    trainer.criterion = criterion
    trainer.amp = False
    trainer.amp_dtype = torch.float32
    trainer.use_scaler = False
    return trainer.evaluate(loader)
