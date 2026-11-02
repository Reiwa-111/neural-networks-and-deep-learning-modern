"""Classical MNIST baselines: average darkness and linear/RBF SVM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import load_raw


@dataclass(slots=True)
class BaselineResult:
    name: str
    accuracy: float
    correct: int
    total: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "accuracy": self.accuracy,
            "correct": self.correct,
            "total": self.total,
        }


def average_darkness(path: str | Path | None = None) -> BaselineResult:
    """Classify a digit by the closest average pixel sum per class."""

    train, _, test = load_raw(path)
    train_x, train_y = np.asarray(train[0]), np.asarray(train[1])
    test_x, test_y = np.asarray(test[0]), np.asarray(test[1])

    darkness = train_x.sum(axis=1)
    averages = np.array([darkness[train_y == digit].mean() for digit in range(10)])
    predictions = np.abs(test_x.sum(axis=1)[:, None] - averages[None, :]).argmin(axis=1)
    correct = int((predictions == test_y).sum())
    return BaselineResult(
        name="average_darkness",
        accuracy=correct / len(test_y),
        correct=correct,
        total=len(test_y),
    )


def svm_baseline(
    path: str | Path | None = None,
    *,
    limit: int | None = 10000,
    kernel: str = "linear",
    c: float = 1.0,
    scale: bool = True,
    seed: int = 42,
) -> BaselineResult:
    """Train a scikit-learn SVM on the book's MNIST split.

    ``limit=10000`` keeps the default runtime reasonable. Pass ``None`` or a
    non-positive value to train on all 50k examples (linear kernel recommended).
    """

    try:
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC, LinearSVC
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "scikit-learn is required for the SVM baseline: pip install scikit-learn"
        ) from exc

    train, _, test = load_raw(path)
    train_x = np.asarray(train[0], dtype=np.float32)
    train_y = np.asarray(train[1], dtype=np.int64)
    test_x = np.asarray(test[0], dtype=np.float32)
    test_y = np.asarray(test[1], dtype=np.int64)

    if limit is not None and limit > 0:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(train_x), size=min(limit, len(train_x)), replace=False)
        train_x, train_y = train_x[indices], train_y[indices]

    if scale:
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

    if kernel == "linear":
        model = LinearSVC(C=c, max_iter=5000, dual="auto", random_state=seed)
    elif kernel == "rbf":
        model = SVC(C=c, kernel="rbf", cache_size=2048)
    else:
        raise ValueError("kernel must be 'linear' or 'rbf'")

    print(f"Training {kernel} SVM on {len(train_x)} examples...")
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    correct = int(accuracy_score(test_y, predictions) * len(test_y))
    return BaselineResult(
        name=f"svm_{kernel}",
        accuracy=correct / len(test_y),
        correct=correct,
        total=len(test_y),
    )
