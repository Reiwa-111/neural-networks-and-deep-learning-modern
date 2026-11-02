"""MNIST loading utilities (modernized, backwards-compatible API).

The original public functions ``load_data``, ``load_data_wrapper`` and
``vectorized_result`` are preserved so older book snippets keep working.
New code should prefer :mod:`nndl.data`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nndl.data import DEFAULT_MNIST_PATH, MNISTBundle, load_mnist, load_raw

__all__ = [
    "DEFAULT_MNIST_PATH",
    "MNISTBundle",
    "load_data",
    "load_data_wrapper",
    "vectorized_result",
]


def load_data(path: str | Path | None = None):
    """Return ``(training_data, validation_data, test_data)``.

    Each split is a ``(images, labels)`` tuple of NumPy arrays.  The pickle is
    read with ``encoding='latin1'`` so the original Python 2 file works on
    modern Python without private pickle APIs.
    """

    return load_raw(path)


def load_data_wrapper(path: str | Path | None = None):
    """Return the book's list-of-pairs format.

    Training pairs are ``(784x1 image, 10x1 one-hot label)``.  Validation and
    test pairs are ``(784x1 image, integer label)``.
    """

    tr_d, va_d, te_d = load_data(path)
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return training_data, validation_data, test_data


def load_data_tensors(
    path: str | Path | None = None,
    *,
    flatten: bool = False,
    normalize: bool = False,
) -> MNISTBundle:
    """Fast path returning PyTorch tensors directly."""

    return load_mnist(path, flatten=flatten, normalize=normalize)


def vectorized_result(j: int) -> np.ndarray:
    """Return a 10-dimensional one-hot column vector."""

    e = np.zeros((10, 1), dtype=np.float32)
    e[int(j)] = 1.0
    return e
