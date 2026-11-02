"""MNIST loading and DataLoader construction.

The repository ships the classic ``data/mnist.pkl.gz`` file used by the book
(50k train / 10k validation / 10k test, Python 2 pickle).  Modern Python can
read it with ``encoding='latin1'``.  An optional torchvision fallback is
available when the pickle is missing.
"""

from __future__ import annotations

import gzip
import math
import pickle
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_MNIST_PATH = DEFAULT_DATA_DIR / "mnist.pkl.gz"

# Statistics of the book's MNIST pickle (computed from the training split).
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

IMAGE_SHAPE = (1, 28, 28)
DEFAULT_IMAGE_SIZE = 28
NUM_CLASSES = 10


def fit_image_size(images: np.ndarray, target_size: int) -> np.ndarray:
    """Pad (or centre-crop) a batch of square images to ``target_size``.

    ``images`` may be ``(N, H, W)`` or ``(N, 1, H, W)``.  Padding is symmetric
    and uses zeros, which matches the black MNIST background.  The function is
    deliberately dependency-free so it can be used before tensor conversion.
    """

    if target_size < 1:
        raise ValueError("target_size must be positive")
    if images.ndim == 4 or images.ndim == 3:
        height, width = images.shape[-2:]
    else:
        raise ValueError(f"expected (N, H, W) or (N, 1, H, W) images, got {images.shape}")
    if height < 1 or width < 1:
        raise ValueError("image height and width must be positive")
    if height == target_size and width == target_size:
        return images
    if height != width:
        raise ValueError("fit_image_size expects square images")

    if height < target_size:
        pad = target_size - height
        top = pad // 2
        bottom = pad - top
        left = pad // 2
        right = pad - left
        pad_width = [(0, 0)] * (images.ndim - 2) + [(top, bottom), (left, right)]
        return np.pad(images, pad_width, mode="constant")
    if height > target_size:
        top = (height - target_size) // 2
        left = (width - target_size) // 2
        slices = [slice(None)] * (images.ndim - 2) + [
            slice(top, top + target_size),
            slice(left, left + target_size),
        ]
        return images[tuple(slices)]
    return images


@dataclass(slots=True)
class MNISTBundle:
    """In-memory MNIST split as PyTorch tensors.

    ``x`` tensors are ``float32`` with shape ``(N, 1, 28, 28)`` by default,
    ``y`` tensors are ``int64`` with shape ``(N,)``.
    """

    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor

    def to(self, device: torch.device | str) -> MNISTBundle:
        device = torch.device(device)
        return replace(
            self,
            train_x=self.train_x.to(device, non_blocking=True),
            train_y=self.train_y.to(device, non_blocking=True),
            val_x=self.val_x.to(device, non_blocking=True),
            val_y=self.val_y.to(device, non_blocking=True),
            test_x=self.test_x.to(device, non_blocking=True),
            test_y=self.test_y.to(device, non_blocking=True),
        )

    def subset(self, train: int | None = None, val: int | None = None) -> MNISTBundle:
        if train is not None and train < 0:
            raise ValueError("train subset size must be non-negative")
        if val is not None and val < 0:
            raise ValueError("validation subset size must be non-negative")
        return replace(
            self,
            train_x=self.train_x[:train] if train else self.train_x,
            train_y=self.train_y[:train] if train else self.train_y,
            val_x=self.val_x[:val] if val else self.val_x,
            val_y=self.val_y[:val] if val else self.val_y,
        )

    def __len__(self) -> int:
        return len(self.train_y)


def load_raw(path: str | Path | None = None) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Load the raw pickle without any reshaping or normalisation."""

    path = Path(path) if path else DEFAULT_MNIST_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"MNIST file not found: {path}. "
            "Place mnist.pkl.gz in data/ or pass --data /path/to/mnist.pkl.gz."
        )
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _as_tensors(
    split: tuple[np.ndarray, np.ndarray],
    *,
    flatten: bool,
    normalize: bool,
    image_size: int = DEFAULT_IMAGE_SIZE,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = split
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    # The book's pickle always stores 28x28 images; ``image_size`` optionally
    # pads them to a larger canvas (32x32 is the modern default for CNNs).
    x = x.reshape(-1, *IMAGE_SHAPE[1:])
    x = fit_image_size(x, image_size)
    if normalize:
        x = (x - MNIST_MEAN) / MNIST_STD
    if flatten:
        x = x.reshape(len(x), image_size * image_size)
    else:
        x = x.reshape(len(x), 1, image_size, image_size)
    return torch.from_numpy(x).to(dtype=dtype), torch.from_numpy(y)


def load_mnist(
    path: str | Path | None = None,
    *,
    flatten: bool = False,
    normalize: bool = False,
    image_size: int = DEFAULT_IMAGE_SIZE,
    dtype: torch.dtype = torch.float32,
) -> MNISTBundle:
    """Load the book's MNIST split into tensors.

    Parameters
    ----------
    path:
        Pickle file (``.pkl``/``.pkl.gz``) or an expanded ``.npz`` file.
    flatten:
        Return ``(N, image_size**2)`` tensors instead of
        ``(N, 1, image_size, image_size)``.
    normalize:
        Apply the standard MNIST mean/std normalisation.
    image_size:
        Output side length.  The native resolution is 28; values larger than
        28 add a symmetric black border (32 gives a 2-pixel border).
    """

    path = Path(path) if path else DEFAULT_MNIST_PATH
    if path.suffix == ".npz":
        bundle = load_expanded(path, image_size=image_size)
        if normalize:
            bundle = replace(
                bundle,
                train_x=(bundle.train_x - MNIST_MEAN) / MNIST_STD,
                val_x=(bundle.val_x - MNIST_MEAN) / MNIST_STD,
                test_x=(bundle.test_x - MNIST_MEAN) / MNIST_STD,
            )
        if flatten:
            bundle = replace(
                bundle,
                train_x=bundle.train_x.reshape(len(bundle.train_x), -1),
                val_x=bundle.val_x.reshape(len(bundle.val_x), -1),
                test_x=bundle.test_x.reshape(len(bundle.test_x), -1),
            )
        return bundle

    train, val, test = load_raw(path)
    train_x, train_y = _as_tensors(
        train, flatten=flatten, normalize=normalize, image_size=image_size, dtype=dtype
    )
    val_x, val_y = _as_tensors(
        val, flatten=flatten, normalize=normalize, image_size=image_size, dtype=dtype
    )
    test_x, test_y = _as_tensors(
        test, flatten=flatten, normalize=normalize, image_size=image_size, dtype=dtype
    )
    return MNISTBundle(train_x, train_y, val_x, val_y, test_x, test_y)


def load_expanded(path: str | Path, *, image_size: int = DEFAULT_IMAGE_SIZE) -> MNISTBundle:
    """Load the augmented dataset produced by ``expand_mnist.py``.

    Flat arrays are accepted regardless of whether they store 28x28 or 32x32
    images, so files produced by older and newer versions both work.
    """

    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            required = {"train_x", "train_y", "val_x", "val_y", "test_x", "test_y"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")

            def image_tensor(values: np.ndarray) -> torch.Tensor:
                values = values.astype(np.float32, copy=False)
                if values.ndim == 2:
                    side = math.isqrt(values.shape[1])
                    if side < 1 or side * side != values.shape[1]:
                        raise ValueError(
                            f"flat image width must be a perfect square, got {values.shape[1]}"
                        )
                    values = values.reshape(-1, side, side)
                elif values.ndim == 4:
                    if values.shape[1] != 1:
                        raise ValueError(
                            f"expected one image channel, got array shape {values.shape}"
                        )
                    values = values.reshape(len(values), values.shape[-2], values.shape[-1])
                elif values.ndim != 3:
                    raise ValueError(f"unsupported image array shape: {values.shape}")
                values = fit_image_size(values, image_size)
                values = values.reshape(len(values), 1, image_size, image_size)
                return torch.from_numpy(values)

            train_x = image_tensor(data["train_x"])
            train_y = torch.from_numpy(data["train_y"].astype(np.int64, copy=False))
            val_x = image_tensor(data["val_x"])
            val_y = torch.from_numpy(data["val_y"].astype(np.int64, copy=False))
            test_x = image_tensor(data["test_x"])
            test_y = torch.from_numpy(data["test_y"].astype(np.int64, copy=False))

            for split_name, images, labels in (
                ("train", train_x, train_y),
                ("validation", val_x, val_y),
                ("test", test_x, test_y),
            ):
                if labels.ndim != 1:
                    raise ValueError(f"{split_name} labels must be one-dimensional")
                if len(images) != len(labels):
                    raise ValueError(
                        f"{split_name} image and label counts do not match: "
                        f"{len(images)} images, {len(labels)} labels"
                    )
    else:
        train, val, test = load_raw(path)
        train_x, train_y = _as_tensors(train, flatten=False, normalize=False, image_size=image_size)
        val_x, val_y = _as_tensors(val, flatten=False, normalize=False, image_size=image_size)
        test_x, test_y = _as_tensors(test, flatten=False, normalize=False, image_size=image_size)
    return MNISTBundle(train_x, train_y, val_x, val_y, test_x, test_y)


def make_dataloaders(
    bundle: MNISTBundle,
    *,
    batch_size: int = 10,
    eval_batch_size: int | None = None,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    seed: int = 42,
    drop_last: bool = False,
    train_shuffle: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/validation/test loaders from an in-memory bundle."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if eval_batch_size is not None and eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    eval_batch_size = max(batch_size, 256) if eval_batch_size is None else eval_batch_size
    generator = torch.Generator().manual_seed(seed)

    def loader(x: torch.Tensor, y: torch.Tensor, *, shuffle: bool, batch: int) -> DataLoader:
        dataset = TensorDataset(x, y)
        return DataLoader(
            dataset,
            batch_size=batch,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last and shuffle,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            generator=generator if shuffle else None,
        )

    return (
        loader(bundle.train_x, bundle.train_y, shuffle=train_shuffle, batch=batch_size),
        loader(bundle.val_x, bundle.val_y, shuffle=False, batch=eval_batch_size),
        loader(bundle.test_x, bundle.test_y, shuffle=False, batch=eval_batch_size),
    )


def to_one_hot(y: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Convert integer labels to a float one-hot matrix."""

    return torch.nn.functional.one_hot(y, num_classes=num_classes).to(torch.float32)


def iter_batches(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Small typing-friendly alias used by the compatibility modules."""

    yield from loader
