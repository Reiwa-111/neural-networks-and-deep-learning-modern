"""Dataset augmentation for MNIST.

Two mechanisms are provided:

``RandomAffine2D``
    On-the-fly, per-sample translation/rotation/scale jitter.  It runs on the
    same device as the batch and is fully vectorised, so the effective training
    set is unbounded without allocating extra memory.

``expand_images`` / ``expand_dataset``
    The book's classic 5x expansion (original plus four one-pixel shifts),
    vectorised and optionally resized to a larger canvas (for example 32x32).
"""

from __future__ import annotations

import gzip
import math
import pickle
import re
import shutil
import struct
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .data import DEFAULT_DATA_DIR, fit_image_size, load_raw

# (shift, axis) for the classic 5x expansion.
_SHIFTS = ((1, 1), (-1, 1), (1, 2), (-1, 2))


class RandomAffine2D(nn.Module):
    """Random per-sample affine jitter for ``(N, C, H, W)`` image batches.

    Parameters
    ----------
    max_translate:
        Maximum absolute translation in pixels.
    max_rotate:
        Maximum absolute rotation in degrees.
    max_scale:
        Maximum relative scale change (``0.1`` means +/-10%).
    padding_mode:
        ``border`` (default) replicates the padded border; ``zeros`` fills with
        the mean value of the input.  MNIST images are surrounded by a zero
        border after :func:`nndl.data.load_mnist`, so ``border`` behaves like
        black padding while avoiding a grey frame on normalised inputs.
    """

    def __init__(
        self,
        max_translate: float = 2.0,
        max_rotate: float = 10.0,
        max_scale: float = 0.0,
        padding_mode: str = "border",
    ) -> None:
        super().__init__()
        if max_translate < 0 or max_rotate < 0 or max_scale < 0:
            raise ValueError("augmentation strengths must be non-negative")
        if max_scale >= 1:
            raise ValueError("max_scale must be less than 1.0")
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError("padding_mode must be 'zeros', 'border', or 'reflection'")
        self.max_translate = float(max_translate)
        self.max_rotate = float(max_rotate)
        self.max_scale = float(max_scale)
        self.padding_mode = padding_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected (N, C, H, W) input, got shape {tuple(x.shape)}")
        if self.max_translate == 0 and self.max_rotate == 0 and self.max_scale == 0:
            return x

        n, _, height, width = x.shape
        device, dtype = x.device, x.dtype

        angle = (torch.rand(n, device=device, dtype=dtype) * 2 - 1) * (
            self.max_rotate * math.pi / 180.0
        )
        scale = 1.0 + (torch.rand(n, device=device, dtype=dtype) * 2 - 1) * self.max_scale
        cos = torch.cos(angle) * scale
        sin = torch.sin(angle) * scale

        # grid_sample coordinates span [-1, 1], so one pixel equals 2/(size-1).
        tx = (torch.rand(n, device=device, dtype=dtype) * 2 - 1) * self.max_translate
        ty = (torch.rand(n, device=device, dtype=dtype) * 2 - 1) * self.max_translate
        tx = tx * 2.0 / max(width - 1, 1)
        ty = ty * 2.0 / max(height - 1, 1)

        theta = torch.zeros(n, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = cos
        theta[:, 0, 1] = -sin
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin
        theta[:, 1, 1] = cos
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=False,
        )

    def extra_repr(self) -> str:
        return (
            f"max_translate={self.max_translate}, max_rotate={self.max_rotate}, "
            f"max_scale={self.max_scale}, padding_mode={self.padding_mode!r}"
        )


def expand_images(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    image_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return five copies of every image: original plus four one-pixel shifts.

    Parameters
    ----------
    images:
        ``(N, 784)`` or ``(N, 28, 28)`` array.
    image_size:
        Optional output side length.  ``None`` keeps the original resolution;
        ``32`` pads the images to 32x32 before shifting.
    """

    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 2:
        side = math.isqrt(images.shape[1])
        if side < 1 or side * side != images.shape[1]:
            raise ValueError(f"flat image width must be a perfect square, got {images.shape[1]}")
        images = images.reshape(-1, side, side)
    elif images.ndim != 3:
        raise ValueError(f"expected images shaped (N, H, W) or (N, H*W), got {images.shape}")
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(images) != len(labels):
        raise ValueError("image and label counts do not match")
    if images.shape[1] != images.shape[2]:
        raise ValueError("expand_images expects square images")
    if image_size is not None:
        images = fit_image_size(images, image_size)

    n, height, width = images.shape
    expanded_x = np.empty((n * 5, height * width), dtype=np.float32)
    expanded_y = np.empty(n * 5, dtype=np.int64)
    expanded_x[0::5] = images.reshape(n, -1)
    expanded_y[0::5] = labels

    for offset, (shift, axis) in enumerate(_SHIFTS, start=1):
        shifted = np.roll(images, shift, axis=axis)
        if axis == 1:
            shifted[:, 0 if shift > 0 else -1, :] = 0.0
        else:
            shifted[:, :, 0 if shift > 0 else -1] = 0.0
        expanded_x[offset::5] = shifted.reshape(n, -1)
        expanded_y[offset::5] = labels

    return expanded_x, expanded_y


def prepare_emnist_digits(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    image_size: int = 28,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare EMNIST Digits arrays for the MNIST pipeline.

    Torchvision exposes EMNIST images in the dataset's stored orientation;
    transposing the spatial axes restores the upright digit orientation.
    Output images are flattened float32 values in [0, 1].
    """

    images = np.asarray(images)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"expected EMNIST images shaped (N, 28, 28), got {images.shape}")
    if len(images) != len(labels):
        raise ValueError("EMNIST image and label counts do not match")
    if labels.size and (labels.min() < 0 or labels.max() > 9):
        raise ValueError("EMNIST Digits labels must be in the range 0..9")

    # EMNIST's IDX images are stored with the two spatial axes transposed.
    images = images.transpose(0, 2, 1).astype(np.float32, copy=False)
    if images.size and images.max() > 1.0:
        images = images / 255.0
    images = fit_image_size(images, image_size)
    return images.reshape(len(images), -1), labels


_EMNIST_HF_REVISION = "e1ce0ec"
_EMNIST_HF_BASE = f"https://huggingface.co/datasets/Heliosoph/EMNIST/resolve/{_EMNIST_HF_REVISION}/"
_EMNIST_BYCLASS_IMAGES = "emnist-byclass-train-images-idx3-ubyte.gz"
_EMNIST_BYCLASS_LABELS = "emnist-byclass-train-labels-idx1-ubyte.gz"


class _RangeNotSupported(RuntimeError):
    pass


def _download_hf_file(url: str, destination: Path, *, workers: int = 8) -> Path:
    """Download a Hugging Face file using resumable parallel HTTP ranges.

    Falls back to a single resumable stream if the mirror/CDN does not support
    byte ranges. Temporary chunks are kept after interruption and reused on the
    next invocation.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "nndl-emnist-downloader/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            total = int(response.headers.get("Content-Length", "0"))
            range_supported = "bytes" in response.headers.get("Accept-Ranges", "").lower()
    except (urllib.error.URLError, TimeoutError, ValueError):
        total, range_supported = 0, False

    # Some object-storage gateways omit Accept-Ranges on HEAD but support it
    # on GET. Probe one byte before giving up on parallel transfer.
    if not range_supported or total <= 0:
        probe = urllib.request.Request(
            url,
            headers={"User-Agent": "nndl-emnist-downloader/1.0", "Range": "bytes=0-0"},
        )
        try:
            with urllib.request.urlopen(probe, timeout=30) as response:
                match = re.match(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
                if getattr(response, "status", response.getcode()) == 206 and match:
                    total = int(match.group(1))
                    range_supported = True
        except (urllib.error.URLError, TimeoutError, ValueError):
            pass

    parts_dir = destination.with_name(destination.name + ".parts")
    if total > 0 and range_supported:
        parts_dir.mkdir(parents=True, exist_ok=True)
        chunk_size = (total + workers - 1) // workers
        ranges = [
            (i, i * chunk_size, min(total - 1, (i + 1) * chunk_size - 1))
            for i in range(workers)
            if i * chunk_size < total
        ]
        progress_lock = Lock()
        progress = tqdm(
            total=total,
            initial=sum(
                min(part.stat().st_size, end - start + 1)
                for i, start, end in ranges
                for part in [parts_dir / f"part-{i:02d}"]
                if part.exists()
            ),
            unit="B",
            unit_scale=True,
            desc=f"{destination.name}",
        )

        def fetch_range(item: tuple[int, int, int]) -> None:
            index, start, end = item
            part = parts_dir / f"part-{index:02d}"
            expected = end - start + 1
            existing = part.stat().st_size if part.exists() else 0
            if existing > expected:
                part.unlink()
                existing = 0
            if existing == expected:
                return
            offset = start + existing
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "nndl-emnist-downloader/1.0",
                    "Range": f"bytes={offset}-{end}",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                if getattr(response, "status", response.getcode()) != 206:
                    raise _RangeNotSupported("server ignored HTTP Range requests")
                content_range = response.headers.get("Content-Range", "")
                match = re.match(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if not match or int(match.group(1)) != offset or int(match.group(3)) != total:
                    raise _RangeNotSupported("server returned an invalid Content-Range")
                with part.open("ab") as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                        with progress_lock:
                            progress.update(len(block))
            if part.stat().st_size != expected:
                raise OSError(
                    f"incomplete download chunk {index}: {part.stat().st_size}/{expected}"
                )

        try:
            with ThreadPoolExecutor(max_workers=min(workers, len(ranges))) as pool:
                futures = [pool.submit(fetch_range, item) for item in ranges]
                for future in as_completed(futures):
                    future.result()
            temp = destination.with_name(destination.name + ".assembling")
            with temp.open("wb") as output:
                for index, _, _ in ranges:
                    with (parts_dir / f"part-{index:02d}").open("rb") as chunk:
                        shutil.copyfileobj(chunk, output, length=4 * 1024 * 1024)
            if temp.stat().st_size != total:
                temp.unlink(missing_ok=True)
                raise OSError(f"download size mismatch for {destination.name}")
            temp.replace(destination)
            shutil.rmtree(parts_dir, ignore_errors=True)
            progress.close()
            return destination
        except _RangeNotSupported:
            progress.close()
            shutil.rmtree(parts_dir, ignore_errors=True)
        except Exception:
            progress.close()
            raise

    # Sequential fallback with HTTP resume support.
    partial = destination.with_name(destination.name + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "nndl-emnist-downloader/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                append = existing > 0 and getattr(response, "status", response.getcode()) == 206
                if existing and not append:
                    existing = 0
                mode = "ab" if append else "wb"
                length = int(response.headers.get("Content-Length", "0")) + existing
                with (
                    partial.open(mode) as handle,
                    tqdm(
                        total=length or None,
                        initial=existing,
                        unit="B",
                        unit_scale=True,
                        desc=f"{destination.name}",
                    ) as progress,
                ):
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                        progress.update(len(block))
            if partial.stat().st_size == 0:
                raise OSError(f"empty download: {url}")
            partial.replace(destination)
            return destination
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 4:
                raise
            time.sleep(2**attempt)
            existing = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "nndl-emnist-downloader/1.0"}
            if existing:
                headers["Range"] = f"bytes={existing}-"
            req = urllib.request.Request(url, headers=headers)
    raise RuntimeError(f"could not download {url}")


def _read_emnist_byclass_digits(
    images_path: Path, labels_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Read the digit classes (IDs 0..9) from EMNIST ByClass IDX gzip files."""
    with gzip.open(labels_path, "rb") as handle:
        label_header = handle.read(8)
        if len(label_header) != 8:
            raise ValueError("truncated EMNIST labels header")
        label_magic, label_count = struct.unpack(">II", label_header)
        if label_magic != 2049:
            raise ValueError(f"invalid EMNIST label IDX magic: {label_magic}")
        labels_bytes = handle.read()
    labels = np.frombuffer(labels_bytes, dtype=np.uint8)
    if len(labels) != label_count:
        raise ValueError(f"EMNIST label count mismatch: header={label_count}, data={len(labels)}")

    with gzip.open(images_path, "rb") as handle:
        image_header = handle.read(16)
        if len(image_header) != 16:
            raise ValueError("truncated EMNIST images header")
        image_magic, image_count, rows, columns = struct.unpack(">IIII", image_header)
        if image_magic != 2051 or (rows, columns) != (28, 28):
            raise ValueError("invalid EMNIST image IDX header")
        images_bytes = handle.read()
    if image_count != label_count or len(images_bytes) != image_count * rows * columns:
        raise ValueError("EMNIST image and label counts do not match")

    # Torchvision's ByClass mapping is sorted ASCII: 0..9, A..Z, a..z.
    digit_mask = labels <= 9
    digit_count = int(digit_mask.sum())
    if digit_count == 0:
        raise ValueError("no digit classes found in EMNIST ByClass labels")
    images = np.frombuffer(images_bytes, dtype=np.uint8).reshape(image_count, 28, 28)
    return images[digit_mask], labels[digit_mask].astype(np.int64, copy=False)


def _load_emnist_digits_huggingface(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Download only the two ByClass training IDX files from a HF mirror.

    This avoids torchvision's 562 MB all-splits NIST ZIP. The mirror files are
    173 MB (images) + 492 KB (labels); labels 0..9 select the digit classes.
    """
    cache = root / "huggingface-byclass"
    print(f"Downloading EMNIST from Hugging Face mirror (cache: {cache})")
    images_path = _download_hf_file(
        _EMNIST_HF_BASE + _EMNIST_BYCLASS_IMAGES, cache / _EMNIST_BYCLASS_IMAGES
    )
    labels_path = _download_hf_file(
        _EMNIST_HF_BASE + _EMNIST_BYCLASS_LABELS, cache / _EMNIST_BYCLASS_LABELS
    )
    print("Extracting digit classes 0..9 from EMNIST ByClass")
    return _read_emnist_byclass_digits(images_path, labels_path)


def expand_dataset(
    source: str | Path | None = None,
    output: str | Path | None = None,
    *,
    limit: int | None = None,
    seed: int = 42,
    image_size: int = 28,
    include_emnist: bool = False,
    emnist_root: str | Path | None = None,
    emnist_limit: int | None = None,
    emnist_source: str = "torchvision",
) -> Path:
    """Create an expanded MNIST dataset, optionally adding EMNIST Digits."""

    source = Path(source) if source else DEFAULT_DATA_DIR / "mnist.pkl.gz"
    output = Path(output) if output else DEFAULT_DATA_DIR / "mnist_expanded.npz"
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when specified")
    if output.exists():
        print(f"{output} already exists; nothing to do.")
        return output

    print(f"Loading {source}")
    train, val, test = load_raw(source)
    train_x, train_y = train
    if limit:
        train_x, train_y = train_x[:limit], train_y[:limit]

    print(
        f"Expanding {len(train_x)} MNIST images to {len(train_x) * 5} at {image_size}x{image_size}"
    )
    expanded_x, expanded_y = expand_images(train_x, train_y, image_size=image_size)

    emnist_count = 0
    if include_emnist:
        root = Path(emnist_root) if emnist_root else DEFAULT_DATA_DIR / "emnist"
        if emnist_source == "huggingface":
            emnist_images, emnist_labels = _load_emnist_digits_huggingface(root)
        elif emnist_source == "torchvision":
            try:
                from torchvision.datasets import EMNIST
            except (
                ImportError,
                RuntimeError,
            ) as exc:  # pragma: no cover - optional binary dependency
                raise RuntimeError(
                    "EMNIST support requires a torchvision build compatible with your PyTorch/ROCm. "
                    "Install the matching torchvision wheel, then retry."
                ) from exc
            print(f"Loading EMNIST Digits training split (download/cache: {root})")
            emnist = EMNIST(root=str(root), split="digits", train=True, download=True)
            emnist_images = emnist.data.cpu().numpy()
            emnist_labels = emnist.targets.cpu().numpy()
        else:
            raise ValueError("emnist_source must be 'torchvision' or 'huggingface'")
        if emnist_limit is not None:
            if emnist_limit < 1:
                raise ValueError("emnist_limit must be positive")
            if emnist_limit > len(emnist_labels):
                raise ValueError(
                    f"emnist_limit={emnist_limit} exceeds available examples ({len(emnist_labels)})"
                )
            # EMNIST files are ordered by source/class; sample before taking a subset.
            sample_indices = np.random.default_rng(seed).choice(
                len(emnist_labels), size=emnist_limit, replace=False
            )
            emnist_images = emnist_images[sample_indices]
            emnist_labels = emnist_labels[sample_indices]
        emnist_x, emnist_y = prepare_emnist_digits(
            emnist_images, emnist_labels, image_size=image_size
        )
        emnist_count = len(emnist_y)
        expanded_x = np.concatenate((expanded_x, emnist_x), axis=0)
        expanded_y = np.concatenate((expanded_y, emnist_y), axis=0)
        print(f"Added {emnist_count} unique EMNIST training images")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(expanded_x))
    expanded_x = expanded_x[order]
    expanded_y = expanded_y[order]

    val_x = fit_image_size(np.asarray(val[0], dtype=np.float32).reshape(-1, 28, 28), image_size)
    test_x = fit_image_size(np.asarray(test[0], dtype=np.float32).reshape(-1, 28, 28), image_size)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".npz":
        np.savez_compressed(
            output,
            train_x=expanded_x,
            train_y=expanded_y,
            val_x=val_x,
            val_y=np.asarray(val[1], dtype=np.int64),
            test_x=test_x,
            test_y=np.asarray(test[1], dtype=np.int64),
        )
    else:
        payload = (
            (expanded_x, expanded_y),
            (val_x.reshape(len(val_x), -1), np.asarray(val[1], dtype=np.int64)),
            (test_x.reshape(len(test_x), -1), np.asarray(test[1], dtype=np.int64)),
        )
        with gzip.open(output, "wb") as handle:
            pickle.dump(payload, handle, protocol=4)
    print(
        f"Saved {output}: {len(expanded_y):,} training images "
        f"({len(train_x) * 5:,} shifted MNIST + {emnist_count:,} EMNIST), "
        f"{len(val[1]):,} validation, {len(test[1]):,} test"
    )
    return output
