from __future__ import annotations

import numpy as np
import pytest
import torch

from nndl.augment import expand_images, prepare_emnist_digits
from nndl.baselines import average_darkness


def test_expand_images_shapes():
    x = np.zeros((3, 784), dtype=np.float32)
    x[:, 14 * 28 + 14] = 1.0
    y = np.array([1, 2, 3], dtype=np.int64)
    expanded_x, expanded_y = expand_images(x, y)
    assert expanded_x.shape == (15, 784)
    assert expanded_y.shape == (15,)
    assert np.array_equal(expanded_y[0::5], y)
    # original copy is at index 0 of each group
    assert np.array_equal(expanded_x[0], x[0])


def test_expand_images_rejects_invalid_input_shapes():
    with pytest.raises(ValueError, match="perfect square"):
        expand_images(np.zeros((2, 1000), dtype=np.float32), np.array([0, 1]))
    with pytest.raises(ValueError, match="counts do not match"):
        expand_images(np.zeros((2, 28, 28), dtype=np.float32), np.array([0]))


def test_expand_images_shifts_each_image_spatially_without_mixing_samples():
    x = np.zeros((2, 28, 28), dtype=np.float32)
    x[0, 10, 10] = 1.0
    x[1, 20, 20] = 1.0
    y = np.array([3, 8], dtype=np.int64)

    expanded_x, expanded_y = expand_images(x, y)
    expanded = expanded_x.reshape(2, 5, 28, 28)

    expected_positions = [(10, 10), (11, 10), (9, 10), (10, 11), (10, 9)]
    for variant, position in enumerate(expected_positions):
        assert np.argwhere(expanded[0, variant] == 1.0).tolist() == [list(position)]
        assert np.argwhere(expanded[1, variant] == 1.0).tolist() == [
            [position[0] + 10, position[1] + 10]
        ]
    assert expanded_y.reshape(2, 5).tolist() == [[3] * 5, [8] * 5]


def test_average_darkness(mnist_path):
    result = average_darkness(mnist_path)
    assert result.total == 10000
    assert result.correct == 2225
    assert abs(result.accuracy - 0.2225) < 1e-6


def test_network_compatibility(mnist_path):
    import network

    net = network.Network([784, 30, 10])
    x = torch.zeros(784, 1).numpy()
    assert net.feedforward(x).shape == (10, 1)
    # Evaluate on a tiny slice; accuracy can be anything, but the API must work.
    import mnist_loader

    _, _, test = mnist_loader.load_data_wrapper(mnist_path)
    assert 0 <= net.evaluate(test[:5]) <= 5


def test_prepare_emnist_digits_transposes_normalizes_and_pads():
    images = np.zeros((2, 28, 28), dtype=np.uint8)
    images[0, 3, 7] = 255
    labels = np.array([2, 9], dtype=np.int64)

    prepared_x, prepared_y = prepare_emnist_digits(images, labels, image_size=32)

    assert prepared_x.shape == (2, 1024)
    assert prepared_y.tolist() == [2, 9]
    prepared_image = prepared_x[0].reshape(32, 32)
    # 2px border + transposed original coordinate (3, 7) -> (9, 5).
    assert prepared_image[9, 5] == 1.0
    assert prepared_image.sum() == 1.0


def test_prepare_emnist_digits_rejects_non_digit_labels():
    images = np.zeros((1, 28, 28), dtype=np.uint8)
    labels = np.array([10], dtype=np.int64)

    try:
        prepare_emnist_digits(images, labels)
    except ValueError as exc:
        assert "0..9" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected out-of-range EMNIST labels to be rejected")


def test_expand_dataset_merges_emnist_without_touching_mnist_eval_splits(tmp_path, monkeypatch):
    import gzip
    import pickle
    import sys
    from types import ModuleType

    import torch

    from nndl.augment import expand_dataset

    source = tmp_path / "mnist.pkl.gz"
    train = (np.zeros((2, 784), dtype=np.float32), np.array([1, 2]))
    val = (np.zeros((1, 784), dtype=np.float32), np.array([3]))
    test = (np.zeros((1, 784), dtype=np.float32), np.array([4]))
    with gzip.open(source, "wb") as handle:
        pickle.dump((train, val, test), handle, protocol=4)

    class FakeEMNIST:
        def __init__(self, *, root, split, train, download):
            assert split == "digits" and train is True and download is True
            self.data = torch.zeros((3, 28, 28), dtype=torch.uint8)
            self.data[0, 2, 5] = 255
            self.targets = torch.tensor([0, 1, 9])

    fake_datasets = ModuleType("torchvision.datasets")
    fake_datasets.EMNIST = FakeEMNIST
    fake_torchvision = ModuleType("torchvision")
    fake_torchvision.datasets = fake_datasets
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.datasets", fake_datasets)
    output = tmp_path / "combined.npz"
    expand_dataset(source, output, image_size=32, include_emnist=True)

    with np.load(output) as saved:
        assert saved["train_x"].shape == (13, 1024)  # 2 MNIST * 5 + 3 EMNIST
        assert saved["train_y"].shape == (13,)
        assert saved["val_x"].shape == (1, 32, 32)
        assert saved["val_y"].tolist() == [3]
        assert saved["test_x"].shape == (1, 32, 32)
        assert saved["test_y"].tolist() == [4]


def test_read_emnist_byclass_digits_filters_labels_and_keeps_image_pairs(tmp_path):
    import gzip
    import struct

    from nndl.augment import _read_emnist_byclass_digits

    images = np.zeros((4, 28, 28), dtype=np.uint8)
    images[0, 1, 2] = 10
    images[1, 3, 4] = 20
    images[2, 5, 6] = 30
    images[3, 7, 8] = 40
    labels = np.array([0, 12, 9, 35], dtype=np.uint8)

    images_path = tmp_path / "images.gz"
    labels_path = tmp_path / "labels.gz"
    with gzip.open(images_path, "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, len(images), 28, 28))
        handle.write(images.tobytes())
    with gzip.open(labels_path, "wb") as handle:
        handle.write(struct.pack(">II", 2049, len(labels)))
        handle.write(labels.tobytes())

    digit_images, digit_labels = _read_emnist_byclass_digits(images_path, labels_path)
    assert digit_labels.tolist() == [0, 9]
    assert digit_images.shape == (2, 28, 28)
    assert digit_images[0, 1, 2] == 10
    assert digit_images[1, 5, 6] == 30


def test_expand_dataset_uses_huggingface_source_without_torchvision(tmp_path, monkeypatch):
    import gzip
    import pickle

    import nndl.augment as augment_module

    source = tmp_path / "mnist.pkl.gz"
    train = (np.zeros((2, 784), dtype=np.float32), np.array([1, 2]))
    val = (np.zeros((1, 784), dtype=np.float32), np.array([3]))
    test = (np.zeros((1, 784), dtype=np.float32), np.array([4]))
    with gzip.open(source, "wb") as handle:
        pickle.dump((train, val, test), handle, protocol=4)

    extra_images = np.zeros((3, 28, 28), dtype=np.uint8)
    extra_images[1, 4, 5] = 255
    extra_labels = np.array([0, 5, 9], dtype=np.int64)
    monkeypatch.setattr(
        augment_module,
        "_load_emnist_digits_huggingface",
        lambda root: (extra_images, extra_labels),
    )

    output = tmp_path / "combined-hf.npz"
    augment_module.expand_dataset(
        source, output, image_size=32, include_emnist=True, emnist_source="huggingface"
    )

    with np.load(output) as saved:
        assert saved["train_x"].shape == (13, 1024)
        assert sorted(saved["train_y"].tolist()).count(5) == 1
        assert saved["val_y"].tolist() == [3]
        assert saved["test_y"].tolist() == [4]
