from __future__ import annotations

import numpy as np
import pytest
import torch

from nndl.data import fit_image_size, load_expanded, load_mnist, make_dataloaders, to_one_hot


def test_load_mnist_shapes(mnist_path):
    bundle = load_mnist(mnist_path)
    assert bundle.train_x.shape == (50000, 1, 28, 28)
    assert bundle.val_x.shape == (10000, 1, 28, 28)
    assert bundle.test_x.shape == (10000, 1, 28, 28)
    assert bundle.train_y.shape == (50000,)
    assert bundle.train_x.dtype == torch.float32
    assert bundle.train_y.dtype == torch.int64
    assert int(bundle.train_y.min()) >= 0
    assert int(bundle.train_y.max()) <= 9


def test_load_mnist_flatten_and_normalize(mnist_path):
    bundle = load_mnist(mnist_path, flatten=True, normalize=True)
    assert bundle.train_x.shape == (50000, 784)
    assert abs(float(bundle.train_x.mean())) < 0.05
    assert abs(float(bundle.train_x.std()) - 1.0) < 0.05


def test_legacy_wrapper(mnist_path):
    import mnist_loader

    train, val, _test = mnist_loader.load_data_wrapper(mnist_path)
    assert len(train) == 50000
    assert train[0][0].shape == (784, 1)
    assert train[0][1].shape == (10, 1)
    assert val[0][0].shape == (784, 1)
    assert np.isscalar(val[0][1]) or np.asarray(val[0][1]).shape == ()


def test_dataloaders(mnist_path):
    bundle = load_mnist(mnist_path).subset(256, 128)
    train, val, test = make_dataloaders(bundle, batch_size=32, eval_batch_size=64)
    assert len(train) == 8
    assert len(val) == 2
    assert len(test) > 0
    x, y = next(iter(train))
    assert x.shape[1:] == (1, 28, 28)
    assert y.shape == (32,)


def test_one_hot():
    labels = torch.tensor([0, 3, 9])
    one_hot = to_one_hot(labels)
    assert one_hot.shape == (3, 10)
    assert torch.equal(one_hot.argmax(dim=1), labels)


def test_fit_image_size_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="target_size"):
        fit_image_size(np.zeros((1, 28, 28), dtype=np.float32), 0)
    with pytest.raises(ValueError, match="square"):
        fit_image_size(np.zeros((1, 28, 27), dtype=np.float32), 32)


def test_load_expanded_rejects_non_square_flat_images(tmp_path):
    path = tmp_path / "invalid.npz"
    np.savez(
        path,
        train_x=np.zeros((2, 1000), dtype=np.float32),
        train_y=np.zeros(2, dtype=np.int64),
        val_x=np.zeros((1, 784), dtype=np.float32),
        val_y=np.zeros(1, dtype=np.int64),
        test_x=np.zeros((1, 784), dtype=np.float32),
        test_y=np.zeros(1, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="perfect square"):
        load_expanded(path)


def test_load_expanded_rejects_mismatched_labels(tmp_path):
    path = tmp_path / "mismatched.npz"
    np.savez(
        path,
        train_x=np.zeros((2, 784), dtype=np.float32),
        train_y=np.zeros(1, dtype=np.int64),
        val_x=np.zeros((1, 784), dtype=np.float32),
        val_y=np.zeros(1, dtype=np.int64),
        test_x=np.zeros((1, 784), dtype=np.float32),
        test_y=np.zeros(1, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="counts do not match"):
        load_expanded(path)


def test_dataloaders_reject_zero_batch_size(mnist_path):
    bundle = load_mnist(mnist_path).subset(8, 4)
    with pytest.raises(ValueError, match="batch_size"):
        make_dataloaders(bundle, batch_size=0)


def test_expanded_loader_npz(tmp_path, mnist_path):
    bundle = load_mnist(mnist_path).subset(10, 5)
    path = tmp_path / "expanded.npz"
    np.savez_compressed(
        path,
        train_x=bundle.train_x.numpy().reshape(-1, 784),
        train_y=bundle.train_y.numpy(),
        val_x=bundle.val_x.numpy().reshape(-1, 28, 28),
        val_y=bundle.val_y.numpy(),
        test_x=bundle.test_x.numpy().reshape(-1, 28, 28),
        test_y=bundle.test_y.numpy(),
    )
    loaded = load_expanded(path)
    assert loaded.train_x.shape == (10, 1, 28, 28)
    assert loaded.val_x.shape == (5, 1, 28, 28)
