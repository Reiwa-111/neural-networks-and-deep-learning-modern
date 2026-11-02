from __future__ import annotations

import pytest
import torch

from nndl.config import TrainConfig
from nndl.models import MLP, ConvNet


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lr": 0.0}, "lr"),
        ({"weight_decay": -1.0}, "weight_decay"),
        ({"momentum": -0.1}, "momentum"),
        ({"num_workers": -1}, "num_workers"),
        ({"early_stopping_patience": 0}, "early_stopping_patience"),
        ({"grad_clip": 0.0}, "grad_clip"),
    ],
)
def test_train_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TrainConfig(**kwargs)


def test_mlp_rejects_invalid_layer_size_and_dropout():
    with pytest.raises(ValueError, match="layer sizes"):
        MLP([784, 0, 10])
    with pytest.raises(ValueError, match="dropout"):
        MLP([784, 10], dropout=1.0)


def test_convnet_supports_integer_kernel_for_every_convolution():
    model = ConvNet(channels=(8,), kernels=5, input_size=32)
    assert model(torch.zeros(2, 1, 32, 32)).shape == (2, 10)
