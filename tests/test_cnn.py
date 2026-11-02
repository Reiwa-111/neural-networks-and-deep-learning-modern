from __future__ import annotations

import torch

from nndl.config import TrainConfig
from nndl.data import load_mnist
from nndl.models import ConvNet
from nndl.trainer import Trainer, make_optimizer


def test_convnet_forward_shapes():
    model = ConvNet(channels=(20, 40), kernels=(5, 5), fc_sizes=(100,))
    assert model(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
    assert model(torch.zeros(2, 784)).shape == (2, 10)


def test_convnet_no_fc():
    model = ConvNet(channels=(20,), kernels=(5,), fc_sizes=())
    assert model(torch.zeros(2, 1, 28, 28)).shape == (2, 10)


def test_network3_compatibility_architecture():
    import network3

    net = network3.Network(
        [
            network3.ConvPoolLayer(
                image_shape=(10, 1, 28, 28),
                filter_shape=(20, 1, 5, 5),
                poolsize=(2, 2),
            ),
            network3.FullyConnectedLayer(n_in=20 * 12 * 12, n_out=100),
            network3.SoftmaxLayer(n_in=100, n_out=10),
        ],
        10,
        device="cpu",
    )
    x = torch.zeros(10, 1, 28, 28)
    assert net.model(x).shape == (10, 10)
    # log-probabilities
    assert torch.allclose(net.model(x).exp().sum(dim=1), torch.ones(10), atol=1e-5)


def test_cnn_one_batch(mnist_path):
    bundle = load_mnist(mnist_path, normalize=True).subset(64, 64)
    model = ConvNet(channels=(8,), kernels=(5,), fc_sizes=(32,))
    config = TrainConfig(
        epochs=1, batch_size=32, lr=1e-3, optimizer="adamw", device="cpu", amp=False
    )
    trainer = Trainer(model, torch.nn.CrossEntropyLoss(), make_optimizer(model, config), config)
    from nndl.data import make_dataloaders

    train, val, _ = make_dataloaders(bundle, batch_size=32)
    history = trainer.fit(train, val, verbose=False)
    assert len(history.train_loss) == 1
