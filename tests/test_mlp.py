from __future__ import annotations

import json

import torch

from nndl.config import TrainConfig
from nndl.data import load_mnist, make_dataloaders
from nndl.losses import BinaryCrossEntropyLoss
from nndl.models import MLP
from nndl.trainer import Trainer, make_optimizer


def test_mlp_forward_shapes():
    model = MLP([784, 30, 10], hidden_activation="relu")
    assert model(torch.zeros(4, 784)).shape == (4, 10)
    assert model(torch.zeros(4, 1, 28, 28)).shape == (4, 10)


def test_short_training(mnist_path):
    bundle = load_mnist(mnist_path, flatten=True).subset(1000, 500)
    train, val, _ = make_dataloaders(bundle, batch_size=50, seed=0)
    model = MLP([784, 30, 10], hidden_activation="relu")
    config = TrainConfig(
        epochs=1, batch_size=50, lr=1e-3, optimizer="adamw", device="cpu", amp=False
    )
    trainer = Trainer(model, torch.nn.CrossEntropyLoss(), make_optimizer(model, config), config)
    history = trainer.fit(train, val, verbose=False)
    assert len(history.train_loss) == 1
    assert 0.0 <= history.train_accuracy[-1] <= 1.0
    assert 0.0 <= history.val_accuracy[0] <= 1.0


def test_network2_json_roundtrip(tmp_path):
    import network2

    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    path = tmp_path / "net.json"
    net.save(path)
    loaded = network2.load(path)
    assert loaded.sizes == [784, 30, 10]
    assert loaded.cost is network2.CrossEntropyCost
    assert [w.shape for w in loaded.weights] == [(30, 784), (10, 30)]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["cost"] == "CrossEntropyCost"


def test_network2_backprop_shapes():
    import network2

    net = network2.Network([4, 3, 2], cost=network2.CrossEntropyCost)
    nabla_b, nabla_w = net.backprop(
        torch.zeros(4, 1).numpy(),
        torch.tensor([[0.0], [1.0]]).numpy(),
    )
    assert [b.shape for b in nabla_b] == [(3, 1), (2, 1)]
    assert [w.shape for w in nabla_w] == [(3, 4), (2, 3)]


def test_binary_cross_entropy_loss():
    logits = torch.zeros(2, 10)
    target = torch.zeros(2, 10)
    target[:, 0] = 1.0
    loss = BinaryCrossEntropyLoss()(logits, target)
    assert loss.item() > 0


def test_restore_best_weights_preserves_full_training_history(tmp_path, mnist_path):
    bundle = load_mnist(mnist_path, flatten=True).subset(256, 128)
    train, val, _ = make_dataloaders(bundle, batch_size=64, seed=0)
    model = MLP([784, 32, 10], hidden_activation="relu")
    config = TrainConfig(
        epochs=2,
        batch_size=64,
        lr=1e-3,
        optimizer="adamw",
        device="cpu",
        amp=False,
    )
    trainer = Trainer(
        model,
        torch.nn.CrossEntropyLoss(),
        make_optimizer(model, config),
        config,
        output_dir=tmp_path,
    )
    trainer.fit(train, val, verbose=False)

    checkpoint = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=True)
    expected_history = trainer.history.to_dict()
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(1.0)

    selected_epoch = trainer.restore_best_weights()

    assert selected_epoch == checkpoint["epoch"]
    assert trainer.history.to_dict() == expected_history
    for name, parameter in trainer.unwrap_model().state_dict().items():
        assert torch.equal(parameter, checkpoint["model_state"][name])
