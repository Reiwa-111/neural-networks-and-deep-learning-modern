from __future__ import annotations

import json

from nndl.cli import main


def test_cli_darkness(tmp_path, mnist_path):
    output = tmp_path / "darkness.json"
    code = main(["baseline", "darkness", "--data", str(mnist_path), "--json", str(output)])
    assert code == 0
    assert output.exists()


def test_cli_quick_mlp(tmp_path, mnist_path):
    output = tmp_path / "run"
    code = main(
        [
            "train",
            "mlp",
            "--data",
            str(mnist_path),
            "--quick",
            "--device",
            "cpu",
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    assert (output / "metrics.json").exists()
    assert (output / "history.json").exists()
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["selected_checkpoint"] == "best.pt"
    assert metrics["selected_epoch"] == 1


def test_cli_quick_cnn_uses_native_28px_input_by_default(tmp_path, mnist_path):
    output = tmp_path / "cnn-run"
    code = main(
        [
            "train",
            "cnn",
            "--data",
            str(mnist_path),
            "--quick",
            "--device",
            "cpu",
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    args = json.loads((output / "args.json").read_text(encoding="utf-8"))
    assert args["args"]["image_size"] == 28
    assert args["dataset"]["input_shape"] == [1, 28, 28]
    assert args["environment"]["torch_version"]
    assert args["model"]["parameters"] > 0
    assert (output / "metrics.json").exists()


def test_cli_quick_cnn_accepts_explicit_32px_input(tmp_path, mnist_path):
    output = tmp_path / "cnn-32-run"
    code = main(
        [
            "train",
            "cnn",
            "--data",
            str(mnist_path),
            "--quick",
            "--image-size",
            "32",
            "--device",
            "cpu",
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    args = json.loads((output / "args.json").read_text(encoding="utf-8"))
    assert args["args"]["image_size"] == 32
    assert (output / "metrics.json").exists()
