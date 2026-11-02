# Neural Networks and Deep Learning — PyTorch edition

A modern, chapter-oriented implementation of the code samples from Michael
Nielsen's book [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com).

The original examples targeted Python 2.6/2.7 and Theano. This edition uses
**Python 3.10+ and PyTorch 2.4+**, with CPU, NVIDIA CUDA, AMD ROCm, and Apple
MPS device support where provided by the installed PyTorch build.

The educational structure remains: compact models, explicit training loops,
and experiments that follow the book. The implementation adds a reusable
package, a command-line interface, modern data pipelines, checkpointing, and
tests while retaining the original chapter APIs where practical.

## Features

- MLP and CNN models with configurable architectures and activations.
- One training/evaluation implementation with Adam/AdamW/SGD, learning-rate
  scheduling, optional mixed precision, early stopping, and checkpoints.
- CPU/CUDA/ROCm/MPS device selection (`--device auto`).
- Original MNIST pickle loader and a 5× translated MNIST expansion pipeline.
- Optional EMNIST Digits integration from a resumable Hugging Face mirror or
  torchvision.
- Historical chapter compatibility modules (`network.py`, `network2.py`,
  `network3.py`) and classical baselines.
- Headless figure-generation scripts and a CPU-compatible test suite.

## Quick start

The repository includes the original MNIST pickle at `data/mnist.pkl.gz`.

```bash
python -m venv .venv
source .venv/bin/activate             # Linux/WSL
# .venv\Scripts\Activate.ps1         # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -e ".[plot,baselines,dev]"

# Modern high-accuracy MLP (32x32 padded MNIST canvas)
python -m nndl train mlp --epochs 50 --device auto

# Chapter 6 convolutional model (native 28x28 images)
python -m nndl train cnn --epochs 10 --device auto

# Reproduce the book's chapter 3 MLP setup
python -m nndl train mlp --book

# One-epoch smoke training run on a small subset
python -m nndl train mlp --quick --device cpu
```

For AMD GPUs, install a PyTorch wheel that matches the ROCm runtime in your
Linux/WSL environment **before** installing the project. ROCm PyTorch uses the
`cuda` device namespace. The correct wheel is platform- and runtime-specific;
the default PyPI installation may be CPU-only. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## CLI reference

```text
python -m nndl --help
python -m nndl train --help
python -m nndl augment --help
python -m nndl baseline --help
```

### Train

```bash
# Modern MLP
python -m nndl train mlp --epochs 50 --batch-size 128 --lr 1e-3

# CNN; defaults to native 28x28 input
python -m nndl train cnn --epochs 10 --device auto

# Explicit CNN canvas size
python -m nndl train cnn --image-size 32 --epochs 10

# Classic book setup: sigmoid + BCE + SGD + L2, 30 epochs
python -m nndl train mlp --book

# Experiment controls
python -m nndl train mlp --hidden 1024 768 512 --dropout 0.1 \
    --optimizer adamw --l2 1e-4 --early-stopping 12 --seed 42

# Disable automatic mixed precision or enable torch.compile
python -m nndl train cnn --no-amp
python -m nndl train mlp --compile
```

Default configurations:

| Setting | Modern MLP | CNN | `--book` MLP |
|---|---:|---:|---:|
| Input canvas | 32×32 | 28×28 | 28×28 |
| Hidden layers | 1024, 768, 512, 256, 128 | FC 100 | 30 |
| Activation | ReLU | ReLU | Sigmoid |
| Optimizer | AdamW | AdamW | SGD |
| Learning rate | 0.001 | 0.001 | 0.5 |
| Batch size | 128 | 64 | 10 |
| Epochs | 50 | 10 | 30 |
| Loss | Cross-entropy | Cross-entropy | Binary cross-entropy |

The 32×32 MLP input is the original MNIST image with a symmetric two-pixel
black border; it contains no additional image information. CNN and book-mode
MLP use native 28×28 input by default. Pass `--image-size 32` to a CNN only
when that canvas is intended; the selected size is also passed to the model.

Training runs are written to `runs/<model>-<timestamp>/` by default:

| Artifact | Contents |
|---|---|
| `best.pt` | Model and optimizer state selected by validation accuracy |
| `last.pt` | Model and optimizer state at the final epoch |
| `history.json` | Per-epoch losses, accuracies, and elapsed time |
| `args.json` | CLI arguments, effective config, environment, dataset and model metadata |
| `metrics.json` | Test metrics calculated using the best validation checkpoint |

The CLI restores `best.pt` before final test evaluation. The validation split
is used for checkpoint selection; the test split is not used for gradient
updates.

### MNIST and EMNIST expansion

The book's classic 5× expansion creates the original image plus four
one-pixel translations:

```bash
python -m nndl augment --output data/mnist_expanded_250k.npz --image-size 32
```

To add EMNIST Digits training images (the original MNIST validation and test
splits remain unchanged):

```bash
python -m nndl augment --include-emnist --emnist-source huggingface \
    --output data/mnist_emnist_490k.npz --image-size 32

python -m nndl train mlp --data data/mnist_emnist_490k.npz \
    --image-size 32 --epochs 100 --batch-size 2048 \
    --early-stopping 15 --device auto
```

The Hugging Face mirror downloads the two ByClass training IDX files and
selects digit labels 0–9. Downloads support parallel byte ranges and resuming
interrupted chunks where the server supports HTTP range requests. The
alternative `torchvision` source downloads the NIST archive and requires a
matching torchvision installation.

The 490k training set consists of 250k shifted MNIST examples plus 240k EMNIST
Digits examples. EMNIST images are transposed into upright orientation before
being merged. The generated dataset and EMNIST download cache are local data
artifacts and are not committed to Git by default.

### Classical baselines

```bash
python -m nndl baseline darkness
python -m nndl baseline svm --limit 10000 --kernel linear
python -m nndl baseline svm --limit 0 --kernel linear  # all 50k train examples
```

The average-darkness baseline scores **22.25% (2,225/10,000)** on the bundled
MNIST test split. SVM training requires the optional `baselines` dependencies.

### Chapter 6 experiment runner

```bash
python src/conv.py shallow --quick
python src/conv.py basic_conv --quick
python src/conv.py dbl_conv_relu --quick
python src/conv.py all --quick
```

The experiment runner writes checkpoints and metrics below `runs/conv/` and
figures to the working directory. Expanded-data experiments require the
generated NPZ dataset.

### Figures

Figure scripts use a non-interactive Matplotlib backend and save PNG files:

```bash
MPLBACKEND=Agg PYTHONPATH=src python fig/valley.py
MPLBACKEND=Agg PYTHONPATH=src python fig/mnist.py
```

Some historical scripts use committed JSON experiment results by default;
see each script's `--help` before requesting a recomputation.

## Project structure

```text
.
├── data/                 Original MNIST pickle and local generated datasets
├── docs/                 Architecture, experiments, and reproducibility notes
├── fig/                  Figure source scripts, JSON inputs, and PNGs
├── runs/                 Local model checkpoints and training metrics (ignored)
├── src/
│   ├── nndl/              Reusable PyTorch package and CLI
│   ├── network.py         Chapter 1 compatibility API
│   ├── network2.py        Chapter 3 compatibility API
│   ├── network3.py        Chapter 6 compatibility API
│   ├── conv.py            Chapter 6 experiment runner
│   └── mnist_*.py         Legacy loaders and classical baselines
├── tests/                 Automated test suite
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for module responsibilities,
data flow, compatibility notes, and checkpoint semantics.

## Tests and code quality

```bash
python -m pytest -q
python -m pytest -m "not slow"
python -m compileall -q src tests fig _smoketest.py
ruff check src tests fig _smoketest.py
ruff format --check src tests _smoketest.py
```

The default suite runs on CPU and covers data loading, MLP/CNN forward passes,
short training runs, checkpoint selection, CLI behavior, data augmentation,
baselines, and legacy API compatibility. GPU-specific tests should be marked
with `gpu` and run on a configured accelerator host.

### Local helper scripts (Linux/WSL)

The helper scripts resolve the repository root automatically and do not
activate a machine-specific virtual environment. To activate one explicitly:

```bash
VIRTUAL_ENV_ACTIVATE=/path/to/venv/bin/activate bash _smoke.sh
bash _envcheck.sh       # short CPU checks for losses, MLP, CNN, and benchmark
bash _gpu_verify.sh     # 10-epoch MLP/CNN runs; requires a visible CUDA/ROCm GPU
bash _figcheck.sh       # regenerate the listed figures (overwrites their PNGs)
```

Smoke and verification runs use unique directories under `runs/` by default.
Set `NNDL_RUN_DIR=/path/to/output` to choose an explicit output directory.

## Results

Current experiment notes are maintained in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). The strongest recorded MLP run on
the combined MNIST/EMNIST training data reached 99.23% validation accuracy.
The same best checkpoint was re-evaluated at 99.23% test accuracy in CPU FP32;
because the test set has already been used during development, this is a
development result rather than a fresh holdout estimate.

## License and attribution

The modernization is distributed under the MIT License. The original examples
are copyright © 2012–2022 Michael Nielsen. Third-party datasets have their own
terms; the project's code license does not change dataset licensing. See
[`LICENSE`](LICENSE) and the source dataset documentation before
redistribution.
