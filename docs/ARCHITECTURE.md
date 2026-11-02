# Architecture

## Design goals

The project modernizes the code samples accompanying Michael Nielsen's
*Neural Networks and Deep Learning*. It keeps the chapter-oriented educational
examples and explicit training loops while moving tensor operations,
autodifferentiation, and accelerator support to PyTorch.

The code is split into a small reusable package (`src/nndl`) and top-level
compatibility modules (`src/network.py`, `network2.py`, and `network3.py`).
New development should target `nndl`; the compatibility modules exist for
older snippets from the book.

## Runtime structure

| Module | Responsibility |
|---|---|
| `nndl.config` | `TrainConfig`, device resolution, runtime seeding and device descriptions |
| `nndl.data` | Original MNIST pickle, expanded NPZ loading, tensor conversion and data loaders |
| `nndl.models` | Configurable MLP and convolutional models, activation resolution and parameter counting |
| `nndl.losses` | Book-compatible quadratic and binary cross-entropy losses, plus log-softmax NLL |
| `nndl.trainer` | Training/evaluation loops, AMP, scheduling, early stopping and checkpoints |
| `nndl.augment` | Translation/affine transforms, five-way MNIST expansion and EMNIST ingestion |
| `nndl.baselines` | Average-darkness and scikit-learn SVM baselines |
| `nndl.cli` | `train`, `augment`, `baseline`, and `benchmark` command-line entry points |
| `src/conv.py` | Chapter 6 experiment collection built on the shared model and trainer code |
| `fig/` | Reproducible figure scripts and source data for the book's visual explanations |

## Training flow

1. The CLI resolves a device (`auto`, CPU, CUDA/ROCm, or MPS) and seeds Python,
   NumPy, and PyTorch.
2. `load_mnist` converts the selected dataset into `MNISTBundle` tensors. The
   original validation and test splits are kept separate from training data.
3. `make_dataloaders` constructs shuffled training batches and deterministic
   validation/test loaders.
4. The CLI creates an `MLP` or `ConvNet`, criterion, optimizer, and
   `TrainConfig`, then passes them to `Trainer`.
5. `Trainer.fit` writes `last.pt` each epoch and updates `best.pt` whenever
   validation accuracy improves. Early stopping is controlled by validation
   metrics only.
6. The CLI restores `best.pt` before calculating the final test metrics. The
   full learning history remains intact in `history.json`.

## Input shapes and defaults

- Original MNIST images are 28×28, with one grayscale channel.
- The modern MLP defaults to a 32×32 canvas. `load_mnist` adds a symmetric
  two-pixel black border; this adds no new image information.
- CNN and book-compatible MLP runs default to native 28×28 inputs.
- CNN `input_size` must match the image side used to construct its classifier.
  The CLI passes this value explicitly.
- Normalization uses the training-set MNIST statistics defined in
  `nndl.data` (`mean=0.1307`, `std=0.3081`).

## Dataset composition

The bundled `data/mnist.pkl.gz` uses the classic split: 50,000 training,
10,000 validation, and 10,000 test examples.

The expanded experiment dataset `data/mnist_emnist_490k.npz` combines five
versions of each of the 50,000 MNIST training examples (the original plus four
one-pixel translations) with 240,000 EMNIST Digits training examples. Its
validation and test arrays remain the original MNIST validation and test
splits. EMNIST images are transposed into upright orientation before use.

The EMNIST cache and generated NPZ files are intentionally excluded from Git by
`.gitignore`; they can be regenerated with `nndl augment`.

## Checkpoint and run artifacts

Each training directory contains:

- `best.pt` — model/optimizer state at the epoch with the best validation
  accuracy.
- `last.pt` — state at the final completed epoch.
- `history.json` — per-epoch training and validation loss/accuracy and elapsed
  training time.
- `args.json` — command arguments, effective configuration, runtime/device
  information, dataset dimensions, and model architecture.
- `metrics.json` — final test metrics, selected checkpoint, and selected epoch.

These are experiment artifacts, not source files. Keep them locally under
`runs/` and export selected results separately when sharing a report.

## Compatibility layer

`network.py`, `network2.py`, and `network3.py` retain the original chapter APIs
where practical. They now delegate tensor computation and optimization to
PyTorch. `mnist_loader.py`, `mnist_average_darkness.py`, `mnist_svm.py`, and
`expand_mnist.py` preserve convenient entry points for the older scripts.

Compatibility means source-level convenience, not bit-for-bit parity with the
original Python 2/Theano implementation. Differences in numerical kernels,
initialization, and hardware execution are expected.
