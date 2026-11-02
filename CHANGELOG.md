# Changelog

Notable changes to this project are recorded here.

## [2.0.1] — 2026-11-02

### Fixed

- CNN training now uses the selected input canvas size when building its
  classifier. The default is 28×28 for CNNs and 32×32 for the modern MLP.
- The final test evaluation now restores the checkpoint with the best
  validation accuracy instead of silently evaluating the last epoch.
- Integer convolution-kernel sizes now expand to every convolutional layer.
- Invalid training and model dimensions, dropout probabilities, and optimizer
  parameters fail early with actionable errors.
- NPZ loading and dataset expansion reject non-square images, mismatched labels,
  invalid image channels, and invalid loader/augmentation parameters.
- `--limit 0` for the SVM baseline consistently means "use the full training set".

### Improved

- Run metadata now records Python, PyTorch, platform, accelerator, dataset
  dimensions, model architecture, and parameter count.
- Checkpoint loading uses PyTorch's restricted `weights_only` deserializer.
- Added regression coverage for CNN input sizes, checkpoint selection, and
  configuration validation.
- Made the local shell checks portable and protected existing experiment runs
  from accidental overwrite by using timestamped output directories.
- Reworked the project documentation and added architecture, experiment, and
  reproducibility notes.

## [2.0.0]

- Replaced the original Theano/Python 2 implementation with a Python 3 and
  PyTorch implementation while retaining the book's educational flow and
  compatibility APIs where practical.
- Added MLP/CNN training, device selection, mixed precision, checkpointing,
  baselines, MNIST expansion, optional EMNIST Digits support, and tests.
