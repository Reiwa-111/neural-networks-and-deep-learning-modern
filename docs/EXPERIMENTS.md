# Experiment log

This file records results that can be traced to artifacts currently present in
`runs/`. Values are observations from this working copy, not performance
guarantees for other systems.

## High-accuracy MLP on MNIST + EMNIST Digits

Run directory: `runs/mlp-20261102-161928/`.

### Dataset

| Split | Examples | Source |
|---|---:|---|
| Training | 490,000 | 250,000 shifted MNIST + 240,000 EMNIST Digits |
| Validation | 10,000 | Original MNIST validation split |
| Test | 10,000 | Original MNIST test split |

Inputs use a 32×32 canvas. MNIST images are padded with a two-pixel black
border; EMNIST images are reoriented and padded to the same size.

### Model and training configuration

| Setting | Value |
|---|---|
| Architecture | 1024 → 1024 → 768 → 512 → 256 → 128 → 10 |
| Hidden activation | ReLU |
| Dropout | 0.1 |
| Loss | Cross-entropy |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 2,048 |
| Seed | 42 |
| Planned epochs | 100 |
| Early stopping | 15 epochs without validation improvement |
| Completed epochs | 35 |
| AMP | Enabled |
| Device recorded in args | `cuda` namespace |

### Recorded and audited metrics

| Metric | Result | Provenance |
|---|---:|---|
| Best validation accuracy | **99.23%** | `history.json`, reached at epoch 20 |
| Test accuracy in original `metrics.json` | **99.08%** | Original CLI evaluation after the final epoch |
| Test accuracy, `best.pt` re-evaluated in CPU FP32 | **99.23% (9,923/10,000)** | Post-run audit of the saved best checkpoint |
| Test accuracy, `last.pt` re-evaluated in CPU FP32 | **99.09% (9,909/10,000)** | Post-run audit of the saved final checkpoint |

The CPU audit used PyTorch 2.4.1+cpu and FP32 inference. The original training
run recorded `device="cuda"` and AMP enabled; the run metadata does not contain
the exact GPU model or ROCm runtime version. Small differences between the
original and CPU-re-evaluated metrics are therefore expected.

The difference between `metrics.json` and the best-checkpoint audit also
exposed a selection issue: the old CLI evaluated the final in-memory weights
after training, although it had saved a better validation checkpoint. Version
2.0.1 fixes this for new runs by restoring `best.pt` before test evaluation.
The historical `metrics.json` file is intentionally left unchanged.

**Evaluation caveat:** this test split has already been evaluated across
multiple experiments in this working copy. Treat these figures as development
results, not as a fresh, untouched holdout estimate. Use validation for further
tuning and reserve new independent data for a final generalization claim.

## Historical CNN verification

Run directory: `runs/verify-cnn/`.

- Architecture recorded in the project documentation: two convolutional
  stages (20 and 40 channels, 5×5 kernels) followed by a 100-unit fully
  connected layer.
- Optimizer: AdamW; learning rate: 0.001; batch size: 64; 10 epochs.
- Stored test accuracy: **99.21%**; stored best validation accuracy: **99.11%**.

This is a historical run produced before the current CLI input-size handling
was corrected. It is retained as an artifact reference, not presented as a
fresh post-fix benchmark.

## Baseline

The average-darkness baseline is tested against the bundled 10,000-example
MNIST test split and returns **22.25% (2,225/10,000)**. This result is asserted
by `tests/test_baselines.py`.
