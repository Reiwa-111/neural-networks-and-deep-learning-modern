# Reproducibility and development

## Requirements

- Python 3.10 or newer
- PyTorch 2.4 or newer
- NumPy 1.26 or newer
- `tqdm`; optional `matplotlib`, `scikit-learn`, `pytest`, and `ruff`

PyTorch wheels depend on the operating system and accelerator. For AMD ROCm,
install a PyTorch build matching the ROCm runtime in the Linux/WSL environment
first, then install this project. PyTorch exposes ROCm devices through the
`torch.cuda` namespace, so the training code uses the same device path for CUDA
and ROCm.

## Installation

Create an environment and install the project with its development extras:

```bash
python -m venv .venv
source .venv/bin/activate             # Linux/WSL
# .venv\Scripts\Activate.ps1         # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -e ".[plot,baselines,dev]"
```

When using a platform-specific PyTorch build, install the matching wheel before
the editable project install. The package metadata (`pyproject.toml`) is the
canonical dependency definition; `requirements.txt` is a convenience list.

Verify the active backend:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda/rocm:', torch.cuda.is_available()); print('hip:', torch.version.hip)"
```

## Reproducible training

Use an explicit seed and, when required, deterministic algorithms:

```bash
python -m nndl train mlp --seed 42 --deterministic --device auto
```

Deterministic execution can be slower and is not guaranteed to be bit-identical
across different PyTorch versions, accelerator backends, or GPU models. Mixed
precision and backend-specific kernels can also change the last few reported
predictions.

Every new run records its effective configuration and environment in
`args.json`. Keep the complete run directory when comparing experiments; a
single accuracy value is not enough to reconstruct a training run.

## Tests and static checks

```bash
python -m pytest -q
python -m pytest -m "not slow"
python -m compileall -q src tests fig _smoketest.py
ruff check src tests fig _smoketest.py
ruff format --check src tests _smoketest.py
```

The default test suite is CPU-compatible. GPU-specific tests should be marked
with `gpu` and run separately on a configured accelerator host.

The repository's `_smoke.sh`, `_envcheck.sh`, `_gpu_verify.sh`, and
`_figcheck.sh` helpers derive the project root from their own location. They
use the currently active Python by default; set `VIRTUAL_ENV_ACTIVATE` to a
shell activation script when a specific environment must be selected. CPU/GPU
verification output is written to a unique `runs/` directory unless
`NNDL_RUN_DIR` is explicitly set. `_figcheck.sh` intentionally regenerates and
overwrites the selected figure PNGs.

## Data and evaluation discipline

- Do not use validation data for gradient updates.
- Select checkpoints and tune hyperparameters using validation metrics.
- Evaluate the test split only for the selected configuration.
- Repeatedly comparing many runs on the same test split turns it into a
  development signal; in that situation, report the limitation and reserve a
  fresh holdout for a final estimate.
- The source and license of third-party datasets are separate from the MIT
  license applied to the project code. Check the dataset's own terms before
  redistributing it.

## Known environment limitation

The project directory currently has no `.git` metadata. Until it is initialized
as a repository, experiment directories cannot be tied to a commit hash. Run
metadata therefore records the package version, Python/PyTorch versions,
platform, and accelerator details for future experiments.
