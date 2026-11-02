# Contributing

## Development setup

Use Python 3.10+ and install the development dependencies:

```bash
python -m pip install -e ".[plot,baselines,dev]"
```

For GPU work, install the PyTorch wheel matching the host accelerator/runtime
before installing the editable project.

## Before submitting changes

Run the checks from the repository root:

```bash
python -m pytest -q
python -m compileall -q src tests fig _smoketest.py
ruff check src tests fig _smoketest.py
ruff format --check src tests _smoketest.py
```

New behavior should have a focused regression test. Keep the CPU test suite
independent of a GPU, network access, and pre-existing `runs/` artifacts.

## Implementation conventions

- Put reusable model, data, training, and CLI logic in `src/nndl/`.
- Keep the top-level chapter modules as compatibility shims; avoid introducing
  new functionality there unless it directly supports a legacy API.
- Keep device-specific behavior behind PyTorch device abstractions.
- Do not use the test split for optimization, early stopping, or checkpoint
  selection. Use validation metrics for those decisions.
- Store generated checkpoints and expanded datasets under ignored local
  artifact directories. Do not commit model weights or generated datasets
  unless the change explicitly requires a small fixture.
- Record meaningful user-facing changes in `CHANGELOG.md` and update the
  relevant documentation page.

## Compatibility

Avoid breaking signatures in `network.py`, `network2.py`, `network3.py`, and
`mnist_loader.py`. When a compatibility break is unavoidable, document the
reason and provide a migration note.
