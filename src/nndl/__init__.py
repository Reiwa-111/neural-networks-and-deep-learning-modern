"""Modern PyTorch toolkit for the *Neural Networks and Deep Learning* samples.

The package keeps the educational spirit of Michael Nielsen's original code:
small, readable models and explicit training loops.  The implementation now
uses PyTorch, automatic differentiation, GPU acceleration and DataLoader-based
input pipelines.
"""

from __future__ import annotations

from .config import TrainConfig, resolve_device, set_seed

__all__ = ["TrainConfig", "__version__", "resolve_device", "set_seed"]
__version__ = "2.0.1"
