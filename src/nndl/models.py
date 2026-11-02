"""Neural network models used by the modernized samples."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn


def _activation_name(value: object) -> str:
    if isinstance(value, str):
        return value.lower()
    name = getattr(value, "__name__", "")
    return str(name).lower()


def resolve_activation(activation: object | None) -> nn.Module:
    """Turn strings, callables or ``nn.Module`` classes into a module.

    Legacy code passes ``sigmoid``, ``tanh`` and ``ReLU`` functions imported
    from ``network3``; the name-based dispatch keeps that API working.
    """

    if activation is None:
        return nn.Identity()
    if isinstance(activation, nn.Module):
        return activation
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation()

    name = _activation_name(activation)
    mapping: dict[str, type[nn.Module]] = {
        "sigmoid": nn.Sigmoid,
        "logistic": nn.Sigmoid,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
        "leakyrelu": nn.LeakyReLU,
        "linear": nn.Identity,
        "identity": nn.Identity,
        "none": nn.Identity,
    }
    if name in mapping:
        return mapping[name]()
    raise ValueError(f"Unsupported activation: {activation!r}")


class MLP(nn.Module):
    """Fully connected network with configurable activations.

    ``sizes=[784, 30, 10]`` reproduces the architecture from chapters 1 and 3.
    The output is left as logits when ``output_activation=None`` so that
    ``BCEWithLogitsLoss``/``CrossEntropyLoss`` can be used safely.
    """

    def __init__(
        self,
        sizes: Sequence[int],
        *,
        hidden_activation: object = "sigmoid",
        output_activation: object | None = None,
        dropout: float = 0.0,
        batchnorm: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if len(sizes) < 2:
            raise ValueError("sizes must contain at least input and output layers")
        self.sizes = tuple(int(s) for s in sizes)
        if any(size < 1 for size in self.sizes):
            raise ValueError("all layer sizes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        self.linears = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batchnorms = nn.ModuleList()

        for index in range(len(self.sizes) - 1):
            in_features, out_features = self.sizes[index], self.sizes[index + 1]
            self.linears.append(nn.Linear(in_features, out_features, bias=bias))
            is_output = index == len(self.sizes) - 2
            if is_output:
                self.activations.append(resolve_activation(output_activation))
                self.dropouts.append(nn.Identity())
                self.batchnorms.append(nn.Identity())
            else:
                if batchnorm:
                    self.batchnorms.append(nn.BatchNorm1d(out_features))
                else:
                    self.batchnorms.append(nn.Identity())
                self.activations.append(resolve_activation(hidden_activation))
                self.dropouts.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())

        self.reset_parameters("default")

    def reset_parameters(self, kind: str = "default") -> None:
        """Initialize weights for stable training of deeper MLPs.

        ``default`` uses Kaiming initialization for ReLU-family activations,
        Xavier for sigmoid/tanh, and zero biases. ``large`` and ``xavier``
        remain available for the legacy examples.
        """
        activation = self.activations[0] if self.activations else nn.Identity()
        relu_family = isinstance(activation, (nn.ReLU, nn.ReLU6, nn.LeakyReLU))
        for linear in self.linears:
            if kind == "large":
                nn.init.normal_(linear.weight, mean=0.0, std=1.0)
            elif kind == "xavier" or not relu_family:
                nn.init.xavier_normal_(linear.weight)
            else:
                nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    @property
    def weights(self) -> list[torch.Tensor]:
        return [linear.weight for linear in self.linears]

    @property
    def biases(self) -> list[torch.Tensor]:
        return [linear.bias for linear in self.linears if linear.bias is not None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        for linear, norm, activation, dropout in zip(
            self.linears, self.batchnorms, self.activations, self.dropouts
        ):
            x = linear(x)
            x = norm(x)
            x = activation(x)
            x = dropout(x)
        return x


class ConvNet(nn.Module):
    """Convolutional network mirroring the book's chapter 6 architecture.

    The default ``channels=(20, 40)``, ``kernels=(5, 5)`` and
    ``fc_sizes=(100,)`` is the ``Conv + Conv + FC`` model from the book.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        channels: Sequence[int] = (20, 40),
        kernels: int | Sequence[int] = (5, 5),
        pool: int | Sequence[int] = 2,
        fc_sizes: Sequence[int] = (100,),
        num_classes: int = 10,
        activation: object = "relu",
        dropout: float | Sequence[float] = 0.0,
        output_dropout: float = 0.0,
        batchnorm: bool = False,
        input_size: int = 28,
    ) -> None:
        super().__init__()
        if len(channels) < 1:
            raise ValueError("channels must contain at least one value")
        channels = tuple(int(value) for value in channels)
        kernels = (kernels,) * len(channels) if isinstance(kernels, int) else tuple(kernels)
        pool = (pool, pool) if isinstance(pool, int) else tuple(pool)
        if len(kernels) != len(channels):
            raise ValueError("kernels must have one entry per convolution")
        if any(
            value < 1
            for value in (in_channels, num_classes, input_size, *channels, *kernels, *pool)
        ):
            raise ValueError("channel counts, kernels, pooling and input size must be positive")
        if any(value < 1 for value in fc_sizes):
            raise ValueError("fully connected layer sizes must be positive")
        if not 0.0 <= output_dropout < 1.0:
            raise ValueError("output_dropout must be in the range [0, 1)")

        self.input_size = input_size
        self.in_channels = in_channels
        self.channels = tuple(channels)
        self.kernels = kernels
        self.pool = pool
        self.fc_sizes = tuple(fc_sizes)
        self.num_classes = num_classes
        self.dropout_p = dropout
        self.output_dropout_p = output_dropout

        blocks: list[nn.Module] = []
        previous = in_channels
        for out_channels, kernel in zip(channels, kernels):
            blocks.append(nn.Conv2d(previous, out_channels, kernel))
            if batchnorm:
                blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(resolve_activation(activation))
            blocks.append(nn.MaxPool2d(pool))
            previous = out_channels
        self.features = nn.Sequential(*blocks)
        self.conv_layers = [m for m in self.features if isinstance(m, nn.Conv2d)]

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            flat_dim = self.features(dummy).numel()

        if isinstance(dropout, (tuple, list)):
            dropouts = tuple(float(value) for value in dropout)
            if len(dropouts) != len(self.fc_sizes):
                raise ValueError("dropout sequence must match the number of fc layers")
        else:
            dropouts = tuple(float(dropout) for _ in self.fc_sizes)
        if any(not 0.0 <= value < 1.0 for value in dropouts):
            raise ValueError("dropout values must be in the range [0, 1)")

        classifier: list[nn.Module] = [nn.Flatten()]
        current = flat_dim
        for hidden, drop in zip(self.fc_sizes, dropouts):
            classifier.append(nn.Linear(current, hidden))
            if batchnorm:
                classifier.append(nn.BatchNorm1d(hidden))
            classifier.append(resolve_activation(activation))
            if drop > 0:
                classifier.append(nn.Dropout(drop))
            current = hidden
        if output_dropout > 0:
            classifier.append(nn.Dropout(output_dropout))
        classifier.append(nn.Linear(current, num_classes))
        self.classifier = nn.Sequential(*classifier)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.reshape(-1, self.in_channels, self.input_size, self.input_size)
        x = self.features(x)
        return self.classifier(x)


class ConvPoolStack(nn.Module):
    """Small helper used by the ``network3`` compatibility layer."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int | Sequence[int] = 2,
        activation: object = "sigmoid",
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size),
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend([resolve_activation(activation), nn.MaxPool2d(pool_size)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parameter_count(model: nn.Module, trainable_only: bool = True) -> int:
    params: Iterable[torch.Tensor] = model.parameters()
    return sum(p.numel() for p in params if p.requires_grad or not trainable_only)
