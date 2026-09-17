"""Standard fully-connected ReLU network for binary classification."""

from __future__ import annotations

import math

import torch
from torch import nn

from layer_lenses.activations import ActivationSpec, resolve_activation


class ReLUMLP(nn.Module):
    """Simple MLP that returns scalar logits of shape (batch,)."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: list[int],
        bias: bool = False,
        activation: str | ActivationSpec = "relu",
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty.")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = resolve_activation(activation)
        self.hidden_layers = nn.ModuleList()

        prev = input_dim
        for width in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev, width, bias=bias))
            prev = width
        self.output_layer = nn.Linear(prev, 1, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize linear layers with Gaussian variance matching PyTorch's default."""
        for layer in [*self.hidden_layers, self.output_layer]:
            fan_in = layer.weight.shape[1]
            std = 1.0 / math.sqrt(3.0 * fan_in)
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.hidden_layers:
            h = self.activation.apply(layer(h))
        return self.output_layer(h).squeeze(-1)
