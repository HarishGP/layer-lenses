"""ReLU activations differing only in the subgradient chosen at zero.

``relu`` is exactly ``x -> max(x, 0)`` and the Clarke subdifferential at the
kink is the whole interval ``[0, 1]``. PyTorch silently picks the lower
endpoint, ``phi'(0) = 0``. That choice is arbitrary but not harmless for the
bias-free single-path experiments: a neuron sitting at ``w_k = 0`` has
``w_k . x = 0`` for every ``x``, so its blinder is zero everywhere and it can
never leave the origin. Selecting the upper endpoint instead frees it, without
changing the forward pass by a single bit.

The two conventions differ only on ``{x : w_k . x = 0}``, a measure-zero set
under any continuous data distribution, so trajectories from a generic
initialization are identical. The choice matters exactly at the degenerate
points the paper's Sec. 3.3 construction deliberately starts from.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


class ReLUOneAtZero(torch.autograd.Function):
    """ReLU whose backward pass uses ``phi'(0) = 1`` rather than ``0``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x.clamp_min(0.0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad_output * (x >= 0).to(grad_output.dtype)


def relu_one_at_zero(x: torch.Tensor) -> torch.Tensor:
    """Forward-identical to ``torch.relu``; backward uses ``phi'(0) = 1``."""
    return ReLUOneAtZero.apply(x)


@dataclass(frozen=True)
class ActivationSpec:
    """One ReLU convention: its forward map and its matching blinder predicate.

    ``gate`` must agree with ``apply``'s backward pass, since analysis code
    reconstructs blinders from pre-activations rather than from autograd.
    """

    name: str
    subgradient_at_zero: float
    apply: Callable[[torch.Tensor], torch.Tensor]
    gate: Callable[[torch.Tensor], torch.Tensor]


RELU = ActivationSpec(
    name="relu",
    subgradient_at_zero=0.0,
    apply=torch.relu,
    gate=lambda pre_activation: pre_activation > 0,
)

RELU_ONE_AT_ZERO = ActivationSpec(
    name="relu_one_at_zero",
    subgradient_at_zero=1.0,
    apply=relu_one_at_zero,
    gate=lambda pre_activation: pre_activation >= 0,
)

ACTIVATIONS: dict[str, ActivationSpec] = {
    RELU.name: RELU,
    RELU_ONE_AT_ZERO.name: RELU_ONE_AT_ZERO,
}


def resolve_activation(activation: str | ActivationSpec) -> ActivationSpec:
    """Return the ``ActivationSpec`` for a name, passing specs through."""
    if isinstance(activation, ActivationSpec):
        return activation
    try:
        return ACTIVATIONS[activation]
    except KeyError:
        raise ValueError(
            f"Unknown activation {activation!r}. "
            f"Expected one of {sorted(ACTIVATIONS)}."
        ) from None
