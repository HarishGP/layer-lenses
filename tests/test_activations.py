"""Tests for the ReLU subgradient-at-zero conventions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from layer_lenses.activations import (
    RELU,
    RELU_ONE_AT_ZERO,
    relu_one_at_zero,
    resolve_activation,
)
from layer_lenses.relu_analysis import activation_tensor_by_layer_neuron_data
from layer_lenses.relu_mlp import ReLUMLP


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_is_bit_identical_to_torch_relu(dtype):
    generator = torch.Generator().manual_seed(0)
    x = torch.cat(
        [
            torch.randn(10_000, dtype=dtype, generator=generator),
            torch.zeros(64, dtype=dtype),
            torch.full((64,), -0.0, dtype=dtype),
        ]
    )
    assert torch.equal(torch.relu(x), relu_one_at_zero(x))


def test_subgradient_at_zero_differs_only_at_zero():
    x = torch.tensor([-2.0, -1e-8, 0.0, -0.0, 1e-8, 3.0])

    standard = x.clone().requires_grad_(True)
    torch.relu(standard).sum().backward()

    one_at_zero = x.clone().requires_grad_(True)
    relu_one_at_zero(one_at_zero).sum().backward()

    assert standard.grad.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert one_at_zero.grad.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def test_resolve_activation_by_name_and_passthrough():
    assert resolve_activation("relu") is RELU
    assert resolve_activation("relu_one_at_zero") is RELU_ONE_AT_ZERO
    assert resolve_activation(RELU_ONE_AT_ZERO) is RELU_ONE_AT_ZERO
    with pytest.raises(ValueError, match="Unknown activation"):
        resolve_activation("gelu")


def test_gate_predicate_matches_backward_pass():
    pre_activation = torch.tensor([-1.0, 0.0, 1.0])
    assert RELU.gate(pre_activation).tolist() == [False, False, True]
    assert RELU_ONE_AT_ZERO.gate(pre_activation).tolist() == [False, True, True]


def test_zero_row_neuron_is_frozen_under_relu_and_freed_otherwise():
    """A bias-free neuron at ``w_k = 0`` can only move when ``phi'(0) = 1``."""
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((512, 6))).float()
    y_bin = torch.from_numpy(rng.integers(0, 2, size=512)).float()

    weights = torch.from_numpy(rng.standard_normal((4, 6))).float()
    weights[0] = 0.0
    output_weights = torch.from_numpy(rng.standard_normal((1, 4))).float()

    grad_norms = {}
    for name in ("relu", "relu_one_at_zero"):
        model = ReLUMLP(input_dim=6, hidden_dims=[4], bias=False, activation=name)
        with torch.no_grad():
            model.hidden_layers[0].weight.copy_(weights)
            model.output_layer.weight.copy_(output_weights)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y_bin)
        loss.backward()
        grad_norms[name] = model.hidden_layers[0].weight.grad.norm(dim=1)

    assert grad_norms["relu"][0] == 0.0
    assert grad_norms["relu_one_at_zero"][0] > 0.0
    # Only the degenerate neuron is affected; the others see identical gradients.
    torch.testing.assert_close(grad_norms["relu"][1:], grad_norms["relu_one_at_zero"][1:])


def test_analysis_blinders_follow_the_model_convention():
    """A pre-activation of exactly zero is open under ``relu_one_at_zero`` only."""
    x = np.zeros((8, 6), dtype=np.float32)
    x[:, 0] = 1.0
    weights = np.zeros((4, 6), dtype=np.float32)
    weights[0, 1] = 1.0  # orthogonal to every x, so neuron 0 sits exactly on its kink
    weights[1:, 0] = 1.0

    gates = {}
    for name in ("relu", "relu_one_at_zero"):
        model = ReLUMLP(input_dim=6, hidden_dims=[4], bias=False, activation=name)
        with torch.no_grad():
            model.hidden_layers[0].weight.copy_(torch.from_numpy(weights))
        gates[name] = activation_tensor_by_layer_neuron_data(model=model, x_eval=x)

    assert not gates["relu"][0, 0].any()
    assert gates["relu_one_at_zero"][0, 0].all()
    np.testing.assert_array_equal(gates["relu"][0, 1:], gates["relu_one_at_zero"][0, 1:])
