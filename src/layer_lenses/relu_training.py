"""Training utilities for standard ReLU MLP models."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from layer_lenses.relu_mlp import ReLUMLP
from layer_lenses.training import set_seed

ParameterUpdateMask = dict[str, np.ndarray | torch.Tensor]


@dataclass(frozen=True)
class ReLUTrainConfig:
    """Training config for single-phase ReLU MLP experiments."""

    epochs: int = 200
    lr: float = 1e-3
    batch_size: int = 1024
    seed: int = 365
    device: str = "cpu"
    snapshot_epochs: tuple[int, ...] = (0,)
    weight_decay: float = 0.0
    lr_scheduler_eta_min_ratio: float = 0.2
    show_progress: bool = False
    # "adamw" (default) or "sgd" (mini-batch SGD; momentum is used only for SGD).
    optimizer: str = "adamw"
    momentum: float = 0.0
    first_layer_grad_orthogonal_to: tuple[float, ...] | np.ndarray | None = None
    parameter_update_mask: ParameterUpdateMask | None = None


def _labels_pm1_to_binary(y: torch.Tensor) -> torch.Tensor:
    if not torch.all((y == -1) | (y == 1)):
        raise ValueError("Expected labels in {-1, +1}.")
    return (y + 1.0) / 2.0


def evaluate_relu_mlp(
    *,
    model: nn.Module,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    device: str = "cpu",
) -> dict[str, int | float]:
    """Evaluate log loss and zero-one loss for a ReLU MLP classifier."""
    model = model.to(torch.device(device))
    model.eval()
    x = torch.from_numpy(x_eval).float().to(device)
    y_pm1 = torch.from_numpy(y_eval).float().to(device)
    y_bin = _labels_pm1_to_binary(y_pm1)

    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        logits = model(x)
        log_loss = float(criterion(logits, y_bin).detach().cpu())
        preds_pm1 = torch.where(logits >= 0, 1.0, -1.0)
        zero_one_loss = float((preds_pm1 != y_pm1).float().mean().detach().cpu())

    return {
        "log_loss": log_loss,
        "zero_one_loss": zero_one_loss,
        "accuracy": 1.0 - zero_one_loss,
        "num_samples": int(x_eval.shape[0]),
    }


def checkpoint_model_from_state(
    template_model: ReLUMLP,
    state_dict: dict[str, torch.Tensor],
) -> ReLUMLP:
    """Return a copy of ``template_model`` loaded with one checkpoint state."""
    model = copy.deepcopy(template_model)
    model.load_state_dict(state_dict)
    return model


def _first_layer_projection_vector(
    *,
    model: ReLUMLP,
    config: ReLUTrainConfig,
    device: torch.device,
) -> torch.Tensor | None:
    """Return normalized vector for first-layer gradient projection, if configured."""
    if config.first_layer_grad_orthogonal_to is None:
        return None

    u = torch.as_tensor(config.first_layer_grad_orthogonal_to, dtype=torch.float32, device=device)
    if u.ndim != 1:
        raise ValueError(
            "first_layer_grad_orthogonal_to must be a 1D vector, "
            f"got shape {tuple(u.shape)}."
        )
    input_dim = model.hidden_layers[0].weight.shape[1]
    if u.shape[0] != input_dim:
        raise ValueError(
            "first_layer_grad_orthogonal_to dimension must match model input_dim. "
            f"Expected {input_dim}, got {u.shape[0]}."
        )
    norm = u.norm()
    if float(norm.detach().cpu()) == 0.0:
        raise ValueError("first_layer_grad_orthogonal_to must be nonzero.")
    return u / norm.clamp_min(1e-12)


def _project_first_layer_weight_gradient(model: ReLUMLP, unit_vector: torch.Tensor | None) -> None:
    """Project first-layer weight gradients row-wise onto unit_vector's orthogonal complement."""
    if unit_vector is None:
        return

    grad = model.hidden_layers[0].weight.grad
    if grad is None:
        return

    grad.sub_((grad @ unit_vector).unsqueeze(1) * unit_vector.unsqueeze(0))


def _parameter_update_masks(
    *,
    model: ReLUMLP,
    config: ReLUTrainConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Return validated 0/1 masks keyed by model parameter name."""
    if config.parameter_update_mask is None:
        return {}

    named_params = dict(model.named_parameters())
    mask_names = set(config.parameter_update_mask.keys())
    param_names = set(named_params.keys())
    if mask_names != param_names:
        missing = sorted(param_names.difference(mask_names))
        unknown = sorted(mask_names.difference(param_names))
        details = []
        if missing:
            details.append(f"missing masks for {missing}")
        if unknown:
            details.append(f"unknown parameters {unknown}")
        raise ValueError("parameter_update_mask must match model parameter names: " + "; ".join(details))

    masks: dict[str, torch.Tensor] = {}
    for name, param in named_params.items():
        mask = torch.as_tensor(
            config.parameter_update_mask[name],
            dtype=param.dtype,
            device=device,
        )
        if mask.shape != param.shape:
            raise ValueError(
                f"parameter_update_mask[{name!r}] has shape {tuple(mask.shape)}, "
                f"expected {tuple(param.shape)}."
            )
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"parameter_update_mask[{name!r}] must contain only 0 and 1.")
        masks[name] = mask
    return masks


def _apply_parameter_update_mask_gradients(
    model: ReLUMLP,
    parameter_update_masks: dict[str, torch.Tensor],
) -> None:
    if not parameter_update_masks:
        return

    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad.mul_(parameter_update_masks[name])


def _snapshot_frozen_parameter_values(
    model: ReLUMLP,
    parameter_update_masks: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not parameter_update_masks:
        return {}

    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if not bool(torch.all(parameter_update_masks[name]).detach().cpu())
    }


def _restore_frozen_parameter_values(
    model: ReLUMLP,
    parameter_update_masks: dict[str, torch.Tensor],
    frozen_parameter_values: dict[str, torch.Tensor],
) -> None:
    if not parameter_update_masks:
        return

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in frozen_parameter_values:
                continue
            mask = parameter_update_masks[name]
            param.copy_(param * mask + frozen_parameter_values[name] * (1 - mask))


def train_relu_mlp(
    *,
    model: ReLUMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: ReLUTrainConfig,
) -> dict[str, object]:
    """Train ReLU MLP with BCE-with-logits, cosine LR annealing, and optional masks."""
    set_seed(config.seed)
    device = torch.device(config.device)
    model = model.to(device)
    first_layer_projection_vector = _first_layer_projection_vector(
        model=model,
        config=config,
        device=device,
    )
    parameter_update_masks = _parameter_update_masks(
        model=model,
        config=config,
        device=device,
    )

    x = torch.from_numpy(x_train).float().to(device)
    y_pm1 = torch.from_numpy(y_train).float().to(device)
    y_bin = _labels_pm1_to_binary(y_pm1)

    criterion = nn.BCEWithLogitsLoss()
    opt_name = config.optimizer.strip().lower()
    if opt_name in {"adamw", "adam"}:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    elif opt_name in {"sgd", "gd"}:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=False,
        )
    else:
        raise ValueError(
            'optimizer must be "adamw" or "sgd", '
            f"got {config.optimizer!r}"
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=max(1, config.epochs),
        eta_min=config.lr * config.lr_scheduler_eta_min_ratio,
    )

    num_samples = x.shape[0]
    losses: list[float] = []
    lr_history: list[float] = []
    checkpoint_snapshots: dict[int, dict[str, torch.Tensor]] = {}

    epoch_iter = range(config.epochs + 1)
    if config.show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Training ReLU MLP", leave=False)

    for epoch in epoch_iter:
        if epoch in config.snapshot_epochs:
            checkpoint_snapshots[epoch] = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        permutation = torch.randperm(num_samples, device=device)
        if epoch == config.epochs:
            break

        running_loss = 0.0
        seen = 0
        for start in range(0, num_samples, config.batch_size):
            end = min(start + config.batch_size, num_samples)
            idx = permutation[start:end]

            optimizer.zero_grad()
            logits = model(x[idx])
            loss = criterion(logits, y_bin[idx])
            loss.backward()
            _project_first_layer_weight_gradient(model, first_layer_projection_vector)
            _apply_parameter_update_mask_gradients(model, parameter_update_masks)
            frozen_parameter_values = _snapshot_frozen_parameter_values(
                model,
                parameter_update_masks,
            )
            optimizer.step()
            _restore_frozen_parameter_values(
                model,
                parameter_update_masks,
                frozen_parameter_values,
            )

            batch_n = end - start
            running_loss += float(loss.detach().cpu()) * batch_n
            seen += batch_n

        losses.append(running_loss / max(1, seen))
        lr_history.append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step()
        if config.show_progress:
            epoch_iter.set_postfix(loss=f"{losses[-1]:.4f}")

    return {
        "model": model,
        "epoch_losses": losses,
        "lr_history": lr_history,
        "checkpoint_snapshots": checkpoint_snapshots,
    }
