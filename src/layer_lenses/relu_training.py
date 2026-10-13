"""Training utilities for standard ReLU MLP models."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import device, nn
from tqdm.auto import tqdm

from layer_lenses.relu_mlp import ReLUMLP
from layer_lenses.training import set_seed

ParameterUpdateMask = dict[str, np.ndarray | torch.Tensor]


@dataclass(frozen=True)
class FirstLayerRotationLock:
    """Per-row rotation lock for the first hidden layer (see `relu_training_step`).

    A row in ``rows`` (every row if None) trains freely until ``|cos(row, t)| >= threshold``
    for some row ``t`` of ``targets`` (for instance the ODT normals). From that moment on
    its direction is fixed for good: after every optimizer step the row is projected back
    onto the unit direction it had when it locked, so only its norm (radial coordinate)
    can still change. Rows already past the threshold at the start lock immediately.
    """

    targets: np.ndarray | torch.Tensor
    threshold: float = 0.99
    rows: tuple[int, ...] | None = None


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
    # "adamw" (default), "sgd" (mini-batch SGD), or "sgd_adam" (hybrid: the first
    # `sgd_adam_split_layer` hidden layers use SGD, everything after them uses AdamW).
    # `momentum` is used only by the SGD parts.
    optimizer: str = "adamw"
    momentum: float = 0.0
    # Hybrid "sgd_adam" only: hidden layers [0, sgd_adam_split_layer) are trained by
    # SGD; the remaining hidden layers and the output layer are trained by AdamW.
    sgd_adam_split_layer: int = 1
    # Hybrid "sgd_adam" only: LR / weight decay for the SGD part. None reuses `lr` /
    # `weight_decay`. See `build_relu_optimizers` for why `sgd_lr` almost always needs
    # to be much larger than `lr`.
    sgd_lr: float | None = None
    sgd_weight_decay: float | None = None
    first_layer_grad_orthogonal_to: tuple[float, ...] | np.ndarray | None = None
    parameter_update_mask: ParameterUpdateMask | None = None
    first_layer_rotation_lock: FirstLayerRotationLock | None = None
    # "scalar_adam" only: (beta1, beta2) of the momentum and of the scalar second-moment estimate.
    adam_betas: tuple[float, float] = (0.9, 0.999)


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
        # A logit of exactly zero (every ReLU off) counts as +1 above; the strict rule
        # counts it as -1. The gap between the two is the mass of dead inputs.
        preds_strict = torch.where(logits > 0, 1.0, -1.0)
        zero_one_loss_strict = float((preds_strict != y_pm1).float().mean().detach().cpu())
        zero_logit_fraction = float((logits == 0).float().mean().detach().cpu())

    return {
        "log_loss": log_loss,
        "zero_one_loss": zero_one_loss,
        "zero_one_loss_strict": zero_one_loss_strict,
        "zero_logit_fraction": zero_logit_fraction,
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
    """Return validated masks keyed by model parameter name.

    0 freezes an entry, 1 trains it normally, and any other positive value multiplies its gradient
    (under SGD / momentum that is a per-entry learning-rate factor; under Adam-type optimizers it
    is not, since the normalisation absorbs a constant scale).
    """
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
        if not torch.all(torch.isfinite(mask) & (mask >= 0)):
            raise ValueError(f"parameter_update_mask[{name!r}] must contain finite non-negative values (0 freezes, 1 trains, other values scale the gradient).")
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
        if not bool(torch.all(parameter_update_masks[name] > 0).detach().cpu())
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
            free = (parameter_update_masks[name] > 0).to(param.dtype)
            param.copy_(param * free + frozen_parameter_values[name] * (1 - free))


@dataclass
class RotationLockState:
    """Runtime state of a `FirstLayerRotationLock` (one entry per first-layer row).

    ``locked`` says which rows are locked, ``direction`` holds each locked row's unit
    direction at lock time (zeros for free rows), ``lock_step`` the global step count at
    which the row locked (-1 if free; ``lock_step == s`` means the update of step ``s`` was
    the first one projected) and ``step`` the number of steps taken so far.
    """

    targets: torch.Tensor
    threshold: float
    eligible: torch.Tensor
    locked: torch.Tensor
    direction: torch.Tensor
    lock_step: torch.Tensor
    step: int

    def snapshot(self) -> dict[str, np.ndarray]:
        """Return the per-row state as numpy arrays (enough to resume or replay exactly)."""
        return {"locked": self.locked.cpu().numpy().copy(), "direction": self.direction.cpu().numpy().copy(),
                "lock_step": self.lock_step.cpu().numpy().copy(), "step": np.int64(self.step)}


def rotation_lock_state(
    *,
    model: ReLUMLP,
    config: ReLUTrainConfig,
    device: torch.device,
    snapshot: dict[str, np.ndarray] | None = None,
    step: int = 0,
) -> RotationLockState | None:
    """Build the lock state for ``config.first_layer_rotation_lock`` (None when unset).

    Without ``snapshot`` the state is fresh and rows already past the threshold lock at
    once (with ``lock_step = step``); with a snapshot (from `RotationLockState.snapshot`,
    e.g. a stored run's ``rotation_lock_snapshots[e]``) the stored state is restored.
    """
    lock = config.first_layer_rotation_lock
    if lock is None:
        return None
    weight = model.hidden_layers[0].weight.detach()
    n_rows, input_dim = weight.shape
    targets = torch.as_tensor(np.asarray(lock.targets), dtype=torch.float32, device=device)
    if targets.ndim != 2 or targets.shape[1] != input_dim:
        raise ValueError(f"rotation lock targets must have shape (K, {input_dim}), got {tuple(targets.shape)}")
    if not 0.0 < lock.threshold <= 1.0:
        raise ValueError(f"rotation lock threshold must be in (0, 1], got {lock.threshold}")
    targets = targets / targets.norm(dim=1, keepdim=True).clamp_min(1e-12)
    eligible = torch.zeros(n_rows, dtype=torch.bool, device=device)
    if lock.rows is None:
        eligible[:] = True
    else:
        eligible[list(lock.rows)] = True
    if snapshot is None:
        state = RotationLockState(targets=targets, threshold=float(lock.threshold), eligible=eligible,
                                  locked=torch.zeros(n_rows, dtype=torch.bool, device=device),
                                  direction=torch.zeros(n_rows, input_dim, dtype=weight.dtype, device=device),
                                  lock_step=torch.full((n_rows,), -1, dtype=torch.int64, device=device), step=int(step))
        _update_rotation_lock(state, weight)
        return state
    return RotationLockState(targets=targets, threshold=float(lock.threshold), eligible=eligible,
                             locked=torch.as_tensor(np.asarray(snapshot["locked"]), dtype=torch.bool, device=device),
                             direction=torch.as_tensor(np.asarray(snapshot["direction"]), dtype=weight.dtype, device=device),
                             lock_step=torch.as_tensor(np.asarray(snapshot["lock_step"]), dtype=torch.int64, device=device),
                             step=int(snapshot.get("step", step)))


def _update_rotation_lock(state: RotationLockState, weight: torch.Tensor) -> None:
    """Lock every free eligible row whose |cosine| with some target reaches the threshold."""
    free = state.eligible & ~state.locked
    if not bool(free.any()):
        return
    rows = free.nonzero().squeeze(1)
    w = weight[rows]
    cos = (w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)) @ state.targets.T
    hit = cos.abs().amax(dim=1) >= state.threshold
    if not bool(hit.any()):
        return
    idx = rows[hit]
    state.locked[idx] = True
    state.direction[idx] = w[hit] / w[hit].norm(dim=1, keepdim=True).clamp_min(1e-12)
    state.lock_step[idx] = state.step


def _project_locked_rows(model: ReLUMLP, state: RotationLockState) -> None:
    """Put every locked first-layer row back on its locked direction (keeps only the radial part of the update)."""
    if not bool(state.locked.any()):
        return
    with torch.no_grad():
        weight = model.hidden_layers[0].weight
        idx = state.locked.nonzero().squeeze(1)
        d = state.direction[idx]
        weight[idx] = (weight[idx] * d).sum(dim=1, keepdim=True) * d


def _advance_rotation_lock(model: ReLUMLP, state: RotationLockState | None) -> None:
    """After an optimizer step: project the locked rows, count the step, lock newly aligned rows."""
    if state is None:
        return
    _project_locked_rows(model, state)
    state.step += 1
    _update_rotation_lock(state, model.hidden_layers[0].weight.detach())


ADAM_OPTIMIZER_NAMES = frozenset({"adamw", "adam"})
SGD_OPTIMIZER_NAMES = frozenset({"sgd", "gd"})
HYBRID_OPTIMIZER_NAMES = frozenset({"sgd_adam", "sgd-adam", "sgdadam"})
SCALAR_ADAM_OPTIMIZER_NAMES = frozenset({"scalar_adam", "scalar-adam", "scalaradam"})


def normalized_optimizer_name(config: ReLUTrainConfig) -> str:
    """Return "adamw", "sgd", "sgd_adam", or "scalar_adam" for a config's optimizer string."""
    name = config.optimizer.strip().lower()
    if name in ADAM_OPTIMIZER_NAMES:
        return "adamw"
    if name in SGD_OPTIMIZER_NAMES:
        return "sgd"
    if name in HYBRID_OPTIMIZER_NAMES:
        return "sgd_adam"
    if name in SCALAR_ADAM_OPTIMIZER_NAMES:
        return "scalar_adam"
    raise ValueError(
        'optimizer must be "adamw", "sgd", "sgd_adam", or "scalar_adam", '
        f"got {config.optimizer!r}"
    )


def split_named_parameters_sgd_adam(
    model: ReLUMLP,
    split_layer: int,
) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    """Split model parameters into an SGD part (early layers) and an AdamW part.

    Hidden layers with index ``< split_layer`` (weights and biases) go to the SGD
    part; the remaining hidden layers and the output layer go to the AdamW part.
    ``split_layer`` must be in ``[1, len(model.hidden_dims)]``: a value of 0 or of
    ``len(hidden_dims) + 1`` would leave one part empty, which is just plain
    ``"adamw"`` or ``"sgd"`` and should be requested as such.
    """
    num_hidden = len(model.hidden_dims)
    if not isinstance(split_layer, int) or isinstance(split_layer, bool):
        raise TypeError(f"sgd_adam_split_layer must be an int, got {split_layer!r}.")
    if not 1 <= split_layer <= num_hidden:
        raise ValueError(
            "sgd_adam_split_layer must be in [1, len(hidden_dims)] = "
            f"[1, {num_hidden}], got {split_layer}. Use optimizer='adamw' or "
            "optimizer='sgd' for the all-one-optimizer cases."
        )

    sgd_named: list[tuple[str, torch.nn.Parameter]] = []
    adam_named: list[tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        if name.startswith("hidden_layers."):
            layer_index = int(name.split(".")[1])
            target = sgd_named if layer_index < split_layer else adam_named
        else:
            target = adam_named
        target.append((name, param))
    return sgd_named, adam_named


def optimizer_group_names(config: ReLUTrainConfig) -> list[str]:
    """Names of the optimizers `build_relu_optimizers` returns, in the same order."""
    kind = normalized_optimizer_name(config)
    return ["sgd", "adamw"] if kind == "sgd_adam" else [kind]


def primary_optimizer_index(config: ReLUTrainConfig) -> int:
    """Index of the optimizer whose LR is reported as the scalar ``lr_history``.

    For the hybrid this is the AdamW part, because ``config.lr`` is the AdamW LR.
    """
    names = optimizer_group_names(config)
    return names.index("adamw") if "adamw" in names else 0


class ScalarAdam(torch.optim.Optimizer):
    """Adam with ONE scalar second-moment estimate shared by every parameter.

    Per step, with ``g`` the gradient of all parameters and ``t`` the step count::

        m_t = beta1 * m_{t-1} + (1 - beta1) * g            (per parameter)
        v_t = beta2 * v_{t-1} + (1 - beta2) * |g|^2        (one scalar, |g| over all parameters)
        p  -= lr * (m_t / (1 - beta1^t)) / (sqrt(v_t / (1 - beta2^t)) + eps)

    So the update direction is that of SGD with momentum ``beta1`` (in its EMA form) and only the
    step length is normalised, by the running RMS of the gradient norm: the step is at most about
    ``lr`` in parameter space, whatever the gradient scale. Weight decay is decoupled (as in AdamW).
    The state stores ``exp_avg`` per parameter and the shared ``exp_avg_sq`` and ``step`` as 0-dim
    tensors in every parameter's state, so it saves and loads like Adam's.
    """

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        if lr < 0 or eps < 0 or not (0 <= betas[0] < 1 and 0 <= betas[1] < 1) or weight_decay < 0:
            raise ValueError(f"invalid ScalarAdam hyperparameters lr={lr}, betas={betas}, eps={eps}, weight_decay={weight_decay}")
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D401 - torch API
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        with_grad = [(group, p) for group in self.param_groups for p in group["params"] if p.grad is not None]
        if not with_grad:
            return loss
        sq = sum((p.grad * p.grad).sum() for _, p in with_grad)
        for group, p in with_grad:
            st = self.state[p]
            if len(st) == 0:
                st["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros((), dtype=torch.float32, device=p.device)
            beta1, beta2 = group["betas"]
            st["step"] += 1
            st["exp_avg"].mul_(beta1).add_(p.grad, alpha=1 - beta1)
            st["exp_avg_sq"].mul_(beta2).add_(sq.to(st["exp_avg_sq"].dtype), alpha=1 - beta2)
            t = st["step"]
            bc1 = 1 - beta1 ** t; bc2 = 1 - beta2 ** t
            denom = (st["exp_avg_sq"] / bc2).sqrt() + group["eps"]
            if group["weight_decay"] != 0:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(st["exp_avg"] / (bc1 * denom), alpha=-group["lr"])
        return loss


def build_relu_optimizers(
    model: ReLUMLP,
    config: ReLUTrainConfig,
) -> list[torch.optim.Optimizer]:
    """Build the optimizer(s) for a config, as a list in `optimizer_group_names` order.

    ``"adamw"``, ``"sgd"`` and ``"scalar_adam"`` (`ScalarAdam`) return a single optimizer over all parameters.
    ``"sgd_adam"`` returns ``[SGD over early layers, AdamW over the rest]``, split by
    ``config.sgd_adam_split_layer``.

    **On choosing ``sgd_lr``.** AdamW's first step moves each coordinate by about
    ``lr``, independent of the gradient's magnitude, whereas SGD moves it by
    ``sgd_lr * |g|``. At initialization ``|g|`` is small, so reusing ``lr`` for the
    SGD part leaves the early layers effectively frozen next to the AdamW part.
    Matching the two step scales needs ``sgd_lr`` on the order of
    ``lr / median|g|``; `suggest_sgd_lr_for_adam_parity` measures this. ``sgd_lr``
    defaults to ``lr`` so nothing is silently guessed, but that default is rarely
    the comparison you want.

    **On ``sgd_weight_decay``.** SGD's weight decay is coupled (it enters the
    gradient), while AdamW's is decoupled. A large ``sgd_lr`` therefore multiplies
    the decay: the shrink factor per step is ``1 - sgd_lr * sgd_weight_decay``, which
    is destructive once ``sgd_lr * sgd_weight_decay`` approaches 1.
    """
    kind = normalized_optimizer_name(config)
    if kind == "adamw":
        return [
            torch.optim.AdamW(
                model.parameters(), lr=config.lr, weight_decay=config.weight_decay
            )
        ]
    if kind == "sgd":
        return [
            torch.optim.SGD(
                model.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                nesterov=False,
            )
        ]
    if kind == "scalar_adam":
        return [ScalarAdam(model.parameters(), lr=config.lr, betas=tuple(config.adam_betas), weight_decay=config.weight_decay)]

    sgd_named, adam_named = split_named_parameters_sgd_adam(
        model, config.sgd_adam_split_layer
    )
    sgd_lr = config.lr if config.sgd_lr is None else config.sgd_lr
    sgd_weight_decay = (
        config.weight_decay if config.sgd_weight_decay is None else config.sgd_weight_decay
    )
    sgd_optimizer = torch.optim.SGD(
        [param for _, param in sgd_named],
        lr=sgd_lr,
        momentum=config.momentum,
        weight_decay=sgd_weight_decay,
        nesterov=False,
    )
    adam_optimizer = torch.optim.AdamW(
        [param for _, param in adam_named],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    return [sgd_optimizer, adam_optimizer]


def optimizer_parameter_names(
    model: ReLUMLP,
    config: ReLUTrainConfig,
) -> list[list[str]]:
    """Parameter names handled by each optimizer, in `optimizer_group_names` order."""
    kind = normalized_optimizer_name(config)
    if kind != "sgd_adam":
        return [[name for name, _ in model.named_parameters()]]
    sgd_named, adam_named = split_named_parameters_sgd_adam(
        model, config.sgd_adam_split_layer
    )
    return [[n for n, _ in sgd_named], [n for n, _ in adam_named]]


def build_relu_schedulers(
    optimizers: list[torch.optim.Optimizer],
    config: ReLUTrainConfig,
) -> list[torch.optim.lr_scheduler.LRScheduler]:
    """Cosine-anneal every optimizer, each from its own base LR."""
    schedulers = []
    for optimizer in optimizers:
        base_lr = float(optimizer.param_groups[0]["lr"])
        schedulers.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=max(1, config.epochs),
                eta_min=base_lr * config.lr_scheduler_eta_min_ratio,
            )
        )
    return schedulers


def relu_training_step(
    *,
    model: ReLUMLP,
    optimizers: list[torch.optim.Optimizer],
    criterion: nn.Module,
    x_batch: torch.Tensor,
    y_batch_bin: torch.Tensor,
    first_layer_projection_vector: torch.Tensor | None,
    parameter_update_masks: dict[str, torch.Tensor],
    rotation_lock: RotationLockState | None = None,
) -> torch.Tensor:
    """Run one mini-batch update and return the detached batch loss.

    This is the single definition of a training step: `train_relu_mlp` and the
    replay helpers in `relu_analysis` both call it, so they cannot drift apart.
    Leaves ``param.grad`` populated (after projection and masking) so callers can
    record the gradient the optimizers actually consumed. With ``rotation_lock`` the
    locked first-layer rows keep only the radial part of this step's update, and rows
    that reach the lock threshold after the step are locked (see `FirstLayerRotationLock`).
    """
    for optimizer in optimizers:
        optimizer.zero_grad()
    logits = model(x_batch)
    loss = criterion(logits, y_batch_bin)
    loss.backward()
    _project_first_layer_weight_gradient(model, first_layer_projection_vector)
    _apply_parameter_update_mask_gradients(model, parameter_update_masks)
    frozen_parameter_values = _snapshot_frozen_parameter_values(model, parameter_update_masks)
    for optimizer in optimizers:
        optimizer.step()
    _restore_frozen_parameter_values(model, parameter_update_masks, frozen_parameter_values)
    _advance_rotation_lock(model, rotation_lock)
    return loss.detach()


def suggest_sgd_lr_for_adam_parity(
    *,
    model: ReLUMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: ReLUTrainConfig,
    num_samples: int | None = 100_000,
) -> dict[str, float]:
    """Measure the SGD LR that matches AdamW's first-step size on the SGD-managed layers.

    AdamW's first step moves each coordinate by ``lr * |g| / (|g| + eps) ~ lr``.
    Plain SGD moves it by ``sgd_lr * |g|``. Equating the two at the median
    SGD-managed coordinate gives ``sgd_lr ~ lr / median|g|``, which this returns as
    ``suggested_sgd_lr``.

    Also returns ``decay_fraction_per_step = suggested_sgd_lr * sgd_weight_decay``,
    the per-step multiplicative shrinkage that coupled SGD weight decay would apply
    at that LR. Values approaching 1 mean the decay would dominate the update.

    ``num_samples`` caps how many training rows the gradient is measured on; None
    uses all of them. The model is not modified and no step is taken.
    """
    device = torch.device(config.device)
    model = model.to(device)
    was_training = model.training
    model.eval()

    n = x_train.shape[0] if num_samples is None else min(int(num_samples), x_train.shape[0])
    x = torch.from_numpy(x_train[:n]).float().to(device)
    y_bin = _labels_pm1_to_binary(torch.from_numpy(y_train[:n]).float().to(device))

    model.zero_grad(set_to_none=True)
    nn.BCEWithLogitsLoss()(model(x), y_bin).backward()

    sgd_names = set(optimizer_parameter_names(model, config)[0])
    grads = [
        p.grad.detach().reshape(-1)
        for name, p in model.named_parameters()
        if name in sgd_names and p.grad is not None
    ]
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    if not grads:
        raise RuntimeError("No gradients found for the SGD-managed parameters.")

    magnitudes = torch.cat(grads).abs()
    median = float(magnitudes.median())
    sgd_weight_decay = (
        config.weight_decay if config.sgd_weight_decay is None else config.sgd_weight_decay
    )
    suggested = float(config.lr / median) if median > 0 else float("inf")
    return {
        "median_abs_grad": median,
        "mean_abs_grad": float(magnitudes.mean()),
        "adam_per_coordinate_step": float(config.lr),
        "suggested_sgd_lr": suggested,
        "decay_fraction_per_step": suggested * sgd_weight_decay,
        "num_samples_used": float(n),
    }


def train_relu_mlp(
    *,
    model: ReLUMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: ReLUTrainConfig,
) -> dict[str, object]:
    """Train ReLU MLP with BCE-with-logits, cosine LR annealing, and optional masks.

    ``config.optimizer`` selects "adamw", "sgd", or the "sgd_adam" hybrid; see
    `build_relu_optimizers`.
    """
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
    optimizers = build_relu_optimizers(model, config)
    schedulers = build_relu_schedulers(optimizers, config)
    group_names = optimizer_group_names(config)
    primary_index = primary_optimizer_index(config)

    num_samples = x.shape[0]
    losses: list[float] = []
    lr_history: list[float] = []
    lr_history_by_group: dict[str, list[float]] = {name: [] for name in group_names}
    checkpoint_snapshots: dict[int, dict[str, torch.Tensor]] = {}
    
    permutation = torch.arange(num_samples, device=device) # Fixed permutation for reproducibility (no shuffling)
    epoch_iter = range(config.epochs + 1)
    if config.show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Training ReLU MLP", leave=False)

    for epoch in epoch_iter:
        if epoch in config.snapshot_epochs:
            checkpoint_snapshots[epoch] = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        # permutation = torch.randperm(num_samples, device=device)
        
        if epoch == config.epochs:
            break

        running_loss = 0.0
        seen = 0
        for start in range(0, num_samples, config.batch_size):
            end = min(start + config.batch_size, num_samples)
            idx = permutation[start:end]

            loss = relu_training_step(
                model=model,
                optimizers=optimizers,
                criterion=criterion,
                x_batch=x[idx],
                y_batch_bin=y_bin[idx],
                first_layer_projection_vector=first_layer_projection_vector,
                parameter_update_masks=parameter_update_masks,
            )

            batch_n = end - start
            running_loss += float(loss.cpu()) * batch_n
            seen += batch_n

        losses.append(running_loss / max(1, seen))
        lr_history.append(float(optimizers[primary_index].param_groups[0]["lr"]))
        for name, optimizer in zip(group_names, optimizers):
            lr_history_by_group[name].append(float(optimizer.param_groups[0]["lr"]))
        for scheduler in schedulers:
            scheduler.step()
        if config.show_progress:
            epoch_iter.set_postfix(loss=f"{losses[-1]:.4f}")

    return {
        "model": model,
        "epoch_losses": losses,
        "lr_history": lr_history,
        "lr_history_by_group": lr_history_by_group,
        "checkpoint_snapshots": checkpoint_snapshots,
    }
