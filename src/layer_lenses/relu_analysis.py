"""Analysis helpers for ReLU MLPs trained on COB-ODT data.

These utilities keep exploratory notebooks focused on plotting choices and
interpretation while preserving the reusable mechanics in importable code.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from layer_lenses.odt import (
    COBODTTree,
    build_axis_aligned_cob_odt_tree,
    generate_cob_odt_data,
    odt_leaf_ids_for_x,
)
from layer_lenses.relu_mlp import ReLUMLP
from layer_lenses.relu_training import (
    ReLUTrainConfig,
    _first_layer_projection_vector,
    _labels_pm1_to_binary,
    _parameter_update_masks,
    build_relu_optimizers,
    build_relu_schedulers,
    checkpoint_model_from_state,
    evaluate_relu_mlp,
    optimizer_group_names,
    normalized_optimizer_name,
    optimizer_parameter_names,
    primary_optimizer_index,
    relu_training_step,
    rotation_lock_state,
    train_relu_mlp,
)
from layer_lenses.training import set_seed


def log_loss_gradients(
    *,
    model: ReLUMLP,
    x: np.ndarray,
    y: np.ndarray,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Gradients of mean BCE-with-logits log loss w.r.t. all parameters, no weight updates.

    Uses the same label convention as training: ``y`` in ``{-1, +1}``, mapped to
    ``{0, 1}`` for ``binary_cross_entropy_with_logits``. The objective is the **mean**
    loss over the ``n`` rows of ``x`` (so gradients scale like ``1/n`` vs a sum loss).

    Returns a dictionary keyed like ``model.state_dict()``; each value is the gradient
    tensor for that parameter, detached and copied to CPU float32. The model's
    parameter tensors are not modified; gradients are cleared before returning.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (n, d), got shape {x.shape}.")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError(
            f"y must be 1D with length n={x.shape[0]}, got shape {y.shape}."
        )
    if not np.all((y == -1) | (y == 1)):
        raise ValueError("y must contain only -1 and +1.")

    dev = torch.device(device)
    model = model.to(dev)
    was_training = model.training
    model.eval()

    x_t = torch.from_numpy(x).float().to(dev)
    y_pm1 = torch.from_numpy(y).float().to(dev)
    y_bin = (y_pm1 + 1.0) / 2.0

    model.zero_grad(set_to_none=True)
    logits = model(x_t)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_bin)
    loss.backward()

    grad_dict: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            raise RuntimeError(f"No gradient computed for parameter {name!r}.")
        grad_dict[name] = param.grad.detach().cpu().clone()

    model.zero_grad(set_to_none=True)
    model.train(was_training)

    return grad_dict


def quadratic_snapshot_epochs(total_epochs: int, n_points: int = 80) -> tuple[int, ...]:
    """Return snapshot epochs dense near initialization and sparser later."""
    if n_points < 2:
        return (0, total_epochs)
    vals = []
    for i in range(n_points):
        frac = i / (n_points - 1)
        vals.append(int(round((frac**2) * total_epochs)))
    vals.extend([0, total_epochs])
    return tuple(sorted(set(vals)))


def run_single_relu_seed(
    master_seed: int,
    *,
    dim: int,
    depth: int,
    n_train: int,
    hidden_dims: list[int],
    total_epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    snapshot_epochs,
    bias: bool = False,
    activation: str = "relu",
    model_init: ReLUMLP | None = None,
    odt_hyperplanes: Literal["random_orthogonal", "coordinate_axes"] = "random_orthogonal",
    parameter_update_mask: dict[str, np.ndarray | torch.Tensor] | None = None,
    threshold: float = 0.0,
    device: str = "cpu",
    show_progress: bool = True,
    lr_scheduler_eta_min_ratio: float = 0.05,
    optimizer: str = "adamw",
    momentum: float = 0.0,
    sgd_adam_split_layer: int = 1,
    sgd_lr: float | None = None,
    sgd_weight_decay: float | None = None,
    project_first_layer_grad_orthogonal: int | None = None,
) -> dict[str, Any]:
    """Generate COB-ODT data, train one ReLU MLP seed, and return notebook-friendly state.

    If ``model_init`` is provided, training starts from a copy of that model instead of
    constructing a new model from ``master_seed + 1000``; the model's own activation
    convention is then used and ``activation`` is ignored.

    If ``project_first_layer_grad_orthogonal`` is an integer, first-layer weight
    gradients are projected orthogonal to that ODT internal node's normal vector.
    """
    data_seed = master_seed
    init_seed = master_seed + 1000
    train_seed = master_seed + 2000

    if odt_hyperplanes == "random_orthogonal":
        generation_tree = None
    elif odt_hyperplanes == "coordinate_axes":
        generation_tree = build_axis_aligned_cob_odt_tree(dim=dim, depth=depth)
    else:
        raise ValueError(
            "odt_hyperplanes must be 'random_orthogonal' or 'coordinate_axes', "
            f"got {odt_hyperplanes!r}."
        )

    x, y, tree, meta = generate_cob_odt_data(
        num_data=2 * n_train,
        dim=dim,
        depth=depth,
        seed=data_seed,
        threshold=threshold,
        tree=generation_tree,
    )
    x_train, y_train = x[:n_train], y[:n_train]
    x_eval, y_eval = x[n_train:], y[n_train:]

    if project_first_layer_grad_orthogonal is None:
        first_layer_projection_vector = None
    elif isinstance(project_first_layer_grad_orthogonal, bool) or not isinstance(
        project_first_layer_grad_orthogonal,
        int,
    ):
        raise TypeError(
            "project_first_layer_grad_orthogonal must be an ODT internal node id "
            f"integer or None, got {project_first_layer_grad_orthogonal!r}."
        )
    elif not 0 <= project_first_layer_grad_orthogonal < tree.w_list.shape[0]:
        raise ValueError(
            "project_first_layer_grad_orthogonal must reference an ODT internal node. "
            f"Expected an integer in [0, {tree.w_list.shape[0] - 1}], "
            f"got {project_first_layer_grad_orthogonal}."
        )
    else:
        first_layer_projection_vector = tree.w_list[project_first_layer_grad_orthogonal]

    if model_init is None:
        set_seed(init_seed)
        model = ReLUMLP(
            input_dim=dim,
            hidden_dims=hidden_dims,
            bias=bias,
            activation=activation,
        )
    else:
        if model_init.input_dim != dim:
            raise ValueError(
                f"model_init input_dim must match dim={dim}, got {model_init.input_dim}."
            )
        if list(model_init.hidden_dims) != list(hidden_dims):
            raise ValueError(
                "model_init hidden_dims must match hidden_dims. "
                f"Expected {hidden_dims}, got {model_init.hidden_dims}."
            )
        model_has_bias = model_init.output_layer.bias is not None
        if model_has_bias != bias:
            raise ValueError(
                f"model_init bias setting must match bias={bias}, got {model_has_bias}."
            )
        model = copy.deepcopy(model_init)
    out = train_relu_mlp(
        model=model,
        x_train=x_train,
        y_train=y_train,
        config=ReLUTrainConfig(
            epochs=total_epochs,
            lr=lr,
            batch_size=batch_size,
            seed=train_seed,
            snapshot_epochs=tuple(snapshot_epochs),
            weight_decay=weight_decay,
            device=device,
            show_progress=show_progress,
            lr_scheduler_eta_min_ratio=lr_scheduler_eta_min_ratio,
            optimizer=optimizer,
            momentum=momentum,
            sgd_adam_split_layer=sgd_adam_split_layer,
            sgd_lr=sgd_lr,
            sgd_weight_decay=sgd_weight_decay,
            first_layer_grad_orthogonal_to=first_layer_projection_vector,
            parameter_update_mask=parameter_update_mask,
        ),
    )

    return {
        "master_seed": master_seed,
        "seeds": {"data_seed": data_seed, "init_seed": init_seed, "train_seed": train_seed},
        "tree": tree,
        "meta": meta,
        "data": {"x_train": x_train, "y_train": y_train, "x_eval": x_eval, "y_eval": y_eval},
        "out": out,
    }


def collect_start_end_metrics(seed_result: dict[str, Any]) -> list[dict[str, float | int | str]]:
    """Evaluate a ReLU run at its first and last checkpoint."""
    out = seed_result["out"]
    ckpts = out["checkpoint_snapshots"]
    epochs_sorted = sorted(ckpts.keys())
    e0, eN = epochs_sorted[0], epochs_sorted[-1]

    x_train = seed_result["data"]["x_train"]
    y_train = seed_result["data"]["y_train"]
    x_eval = seed_result["data"]["x_eval"]
    y_eval = seed_result["data"]["y_eval"]

    rows: list[dict[str, float | int | str]] = []
    for label, ep in [("start", e0), ("end", eN)]:
        model = checkpoint_model_from_state(out["model"], ckpts[ep])
        m_tr = evaluate_relu_mlp(model=model, x_eval=x_train, y_eval=y_train)
        m_te = evaluate_relu_mlp(model=model, x_eval=x_eval, y_eval=y_eval)
        rows.append(
            {
                "timepoint": label,
                "epoch": ep,
                "train_log_loss": float(m_tr["log_loss"]),
                "test_error": float(m_te["zero_one_loss"]),
            }
        )
    return rows


def first_layer_odt_alignment(
    *,
    state_dict: dict[str, torch.Tensor],
    tree: COBODTTree,
) -> dict[str, torch.Tensor | np.ndarray]:
    """Compute first-layer ReLU weight alignment to COB-ODT decision vectors."""
    first_layer_wts = state_dict["hidden_layers.0.weight"].detach().cpu()
    odt_wts = torch.from_numpy(tree.w_list).float()
    row_norms = first_layer_wts.norm(dim=1)
    cos = (first_layer_wts @ odt_wts.T) / (
        row_norms.unsqueeze(1).clamp_min(1e-12)
        * odt_wts.norm(dim=1).unsqueeze(0).clamp_min(1e-12)
    )
    abs_cos = cos.abs()
    max_abs_cos, closest_odt = abs_cos.max(dim=1)
    closest_odt_np = closest_odt.numpy()
    node_level = np.floor(np.log2(closest_odt_np + 1)).astype(int)
    signed_closest_cos = cos[torch.arange(first_layer_wts.shape[0]), closest_odt]

    return {
        "first_layer_wts": first_layer_wts,
        "odt_wts": odt_wts,
        "row_norms": row_norms,
        "cos": cos,
        "abs_cos": abs_cos,
        "max_abs_cos": max_abs_cos,
        "closest_odt": closest_odt,
        "closest_odt_np": closest_odt_np,
        "node_level": node_level,
        "signed_closest_cos": signed_closest_cos,
    }


def plot_first_layer_odt_alignment(
    *,
    state_dict: dict[str, torch.Tensor],
    tree: COBODTTree,
    depth: int,
    epoch: int,
    seed: int | None = None,
    ax=None,
):
    """Plot first-layer norm vs max absolute cosine to ODT decision vectors."""
    alignment = first_layer_odt_alignment(state_dict=state_dict, tree=tree)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    level_names = {i: f"{i}" for i in range(depth)}
    cmap = plt.get_cmap("viridis", depth)
    sc = ax.scatter(
        alignment["row_norms"].numpy(),
        alignment["max_abs_cos"].numpy(),
        c=alignment["node_level"],
        cmap=cmap,
        s=18,
        alpha=0.8,
        vmin=-0.5,
        vmax=depth - 0.5,
    )
    cax = ax.inset_axes([0.86, 0.12, 0.035, 0.36])
    cbar = fig.colorbar(sc, cax=cax, ticks=range(depth))
    cbar.ax.set_yticklabels([level_names[i] for i in range(depth)])
    cbar.set_label("ODT level of argmax |cosine|")
    cbar.ax.tick_params(labelsize=8)
    ax.set_xlabel("First-layer ReLU neuron weight norm")
    ax.set_ylabel("Max |cosine| similarity to ODT vectors")
    seed_text = f"Seed={seed}, " if seed is not None else ""
    ax.set_title(f"{seed_text}Epoch={epoch}: alignment by ODT level")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return alignment, fig, ax


def _layer_y_positions(width: int) -> np.ndarray:
    if width == 1:
        return np.array([0.5])
    return np.linspace(0.98, 0.02, width)


def _hidden_weights_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    hidden_layer_indices = sorted(
        int(name.split(".")[1])
        for name in state_dict
        if name.startswith("hidden_layers.") and name.endswith(".weight")
    )
    num_hidden_layers = len(hidden_layer_indices)
    if not 1 <= num_hidden_layers <= 3:
        raise ValueError(
            "Expected one to three hidden layers, "
            f"but found {num_hidden_layers}."
        )
    if hidden_layer_indices != list(range(num_hidden_layers)):
        raise ValueError(
            "Hidden-layer weight indices must be contiguous and start at zero; "
            f"found {hidden_layer_indices}."
        )

    hidden_weights = [
        state_dict[f"hidden_layers.{layer_idx}.weight"].detach().cpu()
        for layer_idx in hidden_layer_indices
    ]
    output_weight = state_dict["output_layer.weight"].detach().cpu()
    return hidden_weights, output_weight


def _normalize_hidden_node_masks(
    node_mask: np.ndarray | list[np.ndarray] | None,
    hidden_widths: list[int],
    *,
    name: str = "node_mask",
) -> list[np.ndarray]:
    if node_mask is None:
        return [np.ones(width, dtype=bool) for width in hidden_widths]

    if isinstance(node_mask, list):
        if len(node_mask) != len(hidden_widths):
            raise ValueError(
                f"{name} must contain one array per hidden layer. "
                f"Expected {len(hidden_widths)}, got {len(node_mask)}."
            )

        layer_node_masks = []
        for layer_idx, (mask, width) in enumerate(
            zip(node_mask, hidden_widths, strict=True)
        ):
            layer_mask = np.asarray(mask, dtype=bool)
            if layer_mask.ndim != 1 or layer_mask.shape[0] != width:
                raise ValueError(
                    f"{name}[{layer_idx}] must have shape ({width},), "
                    f"got {layer_mask.shape}."
                )
            layer_node_masks.append(layer_mask)
        return layer_node_masks

    rectangular_mask = np.asarray(node_mask, dtype=bool)
    if rectangular_mask.ndim != 2 or rectangular_mask.shape[1] != len(
        hidden_widths
    ):
        raise ValueError(
            f"{name} array must be 2D with one column per hidden layer; "
            f"got shape {rectangular_mask.shape}."
        )
    if rectangular_mask.shape[0] < max(hidden_widths):
        raise ValueError(
            f"{name} array needs at least one row per node in the widest "
            f"hidden layer. Expected at least {max(hidden_widths)}, "
            f"got {rectangular_mask.shape[0]}."
        )
    return [
        rectangular_mask[:width, layer_idx]
        for layer_idx, width in enumerate(hidden_widths)
    ]


def plot_layered_relu_graph(
    *,
    state_dict: dict[str, torch.Tensor],
    tree: COBODTTree,
    depth: int,
    epoch: int,
    node_mask: np.ndarray | list[np.ndarray] | None = None,
    alignment_abs_threshold: float = 0.5,
    edge_abs_threshold: float = 0.2,
    figsize: tuple[float, float] = (14, 18),
    require_both_edge_endpoints_selected: bool = True,
):
    """Plot a layered hidden-node graph with ODT labels on first-layer neurons.

    ``node_mask`` may be either a row-by-layer array or one one-dimensional
    boolean array per hidden layer. The latter supports layers with different
    widths without padding.
    """
    alignment = first_layer_odt_alignment(state_dict=state_dict, tree=tree)
    hidden_weights, output_weight = _hidden_weights_from_state_dict(state_dict)
    num_hidden_layers = len(hidden_weights)
    hidden_widths = [weight.shape[0] for weight in hidden_weights]
    layer_node_masks = _normalize_hidden_node_masks(node_mask, hidden_widths)

    output_layer_idx = num_hidden_layers
    widths = [*hidden_widths, output_weight.shape[0]]
    x_positions = [1.6 * layer_idx for layer_idx in range(num_hidden_layers + 1)]
    layer_names = [
        *(f"hidden {layer_idx + 1}" for layer_idx in range(num_hidden_layers)),
        "output",
    ]

    node_pos = {}
    for layer_idx, width in enumerate(widths):
        ys = _layer_y_positions(width)
        for node_idx, y in enumerate(ys):
            node_pos[(layer_idx, node_idx)] = (x_positions[layer_idx], y)

    fig, ax = plt.subplots(figsize=figsize)
    edge_mats = [
        (hidden_weights[dst_layer], dst_layer - 1, dst_layer)
        for dst_layer in range(1, num_hidden_layers)
    ]
    edge_mats.append(
        (output_weight, num_hidden_layers - 1, output_layer_idx)
    )

    visible_edges = []
    for mat, src_layer, dst_layer in edge_mats:
        for dst_idx in range(mat.shape[0]):
            for src_idx in range(mat.shape[1]):
                weight = float(mat[dst_idx, src_idx])
                if abs(weight) < edge_abs_threshold:
                    continue

                src_selected = bool(layer_node_masks[src_layer][src_idx])
                dst_selected = (
                    True
                    if dst_layer == output_layer_idx
                    else bool(layer_node_masks[dst_layer][dst_idx])
                )
                keep_edge = src_selected and dst_selected
                if not require_both_edge_endpoints_selected:
                    keep_edge = src_selected or dst_selected
                if keep_edge:
                    visible_edges.append((src_layer, src_idx, dst_layer, dst_idx, weight))

    max_visible_abs = max((abs(w) for *_, w in visible_edges), default=1.0)

    for src_layer, src_idx, dst_layer, dst_idx, weight in visible_edges:
        x0, y0 = node_pos[(src_layer, src_idx)]
        x1, y1 = node_pos[(dst_layer, dst_idx)]
        edge_color = "#d62728" if weight > 0 else "#1f77b4"
        edge_alpha = 0.15 + 0.65 * (abs(weight) / max_visible_abs)
        edge_lw = 0.4 + 2.5 * (abs(weight) / max_visible_abs)
        ax.plot([x0, x1], [y0, y1], color=edge_color, alpha=edge_alpha, linewidth=edge_lw, zorder=1)

    max_abs_cos = alignment["max_abs_cos"]
    signed_closest_cos = alignment["signed_closest_cos"]
    closest_odt = alignment["closest_odt"]

    for layer_idx, width in enumerate(widths):
        xs = [node_pos[(layer_idx, i)][0] for i in range(width)]
        ys = [node_pos[(layer_idx, i)][1] for i in range(width)]
        if layer_idx == 0:
            colors = []
            for i in range(width):
                if float(max_abs_cos[i]) < alignment_abs_threshold:
                    colors.append("#bdbdbd")
                elif float(signed_closest_cos[i]) >= 0:
                    colors.append("#d62728")
                else:
                    colors.append("#1f77b4")
        elif layer_idx < num_hidden_layers:
            colors = ["#f2f2f2"] * width
        else:
            colors = ["#f7d774"]

        ax.scatter(
            xs,
            ys,
            s=260 if layer_idx < num_hidden_layers else 520,
            c=colors,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )

    for i in range(hidden_widths[0]):
        x, y = node_pos[(0, i)]
        label = f"{int(closest_odt[i])}\n{float(signed_closest_cos[i]):+.2f}"
        ax.text(x, y, label, ha="center", va="center", fontsize=5, zorder=4)

    for layer_idx in range(1, num_hidden_layers):
        for i in range(widths[layer_idx]):
            x, y = node_pos[(layer_idx, i)]
            ax.text(x, y, str(i), ha="center", va="center", fontsize=5, zorder=4)

    ax.text(
        *node_pos[(output_layer_idx, 0)],
        "out",
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
        zorder=4,
    )
    for layer_idx, name in enumerate(layer_names):
        ax.text(
            x_positions[layer_idx],
            1.045,
            name,
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
        )

    ax.set_title(
        f"ReLU MLP connectivity at epoch {epoch}\n"
        f"First-layer color: signed cosine to closest ODT node; gray if |cos| < {alignment_abs_threshold}. "
        f"Edges shown if |weight| >= {edge_abs_threshold}.",
        fontsize=13,
    )
    ax.set_xlim(-0.45, x_positions[-1] + 0.45)
    ax.set_ylim(-0.04, 1.08)
    ax.axis("off")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d62728",
            markeredgecolor="black",
            markersize=10,
            label="positive closest ODT cosine",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1f77b4",
            markeredgecolor="black",
            markersize=10,
            label="negative closest ODT cosine",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#bdbdbd",
            markeredgecolor="black",
            markersize=10,
            label=f"|cos| < {alignment_abs_threshold}",
        ),
        plt.Line2D([0], [0], color="#d62728", lw=2, label="positive ReLU edge weight"),
        plt.Line2D([0], [0], color="#1f77b4", lw=2, label="negative ReLU edge weight"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3)

    total_edges = sum(weight.numel() for weight, _, _ in edge_mats)
    print(f"Visible edges: {len(visible_edges)} / {total_edges}")
    print(
        "First-layer aligned nodes: "
        f"{int((max_abs_cos >= alignment_abs_threshold).sum())} / "
        f"{hidden_widths[0]}"
    )
    return alignment, visible_edges, fig, ax


def first_layer_neuron_weighted_lens(
    *,
    state_dict: dict[str, torch.Tensor],
    activation_tensor: np.ndarray,
    first_layer_neuron_id: int,
    node_mask: np.ndarray | list[np.ndarray] | None = None,
) -> np.ndarray:
    """Sum active path weights from one first-layer neuron to the output.

    The result contains one value per example in ``activation_tensor``. A
    per-layer ``node_mask`` can restrict the sum to paths through selected
    hidden neurons.

    Write the bias-free network with hidden layers ``i = 0, ..., M-1`` as

        h_{-1} = x,   z_i = W_i h_{i-1},   h_i = phi(z_i),   yhat = a . h_{M-1}

    with ``phi = ReLU`` and blinders ``g_i[j] = phi'(z_i[j])`` in ``{0, 1}``
    (the boolean gates supplied in ``activation_tensor``). For first-layer
    neuron ``k`` this function returns

        Lambda_k(x) = d yhat / d z_0[k]
                    = g_0[k] * sum over paths k -> output of
                        (product of the weights along the path)
                      * (product of the blinders of the *interior* neurons on it)

    i.e. the lens that the rest of the network presents to neuron ``k``, times
    that neuron's own blinder. It is exactly the scalar in the first-layer chain
    rule,

        dL / dw_k = E[ (dL / d yhat) * Lambda_k(x) * x ],

    so it is the "lens x blinder" half of the residue x lens x blinder x y x
    factorisation the notebooks use.

    With a single hidden layer (``M = 1``) there are no interior neurons, the
    path sum is the single edge ``a_k``, and the result collapses to
    ``a_k * g_0[k]`` -- taking only the two values ``{0, a_k}``. That is the
    no-lens case: the factor no longer depends on the other neurons at all.
    """
    hidden_weights, output_weight = _hidden_weights_from_state_dict(state_dict)
    hidden_widths = [weight.shape[0] for weight in hidden_weights]
    num_hidden_layers = len(hidden_widths)

    if not 0 <= first_layer_neuron_id < hidden_widths[0]:
        raise ValueError(
            "first_layer_neuron_id must be in "
            f"[0, {hidden_widths[0] - 1}], got {first_layer_neuron_id}."
        )
    if activation_tensor.ndim != 3:
        raise ValueError(
            "activation_tensor must have shape (layers, neurons, num_data), "
            f"got {activation_tensor.shape}."
        )
    if activation_tensor.shape[0] != num_hidden_layers:
        raise ValueError(
            "activation_tensor must contain one slice per hidden layer. "
            f"Expected {num_hidden_layers}, got {activation_tensor.shape[0]}."
        )
    if activation_tensor.shape[1] < max(hidden_widths):
        raise ValueError(
            "activation_tensor has too few neuron entries for the widest "
            f"hidden layer. Expected at least {max(hidden_widths)}, "
            f"got {activation_tensor.shape[1]}."
        )
    if output_weight.shape != (1, hidden_widths[-1]):
        raise ValueError(
            "Expected a scalar output layer with weight shape "
            f"(1, {hidden_widths[-1]}), got {tuple(output_weight.shape)}."
        )

    layer_node_masks = _normalize_hidden_node_masks(node_mask, hidden_widths)
    activations = np.asarray(activation_tensor, dtype=bool)

    # ``downstream`` is the reverse accumulation ``d yhat / d h_{i-1}``, carried
    # one layer at a time from the output back toward the first layer. Entry
    # ``[j, n]`` is that derivative for neuron ``j`` of layer ``i-1`` on example
    # ``n``. Equivalently: the sum over all paths from that neuron to the output
    # of the product of path weights, keeping only paths whose interior neurons
    # are all active and unmasked.
    #
    # Seed at the output layer: ``d yhat / d h_{M-1}[j] = a_j``, the same for
    # every example. The trailing axis is therefore length 1; it broadcasts
    # against the per-example gates below and becomes example-dependent at the
    # first gating step.
    downstream = output_weight[0].numpy()[:, None]          # (w_{M-1}, 1)

    for layer_idx in range(num_hidden_layers - 1, 0, -1):
        width = hidden_widths[layer_idx]

        # Apply layer i's blinders, turning d yhat / d h_i into d yhat / d z_i:
        #
        #     d yhat / d z_i[j] = g_i[j] * m_i[j] * d yhat / d h_i[j]
        #
        # The mask factor ``m_i`` is how the caller deletes paths: a zero there
        # drops every path through neuron j from the sum, exactly as a closed
        # gate would.
        gated_downstream = (
            activations[layer_idx, :width]                  # (width, n)  gates g_i
            * layer_node_masks[layer_idx][:, None]          # (width, 1)  mask m_i
            * downstream                                    # (width, n) or (width, 1)
        )                                                   # -> (width, n)

        # Step back through the linear map z_i = W_i h_{i-1}:
        #
        #     d yhat / d h_{i-1}[j'] = sum_j W_i[j, j'] * d yhat / d z_i[j]
        #
        # which is W_i transposed applied to the gated vector. One such product
        # per iteration extends every path by one more edge.
        downstream = hidden_weights[layer_idx].numpy().T @ gated_downstream
        #            (w_{i-1}, width) @ (width, n)  ->  (w_{i-1}, n)

    # The loop stops before ``layer_idx == 0``, so ``downstream`` now holds
    # ``d yhat / d h_0``: all path weights from each first-layer neuron to the
    # output, with the blinders of every interior neuron applied but *not* the
    # first layer's own. That remaining factor is the lens Lambda_k in the paper's
    # sense -- it depends only on the other neurons.
    #
    # Multiplying by neuron k's own gate (and its mask entry) closes the chain
    # rule at ``d yhat / d z_0[k]``. When ``M == 1`` the loop body never ran and
    # ``downstream`` is still the seed ``a``, giving ``a_k * g_0[k]``.
    first_layer_active = (
        activations[0, first_layer_neuron_id]
        * layer_node_masks[0][first_layer_neuron_id]
    )
    return first_layer_active * downstream[first_layer_neuron_id]   # (n,)


def masked_relu_logits(
    model: ReLUMLP,
    x: np.ndarray,
    node_mask: np.ndarray | list[np.ndarray],
    *,
    batch_size: int = 8192,
    device: str = "cpu",
) -> np.ndarray:
    """Run a ReLU MLP while zeroing hidden activations selected by a mask."""
    model = model.to(device)
    model.eval()

    hidden_widths = [layer.out_features for layer in model.hidden_layers]
    keep = [
        torch.as_tensor(mask, device=device)
        for mask in _normalize_hidden_node_masks(node_mask, hidden_widths)
    ]

    x_t = torch.from_numpy(x).float().to(device)
    logits_batches = []

    with torch.no_grad():
        for start in range(0, x_t.shape[0], batch_size):
            h = x_t[start : start + batch_size]

            for layer_idx, layer in enumerate(model.hidden_layers):
                h = model.activation.apply(layer(h))
                layer_keep = keep[layer_idx].float()
                h = h * layer_keep.unsqueeze(0)

            logits = model.output_layer(h).squeeze(-1)
            logits_batches.append(logits.detach().cpu())

    return torch.cat(logits_batches).numpy()


def masked_accuracy_by_odt_leaf(
    *,
    model: ReLUMLP,
    node_mask: np.ndarray | list[np.ndarray],
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    tree: COBODTTree,
    batch_size: int = 8192,
    device: str = "cpu",
) -> pd.DataFrame:
    """Aggregate masked-model accuracy and log loss by reached ODT leaf."""
    logits = masked_relu_logits(
        model=model,
        x=x_eval,
        node_mask=node_mask,
        batch_size=batch_size,
        device=device,
    )
    y_eval_pm1 = y_eval.astype(np.int8)
    pred_pm1 = np.where(logits >= 0, 1, -1).astype(np.int8)
    per_example_log_loss = np.logaddexp(0.0, -y_eval_pm1.astype(np.float64) * logits)

    leaf_node = odt_leaf_ids_for_x(x_eval, tree)
    num_internal = tree.w_list.shape[0]

    rows = pd.DataFrame(
        {
            "leaf_node": leaf_node,
            "leaf_index": leaf_node - num_internal,
            "true_label": y_eval_pm1,
            "pred_label": pred_pm1,
            "correct": pred_pm1 == y_eval_pm1,
            "logit": logits,
            "log_loss": per_example_log_loss,
        }
    )

    return (
        rows.groupby(["leaf_node", "leaf_index", "true_label"], as_index=False)
        .agg(
            num_samples=("correct", "size"),
            num_correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_logit=("logit", "mean"),
            mean_abs_logit=("logit", lambda s: float(np.mean(np.abs(s)))),
        )
        .sort_values("leaf_node")
        .reset_index(drop=True)
    )


def neuron_leaf_responsibility_from_ablation(
    *,
    model: ReLUMLP,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    tree: COBODTTree,
    base_node_mask: np.ndarray | list[np.ndarray] | None = None,
    min_log_loss_delta: float = 0.05,
    min_relative_log_loss_delta: float | None = None,
    min_accuracy_drop: float | None = None,
    batch_size: int = 8192,
    device: str = "cpu",
) -> tuple[dict[tuple[int, int], dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Assign hidden neurons to ODT leaves by single-neuron ablation impact."""
    hidden_widths = [layer.out_features for layer in model.hidden_layers]
    layer_node_masks = [
        mask.copy()
        for mask in _normalize_hidden_node_masks(
            base_node_mask,
            hidden_widths,
            name="base_node_mask",
        )
    ]

    baseline_df = masked_accuracy_by_odt_leaf(
        model=model,
        node_mask=layer_node_masks,
        x_eval=x_eval,
        y_eval=y_eval,
        tree=tree,
        batch_size=batch_size,
        device=device,
    ).rename(
        columns={
            "accuracy": "baseline_accuracy",
            "mean_log_loss": "baseline_mean_log_loss",
            "mean_logit": "baseline_mean_logit",
            "mean_abs_logit": "baseline_mean_abs_logit",
        }
    )

    responsibility: dict[tuple[int, int], dict[str, Any]] = {}
    detail_rows: list[dict[str, Any]] = []

    for layer_idx, width in enumerate(hidden_widths):
        for neuron_id in range(width):
            if not layer_node_masks[layer_idx][neuron_id]:
                continue

            ablated_mask = [mask.copy() for mask in layer_node_masks]
            ablated_mask[layer_idx][neuron_id] = False

            ablated_df = masked_accuracy_by_odt_leaf(
                model=model,
                node_mask=ablated_mask,
                x_eval=x_eval,
                y_eval=y_eval,
                tree=tree,
                batch_size=batch_size,
                device=device,
            ).rename(
                columns={
                    "accuracy": "ablated_accuracy",
                    "mean_log_loss": "ablated_mean_log_loss",
                    "mean_logit": "ablated_mean_logit",
                    "mean_abs_logit": "ablated_mean_abs_logit",
                }
            )

            merged = baseline_df.merge(
                ablated_df[
                    [
                        "leaf_node",
                        "ablated_accuracy",
                        "ablated_mean_log_loss",
                        "ablated_mean_logit",
                        "ablated_mean_abs_logit",
                    ]
                ],
                on="leaf_node",
                how="inner",
            )
            merged["log_loss_delta"] = (
                merged["ablated_mean_log_loss"] - merged["baseline_mean_log_loss"]
            )
            merged["relative_log_loss_delta"] = merged["log_loss_delta"] / merged[
                "baseline_mean_log_loss"
            ].clip(lower=1e-12)
            merged["accuracy_drop"] = merged["baseline_accuracy"] - merged["ablated_accuracy"]

            selected = merged[merged["log_loss_delta"] >= min_log_loss_delta].copy()
            if min_relative_log_loss_delta is not None:
                selected = selected[
                    selected["relative_log_loss_delta"] >= min_relative_log_loss_delta
                ]
            if min_accuracy_drop is not None:
                selected = selected[selected["accuracy_drop"] >= min_accuracy_drop]

            selected = selected.sort_values("log_loss_delta", ascending=False).reset_index(drop=True)
            if selected.empty:
                responsibility[(layer_idx, neuron_id)] = {
                    "leaf_nodes": [],
                    "num_responsible_leaves": 0,
                    "total_weighted_log_loss_delta": 0.0,
                    "max_log_loss_delta": 0.0,
                    "details": [],
                }
                continue

            details = []
            for row in selected.to_dict(orient="records"):
                detail = {
                    "leaf_node": int(row["leaf_node"]),
                    "leaf_index": int(row["leaf_index"]),
                    "true_label": int(row["true_label"]),
                    "num_samples": int(row["num_samples"]),
                    "baseline_accuracy": float(row["baseline_accuracy"]),
                    "ablated_accuracy": float(row["ablated_accuracy"]),
                    "accuracy_drop": float(row["accuracy_drop"]),
                    "baseline_mean_log_loss": float(row["baseline_mean_log_loss"]),
                    "ablated_mean_log_loss": float(row["ablated_mean_log_loss"]),
                    "log_loss_delta": float(row["log_loss_delta"]),
                    "relative_log_loss_delta": float(row["relative_log_loss_delta"]),
                }
                details.append(detail)
                detail_rows.append({"layer": layer_idx, "neuron_id": neuron_id, **detail})

            responsibility[(layer_idx, neuron_id)] = {
                "leaf_nodes": [d["leaf_node"] for d in details],
                "num_responsible_leaves": len(details),
                "total_weighted_log_loss_delta": float(
                    np.average(selected["log_loss_delta"], weights=selected["num_samples"])
                ),
                "max_log_loss_delta": float(selected["log_loss_delta"].max()),
                "details": details,
            }

    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(
            ["layer", "neuron_id", "log_loss_delta"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    return responsibility, detail_df, baseline_df


def summarize_neuron_responsibility(
    responsibility: dict[tuple[int, int], dict[str, Any]],
) -> pd.DataFrame:
    """Return one compact row per neuron with at least one attributed ODT leaf."""
    rows = [
        {
            "layer": layer,
            "neuron_id": neuron_id,
            "leaf_nodes": payload["leaf_nodes"],
            "num_responsible_leaves": payload["num_responsible_leaves"],
            "max_log_loss_delta": payload["max_log_loss_delta"],
            "total_weighted_log_loss_delta": payload["total_weighted_log_loss_delta"],
        }
        for (layer, neuron_id), payload in responsibility.items()
        if payload["num_responsible_leaves"] > 0
    ]
    if not rows:
        return pd.DataFrame(
            columns=[
                "layer",
                "neuron_id",
                "leaf_nodes",
                "num_responsible_leaves",
                "max_log_loss_delta",
                "total_weighted_log_loss_delta",
            ]
        )
    return pd.DataFrame(rows).sort_values(["layer", "neuron_id"]).reset_index(drop=True)


def _precompute_hidden_activation_masks_and_leaf_ids(
    *,
    model: ReLUMLP,
    x_eval: np.ndarray,
    tree: COBODTTree,
    batch_size: int = 8192,
    device: str = "cpu",
) -> dict[str, Any]:
    """Precompute hidden ReLU activation masks and ODT leaf ids for ``x_eval``."""
    precomputed = _precompute_hidden_activation_masks(
        model=model,
        x_eval=x_eval,
        batch_size=batch_size,
        device=device,
    )
    return {**precomputed, "leaf_ids": odt_leaf_ids_for_x(x_eval, tree)}


def _resolve_hidden_neuron_id(
    neuron_id: int | tuple[int, int],
    *,
    layer_widths: list[int],
    layer_offsets: list[int],
    total_hidden_neurons: int,
) -> tuple[int, int, int]:
    """Resolve a neuron id to (layer_idx, local_idx, global_idx)."""
    if isinstance(neuron_id, tuple):
        if len(neuron_id) != 2:
            raise ValueError("Tuple neuron_id must be (layer_idx, local_neuron_idx).")
        layer_idx, local_idx = int(neuron_id[0]), int(neuron_id[1])
        if layer_idx < 0 or layer_idx >= len(layer_widths):
            raise ValueError(
                f"layer_idx must be in [0, {len(layer_widths) - 1}], got {layer_idx}."
            )
        if local_idx < 0 or local_idx >= layer_widths[layer_idx]:
            raise ValueError(
                f"local neuron index must be in [0, {layer_widths[layer_idx] - 1}] "
                f"for layer {layer_idx}, got {local_idx}."
            )
        global_idx = layer_offsets[layer_idx] + local_idx
        return layer_idx, local_idx, global_idx

    global_idx = int(neuron_id)
    if global_idx < 0 or global_idx >= total_hidden_neurons:
        raise ValueError(
            f"Global neuron id must be in [0, {total_hidden_neurons - 1}], got {global_idx}."
        )
    layer_idx = max(
        i
        for i, offset in enumerate(layer_offsets)
        if global_idx >= offset
    )
    local_idx = global_idx - layer_offsets[layer_idx]
    return layer_idx, local_idx, global_idx


def _precompute_hidden_activation_masks(
    *,
    model: ReLUMLP,
    x_eval: np.ndarray,
    batch_size: int = 8192,
    device: str = "cpu",
) -> dict[str, Any]:
    """Precompute boolean hidden-layer activation masks for ``x_eval``."""
    model = model.to(device)
    model.eval()

    x_t = torch.from_numpy(x_eval).float().to(device)
    hidden_masks_per_layer: list[list[torch.Tensor]] = [
        [] for _ in range(len(model.hidden_layers))
    ]

    with torch.no_grad():
        for start in range(0, x_t.shape[0], batch_size):
            h = x_t[start : start + batch_size]
            for layer_idx, layer in enumerate(model.hidden_layers):
                pre_act = layer(h)
                # Blinders must follow the model's own subgradient convention:
                # ``relu`` gates on ``> 0``, ``relu_one_at_zero`` on ``>= 0``.
                hidden_masks_per_layer[layer_idx].append(
                    model.activation.gate(pre_act).detach().cpu()
                )
                h = model.activation.apply(pre_act)

    activation_masks = [torch.cat(parts, dim=0).numpy() for parts in hidden_masks_per_layer]
    layer_widths = [layer.out_features for layer in model.hidden_layers]
    layer_offsets = np.cumsum([0, *layer_widths[:-1]]).tolist()
    total_hidden_neurons = int(sum(layer_widths))

    return {
        "activation_masks": activation_masks,
        "layer_widths": layer_widths,
        "layer_offsets": layer_offsets,
        "total_hidden_neurons": total_hidden_neurons,
    }


def active_point_mask_for_neuron(
    *,
    model: ReLUMLP,
    x_eval: np.ndarray,
    neuron_id: int | tuple[int, int],
    batch_size: int = 8192,
    device: str = "cpu",
    precomputed: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return boolean mask of points in ``x_eval`` that activate a hidden neuron.

    Args:
        model: Checkpoint-loaded ReLU model.
        x_eval: Evaluation inputs of shape ``(n, d)``.
        neuron_id: Either global hidden neuron id (0..total_hidden-1) or tuple
            ``(layer_idx, local_neuron_idx)``.
        precomputed: Optional cache from ``_precompute_hidden_activation_masks``
            for repeated queries.

    Returns:
        Boolean array of shape ``(n,)``; ``True`` means the input activates that neuron.
    """
    if precomputed is None:
        precomputed = _precompute_hidden_activation_masks(
            model=model,
            x_eval=x_eval,
            batch_size=batch_size,
            device=device,
        )

    layer_idx, local_idx, _ = _resolve_hidden_neuron_id(
        neuron_id,
        layer_widths=precomputed["layer_widths"],
        layer_offsets=precomputed["layer_offsets"],
        total_hidden_neurons=precomputed["total_hidden_neurons"],
    )
    return precomputed["activation_masks"][layer_idx][:, local_idx].astype(bool)


def activation_tensor_by_layer_neuron_data(
    *,
    model: ReLUMLP,
    x_eval: np.ndarray,
    batch_size: int = 8192,
    device: str = "cpu",
    precomputed: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return hidden activation tensor shaped ``(layers, neurons, num_data)``.

    The neuron axis uses the maximum hidden width across layers. For layers with
    fewer neurons, trailing entries are padded with ``False``.
    """
    if precomputed is None:
        precomputed = _precompute_hidden_activation_masks(
            model=model,
            x_eval=x_eval,
            batch_size=batch_size,
            device=device,
        )

    activation_masks: list[np.ndarray] = precomputed["activation_masks"]
    layer_widths: list[int] = precomputed["layer_widths"]
    num_layers = len(layer_widths)
    num_data = int(x_eval.shape[0])
    max_width = max(layer_widths) if layer_widths else 0

    tensor = np.zeros((num_layers, max_width, num_data), dtype=bool)
    for layer_idx, (width, mask) in enumerate(zip(layer_widths, activation_masks)):
        # mask shape: (num_data, width) -> tensor slice shape: (width, num_data)
        tensor[layer_idx, :width, :] = mask.T.astype(bool)
    return tensor


def count_active_points_for_neuron_leaf(
    *,
    model: ReLUMLP,
    neuron_id: int | tuple[int, int],
    leaf_node_id: int,
    x_eval: np.ndarray,
    tree: COBODTTree,
    batch_size: int = 8192,
    device: str = "cpu",
    precomputed: dict[str, Any] | None = None,
) -> dict[str, int | float]:
    """Count how many points in a given ODT leaf activate a given hidden neuron.

    Args:
        model: Checkpoint-loaded ReLU model.
        neuron_id: Either global hidden neuron id (0..total_hidden-1) or
            tuple ``(layer_idx, local_neuron_idx)``.
        leaf_node_id: ODT label leaf id in heap indexing (e.g. 31..62 for depth 5).
        x_eval: Input data used for counting.
        tree: COB-ODT tree used to assign leaves.
        precomputed: Optional cache from
            ``_precompute_hidden_activation_masks_and_leaf_ids`` to reuse in loops.

    Returns:
        Dictionary with layer/local/global neuron id, total points in leaf, active points,
        and active fraction.
    """
    if precomputed is None:
        precomputed = _precompute_hidden_activation_masks_and_leaf_ids(
            model=model,
            x_eval=x_eval,
            tree=tree,
            batch_size=batch_size,
            device=device,
        )

    layer_idx, local_idx, global_idx = _resolve_hidden_neuron_id(
        neuron_id,
        layer_widths=precomputed["layer_widths"],
        layer_offsets=precomputed["layer_offsets"],
        total_hidden_neurons=precomputed["total_hidden_neurons"],
    )

    leaf_ids = precomputed["leaf_ids"]
    leaf_mask = leaf_ids == int(leaf_node_id)
    total_leaf_points = int(leaf_mask.sum())
    if total_leaf_points == 0:
        active_points = 0
    else:
        active_points = int(precomputed["activation_masks"][layer_idx][leaf_mask, local_idx].sum())
    active_fraction = float(active_points / total_leaf_points) if total_leaf_points > 0 else 0.0

    return {
        "leaf_node_id": int(leaf_node_id),
        "layer": int(layer_idx),
        "local_neuron_id": int(local_idx),
        "global_neuron_id": int(global_idx),
        "total_points_in_leaf": total_leaf_points,
        "active_points_in_leaf": active_points,
        "active_fraction_in_leaf": active_fraction,
    }


def summarize_neuron_leaf_activity(
    *,
    model: ReLUMLP,
    x_eval: np.ndarray,
    tree: COBODTTree,
    batch_size: int = 8192,
    device: str = "cpu",
) -> pd.DataFrame:
    """Return activity summary for all hidden neurons across all reached ODT leaves."""
    precomputed = _precompute_hidden_activation_masks_and_leaf_ids(
        model=model,
        x_eval=x_eval,
        tree=tree,
        batch_size=batch_size,
        device=device,
    )
    leaf_ids = precomputed["leaf_ids"].astype(np.int64)
    unique_leaf_ids, leaf_inverse, leaf_totals = np.unique(
        leaf_ids, return_inverse=True, return_counts=True
    )
    layer_widths: list[int] = precomputed["layer_widths"]
    layer_offsets: list[int] = precomputed["layer_offsets"]
    activation_masks: list[np.ndarray] = precomputed["activation_masks"]

    rows: list[dict[str, int | float]] = []
    for layer_idx, (width, layer_mask) in enumerate(zip(layer_widths, activation_masks)):
        # Aggregate per-sample hidden activations into per-leaf neuron counts in one pass.
        # Result shape: (num_leaves, width)
        active_counts = np.zeros((len(unique_leaf_ids), width), dtype=np.int64)
        np.add.at(active_counts, leaf_inverse, layer_mask.astype(np.int64))

        for local_idx in range(width):
            global_idx = layer_offsets[layer_idx] + local_idx
            for leaf_pos, leaf_node_id in enumerate(unique_leaf_ids):
                total_leaf_points = int(leaf_totals[leaf_pos])
                active_points = int(active_counts[leaf_pos, local_idx])
                active_fraction = (
                    float(active_points / total_leaf_points) if total_leaf_points > 0 else 0.0
                )
                rows.append(
                    {
                        "leaf_node_id": int(leaf_node_id),
                        "layer": int(layer_idx),
                        "local_neuron_id": int(local_idx),
                        "global_neuron_id": int(global_idx),
                        "total_points_in_leaf": total_leaf_points,
                        "active_points_in_leaf": active_points,
                        "active_fraction_in_leaf": active_fraction,
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["layer", "local_neuron_id", "leaf_node_id"],
        ignore_index=True,
    )


def neuron_leaf_activity_dict(
    neuron_leaf_activity_summary: pd.DataFrame,
    *,
    key_mode: str = "global",
) -> tuple[dict[int | tuple[int, int], np.ndarray], np.ndarray]:
    """Convert neuron-leaf activity table into neuron -> leaf-activity vector mapping.

    Args:
        neuron_leaf_activity_summary: DataFrame produced by
            ``summarize_neuron_leaf_activity`` (or equivalent), expected to contain:
            - ``leaf_node_id``
            - ``active_points_in_leaf``
            - and either:
              - ``global_neuron_id`` (for ``key_mode='global'``), or
              - ``layer`` and ``local_neuron_id`` (for ``key_mode='layer_local'``).
        key_mode: Key format for the output dictionary:
            - ``'global'`` -> keys are global neuron ids (int).
            - ``'layer_local'`` -> keys are ``(layer, local_neuron_id)`` tuples.

    Returns:
        (mapping, leaf_order)
            mapping: dict where each key is a neuron id and each value is a numpy array
                of length ``len(leaf_order)``, with entries equal to
                ``active_points_in_leaf`` in that leaf.
            leaf_order: sorted numpy array of leaf node ids used as vector order.
    """
    df = neuron_leaf_activity_summary.copy()
    if df.empty:
        return {}, np.array([], dtype=np.int64)

    required_common = {"leaf_node_id", "active_points_in_leaf"}
    missing_common = required_common.difference(df.columns)
    if missing_common:
        raise ValueError(
            "neuron_leaf_activity_summary missing required columns: "
            f"{sorted(missing_common)}"
        )

    if key_mode not in {"global", "layer_local"}:
        raise ValueError(f"key_mode must be 'global' or 'layer_local', got {key_mode!r}.")

    if key_mode == "global":
        if "global_neuron_id" not in df.columns:
            raise ValueError(
                "For key_mode='global', neuron_leaf_activity_summary must include "
                "'global_neuron_id'."
            )
    else:
        required_pair = {"layer", "local_neuron_id"}
        missing_pair = required_pair.difference(df.columns)
        if missing_pair:
            raise ValueError(
                "For key_mode='layer_local', neuron_leaf_activity_summary missing columns: "
                f"{sorted(missing_pair)}"
            )

    leaf_order = np.sort(df["leaf_node_id"].astype(np.int64).unique())
    leaf_to_pos = {leaf_id: i for i, leaf_id in enumerate(leaf_order)}

    mapping: dict[int | tuple[int, int], np.ndarray] = {}

    if key_mode == "global":
        neuron_keys = sorted(df["global_neuron_id"].astype(int).unique().tolist())
        for neuron_id in neuron_keys:
            vec = np.zeros(len(leaf_order), dtype=np.int64)
            sub = df[df["global_neuron_id"].astype(int) == neuron_id]
            for row in sub.itertuples(index=False):
                pos = leaf_to_pos[int(row.leaf_node_id)]
                vec[pos] = int(row.active_points_in_leaf)
            mapping[int(neuron_id)] = vec
    else:
        subkeys = (
            df[["layer", "local_neuron_id"]]
            .astype(int)
            .drop_duplicates()
            .sort_values(["layer", "local_neuron_id"])
        )
        for rowkey in subkeys.itertuples(index=False):
            key = (int(rowkey.layer), int(rowkey.local_neuron_id))
            vec = np.zeros(len(leaf_order), dtype=np.int64)
            sub = df[
                (df["layer"].astype(int) == key[0])
                & (df["local_neuron_id"].astype(int) == key[1])
            ]
            for row in sub.itertuples(index=False):
                pos = leaf_to_pos[int(row.leaf_node_id)]
                vec[pos] = int(row.active_points_in_leaf)
            mapping[key] = vec

    return mapping, leaf_order


# ---------------------------------------------------------------------------
# Replayable training: optimizer-state snapshots and one-epoch replays.
#
# The functions below exist to separate the *optimizer* (Adam / mini-batches)
# from the *population gradient* in the learning-dynamics analysis. A
# "replayable" seed result stores, at each snapshot epoch, everything needed to
# reproduce the epoch that follows it bit-for-bit: the model state, the
# optimizer state (Adam moments and step counters, or SGD momentum buffers),
# the current learning rate, and the state of the generator that produces the
# mini-batch permutation. It does *not* store the data; the data is regenerated
# from its seed on demand (``regenerate_seed_result_data``).
# ---------------------------------------------------------------------------

def _as_optimizer_state_list(snapshot: Any) -> list[dict[str, Any]]:
    """Normalize a stored optimizer snapshot to a list of optimizer state dicts.

    Snapshots taken before the ``"sgd_adam"`` hybrid stored one state dict; hybrid
    runs store one per optimizer. Both shapes load.
    """
    if isinstance(snapshot, dict) and "param_groups" in snapshot:
        return [snapshot]
    if isinstance(snapshot, list):
        return list(snapshot)
    raise TypeError(f"Unrecognized optimizer snapshot of type {type(snapshot).__name__}.")


def _state_dict_to_cpu(state: Any) -> Any:
    """Recursively detach, clone, and move every tensor in a (nested) state dict to CPU."""
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().clone()
    if isinstance(state, dict):
        return {k: _state_dict_to_cpu(v) for k, v in state.items()}
    if isinstance(state, list):
        return [_state_dict_to_cpu(v) for v in state]
    if isinstance(state, tuple):
        return tuple(_state_dict_to_cpu(v) for v in state)
    return copy.deepcopy(state)


def _named_parameter_tensors(model: ReLUMLP) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}


def _named_parameter_gradients(model: ReLUMLP) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            raise RuntimeError(f"No gradient computed for parameter {name!r}.")
        grads[name] = p.grad.detach().cpu().clone()
    return grads


def _named_optimizer_state(
    model: ReLUMLP,
    optimizers: list[torch.optim.Optimizer],
) -> dict[str, dict[str, torch.Tensor | float | int]]:
    """Return optimizer state keyed by parameter *name*, with tensors on CPU.

    Searches every optimizer, so a hybrid run yields Adam moments for the AdamW
    parameters and momentum buffers for the SGD parameters in one flat dict.
    """
    out: dict[str, dict[str, torch.Tensor | float | int]] = {}
    for name, p in model.named_parameters():
        state: dict[str, Any] = {}
        for optimizer in optimizers:
            if p in optimizer.state:
                state = optimizer.state[p]
                break
        out[name] = {k: _state_dict_to_cpu(v) for k, v in state.items()}
    return out


def _epoch_permutation(
    *,
    num_samples: int,
    shuffle_each_epoch: bool,
    permutation_generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor:
    if not shuffle_each_epoch:
        return torch.arange(num_samples, device=device)
    if permutation_generator is None:
        raise ValueError("permutation_generator is required when shuffle_each_epoch=True.")
    return torch.randperm(num_samples, generator=permutation_generator).to(device)


def train_relu_mlp_with_optimizer_snapshots(
    *,
    model: ReLUMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: ReLUTrainConfig,
    shuffle_each_epoch: bool = False,
    start_epoch: int = 0,
    initial_optimizer_states: list[dict[str, Any]] | None = None,
    initial_rotation_lock: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Train like ``train_relu_mlp`` but also snapshot optimizer / LR / RNG state.

    ``start_epoch`` resumes the schedule at that epoch: the model is taken as the state at
    ``start_epoch``, the cosine schedulers are advanced to it, and only epochs
    ``start_epoch .. config.epochs - 1`` are run; ``epoch_losses`` / ``lr_history`` keep
    length ``config.epochs`` with NaN for the skipped epochs. ``initial_optimizer_states``
    (one state dict per optimizer, e.g. a stored snapshot) continues the optimizer exactly;
    without it the optimizer starts fresh (what an optimizer switch wants).

    At every epoch ``e`` in ``config.snapshot_epochs`` (taken *before* the
    epoch's mini-batch steps, exactly like ``checkpoint_snapshots``) this records:

    - ``checkpoint_snapshots[e]``: model ``state_dict`` (same as ``train_relu_mlp``),
    - ``optimizer_snapshots[e]``: ``optimizer.state_dict()`` on CPU (Adam moments
      and step counts, or SGD momentum buffers, plus the current ``lr``),
    - ``permutation_generator_snapshots[e]``: state of the CPU generator that
      draws the mini-batch permutation (only when ``shuffle_each_epoch``),
    - ``rotation_lock_snapshots[e]``: the first-layer rotation-lock state (which rows are
      locked, their locked directions, the step each locked at) when
      ``config.first_layer_rotation_lock`` is set; ``initial_rotation_lock`` restores such a
      snapshot when resuming. ``rotation_lock_final`` is the state at the end.

    With ``shuffle_each_epoch=False`` (the default, matching the current
    ``train_relu_mlp``) the permutation is the identity in every epoch.
    Together these make ``replay_epoch_from_snapshot`` reproduce epoch ``e``
    step-for-step. Snapshots are small (parameters + 2x parameters for Adam),
    so dense snapshot schedules remain cheap.
    """
    set_seed(config.seed)
    device = torch.device(config.device)
    model = model.to(device)
    first_layer_projection_vector = _first_layer_projection_vector(
        model=model, config=config, device=device
    )
    parameter_update_masks = _parameter_update_masks(model=model, config=config, device=device)

    x = torch.from_numpy(x_train).float().to(device)
    y_pm1 = torch.from_numpy(y_train).float().to(device)
    y_bin = _labels_pm1_to_binary(y_pm1)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizers = build_relu_optimizers(model, config)
    schedulers = build_relu_schedulers(optimizers, config)
    group_names = optimizer_group_names(config)
    primary_index = primary_optimizer_index(config)
    if not 0 <= start_epoch <= config.epochs:
        raise ValueError(f"start_epoch must be in [0, {config.epochs}], got {start_epoch}")
    steps_per_epoch = -(-x.shape[0] // config.batch_size)
    rotation_lock = rotation_lock_state(model=model, config=config, device=device, snapshot=initial_rotation_lock, step=start_epoch * steps_per_epoch)
    if initial_optimizer_states is not None:
        states = _as_optimizer_state_list(initial_optimizer_states)
        if len(states) != len(optimizers):
            raise ValueError(f"got {len(states)} initial optimizer state(s) for {len(optimizers)} optimizer(s)")
        for optimizer, st in zip(optimizers, states):
            optimizer.load_state_dict(copy.deepcopy(st))
        # the stored state carries the lr of its epoch; put the base lr back so that advancing
        # the (recursive) cosine scheduler below lands on the schedule's value at start_epoch
        for optimizer, scheduler in zip(optimizers, schedulers):
            for group, base in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = base; group["initial_lr"] = base
    for _ in range(start_epoch):          # advance the cosine schedules to start_epoch (sets the lr as well)
        for scheduler in schedulers:
            scheduler.step()

    permutation_generator: torch.Generator | None = None
    if shuffle_each_epoch:
        permutation_generator = torch.Generator(device="cpu")
        permutation_generator.manual_seed(config.seed)

    num_samples = x.shape[0]
    losses: list[float] = [float("nan")] * start_epoch
    lr_history: list[float] = [float("nan")] * start_epoch
    lr_history_by_group: dict[str, list[float]] = {name: [float("nan")] * start_epoch for name in group_names}
    checkpoint_snapshots: dict[int, dict[str, torch.Tensor]] = {}
    optimizer_snapshots: dict[int, list[dict[str, Any]]] = {}
    permutation_generator_snapshots: dict[int, torch.Tensor] = {}
    rotation_lock_snapshots: dict[int, dict[str, np.ndarray]] = {}

    epoch_iter = range(start_epoch, config.epochs + 1)
    if config.show_progress:
        from tqdm.auto import tqdm

        epoch_iter = tqdm(epoch_iter, desc="Training ReLU MLP (replayable)", leave=False)

    for epoch in epoch_iter:
        if epoch in config.snapshot_epochs:
            checkpoint_snapshots[epoch] = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            optimizer_snapshots[epoch] = [
                _state_dict_to_cpu(optimizer.state_dict()) for optimizer in optimizers
            ]
            if permutation_generator is not None:
                permutation_generator_snapshots[epoch] = permutation_generator.get_state().clone()
            if rotation_lock is not None:
                rotation_lock_snapshots[epoch] = rotation_lock.snapshot()

        if epoch == config.epochs:
            break

        permutation = _epoch_permutation(
            num_samples=num_samples,
            shuffle_each_epoch=shuffle_each_epoch,
            permutation_generator=permutation_generator,
            device=device,
        )

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
                rotation_lock=rotation_lock,
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
        "optimizer_snapshots": optimizer_snapshots,
        "optimizer_group_names": group_names,
        "optimizer_parameter_names": optimizer_parameter_names(model, config),
        "permutation_generator_snapshots": permutation_generator_snapshots,
        "shuffle_each_epoch": shuffle_each_epoch,
        "start_epoch": start_epoch,
        "rotation_lock_snapshots": rotation_lock_snapshots,
        "rotation_lock_final": rotation_lock.snapshot() if rotation_lock is not None else None,
    }


def _generate_cob_odt_split(
    *,
    dim: int,
    depth: int,
    n_train: int,
    data_seed: int,
    threshold: float,
    odt_hyperplanes: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, COBODTTree, dict[str, int | float]]:
    """Generate the train/eval split exactly as ``run_single_relu_seed`` does."""
    if odt_hyperplanes == "random_orthogonal":
        generation_tree = None
    elif odt_hyperplanes == "coordinate_axes":
        generation_tree = build_axis_aligned_cob_odt_tree(dim=dim, depth=depth)
    else:
        raise ValueError(
            "odt_hyperplanes must be 'random_orthogonal' or 'coordinate_axes', "
            f"got {odt_hyperplanes!r}."
        )
    x, y, tree, meta = generate_cob_odt_data(
        num_data=2 * n_train,
        dim=dim,
        depth=depth,
        seed=data_seed,
        threshold=threshold,
        tree=generation_tree,
    )
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:], tree, meta


def build_seed_model_and_data(
    master_seed: int,
    *,
    dim: int,
    depth: int,
    n_train: int,
    hidden_dims: list[int],
    bias: bool = False,
    activation: str = "relu",
    odt_hyperplanes: Literal["random_orthogonal", "coordinate_axes"] = "random_orthogonal",
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Build the data split and the model at initialization for a seed, without training.

    Uses the same seed convention as `run_single_relu_seed_replayable`
    (``data_seed = master_seed``, ``init_seed = master_seed + 1000``,
    ``train_seed = master_seed + 2000``) and is what that function calls, so the
    returned model and data are exactly the ones a run would start from. Useful for
    pre-run diagnostics such as `suggest_sgd_lr_for_adam_parity`.
    """
    data_seed = master_seed
    init_seed = master_seed + 1000
    train_seed = master_seed + 2000

    x_train, y_train, x_eval, y_eval, tree, meta = _generate_cob_odt_split(
        dim=dim,
        depth=depth,
        n_train=n_train,
        data_seed=data_seed,
        threshold=threshold,
        odt_hyperplanes=odt_hyperplanes,
    )
    set_seed(init_seed)
    model = ReLUMLP(
        input_dim=dim,
        hidden_dims=hidden_dims,
        bias=bias,
        activation=activation,
    )
    return {
        "model": model,
        "data": {"x_train": x_train, "y_train": y_train, "x_eval": x_eval, "y_eval": y_eval},
        "tree": tree,
        "meta": meta,
        "seeds": {"data_seed": data_seed, "init_seed": init_seed, "train_seed": train_seed},
    }


def run_single_relu_seed_replayable(
    master_seed: int,
    *,
    dim: int,
    depth: int,
    n_train: int,
    hidden_dims: list[int],
    total_epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    snapshot_epochs,
    bias: bool = False,
    activation: str = "relu",
    odt_hyperplanes: Literal["random_orthogonal", "coordinate_axes"] = "random_orthogonal",
    parameter_update_mask: dict[str, np.ndarray | torch.Tensor] | None = None,
    threshold: float = 0.0,
    device: str = "cpu",
    show_progress: bool = True,
    lr_scheduler_eta_min_ratio: float = 0.05,
    optimizer: str = "adamw",
    momentum: float = 0.0,
    sgd_adam_split_layer: int = 1,
    sgd_lr: float | None = None,
    sgd_weight_decay: float | None = None,
    project_first_layer_grad_orthogonal: int | None = None,
    shuffle_each_epoch: bool = False,
) -> dict[str, Any]:
    """Like ``run_single_relu_seed`` but stores optimizer snapshots and *no data arrays*.

    Seeds follow the same convention (``data_seed = master_seed``,
    ``init_seed = master_seed + 1000``, ``train_seed = master_seed + 2000``), so
    the data and initial model are identical to a ``run_single_relu_seed`` call
    with the same arguments.

    The returned dict has no ``"data"`` entry. Instead ``"data_spec"`` holds the
    arguments needed to regenerate the split; call
    ``regenerate_seed_result_data(seed_result)`` when the arrays are needed.
    ``"train_config"`` holds the ``ReLUTrainConfig`` used, and ``"out"`` has the
    extra keys documented in ``train_relu_mlp_with_optimizer_snapshots``.
    ``out["checkpoint_snapshots"]`` keeps the same layout as before, so the
    existing checkpoint-based analysis helpers work unchanged.
    """
    bundle = build_seed_model_and_data(
        master_seed,
        dim=dim,
        depth=depth,
        n_train=n_train,
        hidden_dims=hidden_dims,
        bias=bias,
        activation=activation,
        odt_hyperplanes=odt_hyperplanes,
        threshold=threshold,
    )
    data_seed = bundle["seeds"]["data_seed"]
    init_seed = bundle["seeds"]["init_seed"]
    train_seed = bundle["seeds"]["train_seed"]
    tree, meta = bundle["tree"], bundle["meta"]
    x_train = bundle["data"]["x_train"]
    y_train = bundle["data"]["y_train"]
    model = bundle["model"]

    if project_first_layer_grad_orthogonal is None:
        first_layer_projection_vector = None
    elif isinstance(project_first_layer_grad_orthogonal, bool) or not isinstance(
        project_first_layer_grad_orthogonal, int
    ):
        raise TypeError(
            "project_first_layer_grad_orthogonal must be an ODT internal node id "
            f"integer or None, got {project_first_layer_grad_orthogonal!r}."
        )
    elif not 0 <= project_first_layer_grad_orthogonal < tree.w_list.shape[0]:
        raise ValueError(
            "project_first_layer_grad_orthogonal must reference an ODT internal node. "
            f"Expected an integer in [0, {tree.w_list.shape[0] - 1}], "
            f"got {project_first_layer_grad_orthogonal}."
        )
    else:
        first_layer_projection_vector = tree.w_list[project_first_layer_grad_orthogonal]

    train_config = ReLUTrainConfig(
        epochs=total_epochs,
        lr=lr,
        batch_size=batch_size,
        seed=train_seed,
        snapshot_epochs=tuple(snapshot_epochs),
        weight_decay=weight_decay,
        device=device,
        show_progress=show_progress,
        lr_scheduler_eta_min_ratio=lr_scheduler_eta_min_ratio,
        optimizer=optimizer,
        momentum=momentum,
        sgd_adam_split_layer=sgd_adam_split_layer,
        sgd_lr=sgd_lr,
        sgd_weight_decay=sgd_weight_decay,
        first_layer_grad_orthogonal_to=first_layer_projection_vector,
        parameter_update_mask=parameter_update_mask,
    )
    out = train_relu_mlp_with_optimizer_snapshots(
        model=model,
        x_train=x_train,
        y_train=y_train,
        config=train_config,
        shuffle_each_epoch=shuffle_each_epoch,
    )

    return {
        "master_seed": master_seed,
        "seeds": {"data_seed": data_seed, "init_seed": init_seed, "train_seed": train_seed},
        "tree": tree,
        "meta": meta,
        "data_spec": {
            "dim": dim,
            "depth": depth,
            "n_train": n_train,
            "data_seed": data_seed,
            "threshold": threshold,
            "odt_hyperplanes": odt_hyperplanes,
        },
        "model_spec": {
            "input_dim": dim,
            "hidden_dims": list(hidden_dims),
            "bias": bias,
            "activation": activation,
        },
        "train_config": train_config,
        "out": out,
    }


def regenerate_seed_result_data(seed_result: dict[str, Any]) -> dict[str, np.ndarray]:
    """Regenerate ``x_train, y_train, x_eval, y_eval`` for a replayable seed result.

    Uses ``seed_result["data_spec"]`` and checks that the regenerated ODT matches
    the stored ``seed_result["tree"]``. If the result still carries a ``"data"``
    entry (legacy ``run_single_relu_seed`` output), that is returned directly.
    """
    if "data" in seed_result and seed_result["data"] is not None:
        return dict(seed_result["data"])
    if "data_spec" not in seed_result:
        raise KeyError("seed_result has neither 'data' nor 'data_spec'.")

    spec = seed_result["data_spec"]
    if spec.get("kind") == "sweep_chunked":
        # Designed-init runs: tree from `init_sweep.build_tree`, training data from
        # `build_data_chunked`, evaluation data from `build_data` with its own seed.
        from layer_lenses.init_sweep import SweepRunSpec, build_data, build_data_chunked, build_tree

        base = dict(tree_seed=int(spec["tree_seed"]), init_seed=0, batch_size=1, dim=int(spec["dim"]), depth=int(spec["depth"]))
        tree = build_tree(SweepRunSpec(data_seed=0, **base))
        x_train, y_train = build_data_chunked(SweepRunSpec(data_seed=int(spec["data_seed"]), n_train=int(spec["n_train"]), **base), tree, chunk_size=int(spec.get("chunk_size", 1_000_000)))
        x_eval, y_eval = build_data(SweepRunSpec(data_seed=int(spec["eval_seed"]), n_train=int(spec["n_eval"]), **base), tree)
        stored_tree = seed_result.get("tree")
        if stored_tree is not None and not np.allclose(stored_tree.w_list, tree.w_list):
            raise RuntimeError("Regenerated ODT does not match the stored tree; data_spec is inconsistent.")
        return {"x_train": x_train, "y_train": y_train, "x_eval": x_eval, "y_eval": y_eval}
    x_train, y_train, x_eval, y_eval, tree, _ = _generate_cob_odt_split(
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        n_train=int(spec["n_train"]),
        data_seed=int(spec["data_seed"]),
        threshold=float(spec["threshold"]),
        odt_hyperplanes=str(spec["odt_hyperplanes"]),
    )
    stored_tree = seed_result.get("tree")
    if stored_tree is not None and not (
        np.allclose(stored_tree.w_list, tree.w_list)
        and np.allclose(stored_tree.b_list, tree.b_list)
        and np.array_equal(stored_tree.leaf_labels, tree.leaf_labels)
    ):
        raise RuntimeError(
            "Regenerated ODT does not match the stored tree; data_spec is inconsistent."
        )
    return {"x_train": x_train, "y_train": y_train, "x_eval": x_eval, "y_eval": y_eval}


def save_seed_result_pickle(seed_result: dict[str, Any], path: str | Path) -> Path:
    """Pickle a seed result (creating parent directories)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(seed_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def load_seed_result_pickle(path: str | Path) -> dict[str, Any]:
    """Load a pickled seed result."""
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict from pickle, got {type(obj).__name__}")
    return obj


def _adam_step_decomposition(
    *,
    optimizer_state_after: dict[str, dict[str, Any]],
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    params_before: dict[str, torch.Tensor],
    parameter_names: set[str] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Decompose one AdamW update into per-coordinate step size, direction, and decay.

    torch's AdamW performs::

        theta <- theta * (1 - lr * wd)
        theta <- theta - (lr / bc1) * m / (sqrt(v) / sqrt(bc2) + eps)

    with ``bc1 = 1 - beta1**t``, ``bc2 = 1 - beta2**t`` and ``m, v`` the moments
    *after* incorporating the current gradient. This returns, per parameter:

    - ``adam_step_size``: ``lr / (sqrt(v) / sqrt(bc2) + eps)`` (per coordinate),
    - ``adam_direction``: ``m / bc1`` (bias-corrected first moment),
    - ``adam_update``: ``-adam_step_size * adam_direction``,
    - ``weight_decay_update``: ``-lr * wd * theta_before``,

    so that ``adam_update + weight_decay_update`` equals the recorded ``update``
    up to floating point.
    """
    beta1, beta2 = betas
    out: dict[str, dict[str, torch.Tensor]] = {
        "adam_step_size": {},
        "adam_direction": {},
        "adam_update": {},
        "weight_decay_update": {},
    }
    for name, st in optimizer_state_after.items():
        # Skip empty state and SGD-managed parameters (which carry momentum_buffer,
        # not exp_avg) so a hybrid run decomposes only its AdamW part.
        if not st or "exp_avg" not in st:
            continue
        if parameter_names is not None and name not in parameter_names:
            continue
        step = st["step"]
        t = float(step.item()) if isinstance(step, torch.Tensor) else float(step)
        bc1 = 1.0 - beta1**t
        bc2 = 1.0 - beta2**t
        m = st["exp_avg"]
        v = st["exp_avg_sq"]
        denom = v.sqrt() / (bc2**0.5) + eps
        step_size = lr / denom
        direction = m / bc1
        out["adam_step_size"][name] = step_size
        out["adam_direction"][name] = direction
        out["adam_update"][name] = -step_size * direction
        out["weight_decay_update"][name] = -lr * weight_decay * params_before[name]
    return out


def replay_epoch_from_snapshot(
    seed_result: dict[str, Any],
    epoch: int,
    *,
    data: dict[str, np.ndarray] | None = None,
    device: str | None = None,
    record_optimizer_state: bool = True,
    record_full_data_gradient: bool = False,
    verify_against_next_snapshot: bool = True,
) -> dict[str, Any]:
    """Re-run one training epoch from a snapshot and record every mini-batch step.

    ``epoch`` must be in ``out["optimizer_snapshots"]`` of a result produced by
    ``run_single_relu_seed_replayable``. The model and optimizer are restored to
    their state at the *start* of that epoch, the same mini-batch order is used,
    and each step is recorded. The seed result itself is not modified.

    Returns a dict with:

    - ``"epoch"``, ``"lr"`` (the LR in force during this epoch), ``"optimizer"``,
      ``"betas"``, ``"eps"``, ``"weight_decay"``, ``"num_steps"``,
    - ``"params_before"`` / ``"params_after"``: parameter dicts at epoch start / end,
    - ``"steps"``: a list, one entry per mini-batch, each with
        ``"step"``, ``"batch_start"``, ``"batch_end"``, ``"batch_loss"``,
        ``"grad"`` (the mini-batch gradient the optimizer consumed, after any
        projection / mask), ``"update"`` (``theta_after - theta_before`` for the
        step), ``"params_before"``, and, when ``record_optimizer_state``,
        ``"optimizer_state_after"`` keyed by parameter name (Adam: ``step``,
        ``exp_avg``, ``exp_avg_sq``; SGD with momentum: ``momentum_buffer``).
        For Adam the decomposition from ``_adam_step_decomposition`` is added
        (``adam_step_size``, ``adam_direction``, ``adam_update``,
        ``weight_decay_update``). When ``record_full_data_gradient`` the
        *population* gradient over the whole training set at the step's starting
        parameters is stored under ``"full_grad"``.
    - ``"verification"``: if ``epoch + 1`` is also a snapshot and
      ``verify_against_next_snapshot``, the max abs difference between the
      replayed end-of-epoch parameters (and Adam moments) and that stored snapshot.
      Zero (or ~1e-7) means the replay is faithful.

    All recorded tensors are on CPU. Memory per step is roughly
    ``(2 + 2 * is_adam + 4 * is_adam + record_full_data_gradient) * n_params`` floats.
    """
    out = seed_result["out"]
    if "optimizer_snapshots" not in out:
        raise KeyError(
            "seed_result has no optimizer snapshots; produce it with "
            "run_single_relu_seed_replayable."
        )
    if epoch not in out["optimizer_snapshots"]:
        available = sorted(out["optimizer_snapshots"].keys())
        raise KeyError(f"epoch {epoch} not in optimizer snapshots; available: {available}")
    config: ReLUTrainConfig = seed_result["train_config"]
    if epoch >= config.epochs:
        raise ValueError(f"epoch {epoch} is the final snapshot; no epoch follows it to replay.")

    dev = torch.device(device if device is not None else config.device)
    if data is None:
        data = regenerate_seed_result_data(seed_result)
    x = torch.from_numpy(data["x_train"]).float().to(dev)
    y_bin = _labels_pm1_to_binary(torch.from_numpy(data["y_train"]).float().to(dev))
    num_samples = x.shape[0]

    model = checkpoint_model_from_state(out["model"], out["checkpoint_snapshots"][epoch]).to(dev)
    model.train()
    optimizers = build_relu_optimizers(model, config)
    group_names = optimizer_group_names(config)
    stored_snapshots = _as_optimizer_state_list(out["optimizer_snapshots"][epoch])
    if len(stored_snapshots) != len(optimizers):
        raise ValueError(
            f"snapshot at epoch {epoch} holds {len(stored_snapshots)} optimizer state(s) "
            f"but the config builds {len(optimizers)} ({group_names}); the stored "
            "train_config and the snapshot disagree."
        )
    for optimizer, stored in zip(optimizers, stored_snapshots):
        optimizer.load_state_dict(copy.deepcopy(stored))

    parameter_groups = {
        name: list(names)
        for name, names in zip(group_names, optimizer_parameter_names(model, config))
    }
    lr_by_group = {
        name: float(optimizer.param_groups[0]["lr"])
        for name, optimizer in zip(group_names, optimizers)
    }
    adam_index = group_names.index("adamw") if "adamw" in group_names else None
    adam_group = optimizers[adam_index].param_groups[0] if adam_index is not None else None
    adam_parameter_names = (
        set(parameter_groups[group_names[adam_index]]) if adam_index is not None else set()
    )
    lr = lr_by_group[group_names[primary_optimizer_index(config)]]

    first_layer_projection_vector = _first_layer_projection_vector(model=model, config=config, device=dev)
    parameter_update_masks = _parameter_update_masks(model=model, config=config, device=dev)
    criterion = torch.nn.BCEWithLogitsLoss()

    permutation_generator: torch.Generator | None = None
    shuffle_each_epoch = bool(out.get("shuffle_each_epoch", False))
    if shuffle_each_epoch:
        permutation_generator = torch.Generator(device="cpu")
        permutation_generator.set_state(out["permutation_generator_snapshots"][epoch].clone())
    permutation = _epoch_permutation(
        num_samples=num_samples,
        shuffle_each_epoch=shuffle_each_epoch,
        permutation_generator=permutation_generator,
        device=dev,
    )

    params_before_epoch = _named_parameter_tensors(model)
    steps: list[dict[str, Any]] = []
    for step_idx, start in enumerate(range(0, num_samples, config.batch_size)):
        end = min(start + config.batch_size, num_samples)
        idx = permutation[start:end]
        params_before = _named_parameter_tensors(model)

        record: dict[str, Any] = {
            "step": step_idx,
            "batch_start": int(start),
            "batch_end": int(end),
            "params_before": params_before,
        }
        if record_full_data_gradient:
            record["full_grad"] = log_loss_gradients(
                model=model, x=data["x_train"], y=data["y_train"], device=str(dev)
            )
            model.train()

        loss = relu_training_step(
            model=model,
            optimizers=optimizers,
            criterion=criterion,
            x_batch=x[idx],
            y_batch_bin=y_bin[idx],
            first_layer_projection_vector=first_layer_projection_vector,
            parameter_update_masks=parameter_update_masks,
        )
        record["batch_loss"] = float(loss.cpu())
        record["grad"] = _named_parameter_gradients(model)
        params_after = _named_parameter_tensors(model)
        record["update"] = {n: params_after[n] - params_before[n] for n in params_after}

        if record_optimizer_state:
            state_after = _named_optimizer_state(model, optimizers)
            record["optimizer_state_after"] = state_after
            if adam_group is not None:
                record.update(
                    _adam_step_decomposition(
                        optimizer_state_after=state_after,
                        lr=float(adam_group["lr"]),
                        betas=tuple(adam_group["betas"]),
                        eps=float(adam_group["eps"]),
                        weight_decay=float(adam_group["weight_decay"]),
                        params_before=params_before,
                        parameter_names=adam_parameter_names,
                    )
                )
        steps.append(record)

    params_after_epoch = _named_parameter_tensors(model)

    verification: dict[str, Any] | None = None
    next_epoch = epoch + 1
    if verify_against_next_snapshot and next_epoch in out["checkpoint_snapshots"]:
        next_state = out["checkpoint_snapshots"][next_epoch]
        param_diff = {
            n: float((params_after_epoch[n] - next_state[n].float()).abs().max())
            for n in params_after_epoch
        }
        verification = {
            "next_epoch": next_epoch,
            "max_abs_param_diff": param_diff,
            "max_abs_param_diff_overall": max(param_diff.values()),
        }
        if next_epoch in out["optimizer_snapshots"]:
            moment_diff = 0.0
            next_snapshots = _as_optimizer_state_list(out["optimizer_snapshots"][next_epoch])
            for optimizer, stored in zip(optimizers, next_snapshots):
                replayed = _state_dict_to_cpu(optimizer.state_dict())["state"]
                for k, st in stored["state"].items():
                    for key, val in st.items():
                        if isinstance(val, torch.Tensor) and val.dtype.is_floating_point:
                            moment_diff = max(
                                moment_diff,
                                float((replayed[k][key].float() - val.float()).abs().max()),
                            )
            verification["max_abs_optimizer_state_diff"] = moment_diff

    return {
        "epoch": epoch,
        "lr": lr,
        "lr_by_group": lr_by_group,
        "optimizer": config.optimizer,
        "optimizer_group_names": group_names,
        "parameter_groups": parameter_groups,
        "betas": tuple(adam_group["betas"]) if adam_group is not None else None,
        "eps": float(adam_group["eps"]) if adam_group is not None else None,
        "weight_decay": float(adam_group["weight_decay"])
        if adam_group is not None
        else float(optimizers[0].param_groups[0]["weight_decay"]),
        "momentum": float(optimizers[0].param_groups[0].get("momentum", 0.0))
        if "sgd" in group_names
        else None,
        "shuffle_each_epoch": shuffle_each_epoch,
        "num_steps": len(steps),
        "params_before": params_before_epoch,
        "params_after": params_after_epoch,
        "steps": steps,
        "verification": verification,
    }


def flatten_named_tensors(
    named: dict[str, torch.Tensor],
    names: list[str] | None = None,
) -> torch.Tensor:
    """Concatenate a name-keyed tensor dict into one 1D float64 CPU vector.

    ``names`` fixes the order (default: sorted keys) so vectors from different
    dicts are comparable coordinate by coordinate.
    """
    keys = sorted(named.keys()) if names is None else names
    return torch.cat([named[k].detach().cpu().double().reshape(-1) for k in keys])


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm() * b.norm())
    return float(a @ b) / denom if denom > 0 else float("nan")


def epoch_replay_summary_table(
    replay: dict[str, Any],
    *,
    parameter_names: list[str] | None = None,
) -> pd.DataFrame:
    """One row per mini-batch step of a replay, with norms and alignment cosines.

    ``parameter_names`` restricts the vectors to a subset of parameters (e.g.
    ``["hidden_layers.0.weight"]``); default is all parameters.

    Columns: ``step``, ``batch_loss``, ``lr``, ``grad_norm``, ``update_norm``,
    ``cos_update_grad`` (update vs the mini-batch gradient that produced it; for
    plain GD this is exactly -1), ``cos_update_epoch_delta`` (this step's update
    vs the whole epoch's parameter change), ``cos_grad_epoch_delta``,
    ``cum_update_norm`` (norm of the running sum of updates) and, when the replay
    recorded population gradients, ``full_grad_norm``, ``cos_grad_full_grad``,
    ``cos_update_full_grad``. For a hybrid ``"sgd_adam"`` replay the Adam columns
    below cover only the AdamW-managed parameters; pass ``parameter_names`` (e.g.
    ``replay["parameter_groups"]["sgd"]``) to restrict every column to one group.
    For Adam replays: ``adam_step_size_min / median /
    max`` (per-coordinate effective learning rate; coordinates with a zero
    second moment, e.g. dead ReLU rows, give ``lr / eps`` and dominate the max) and
    ``weight_decay_fraction`` (``||wd update|| / ||update||``).
    """
    steps = replay["steps"]
    if not steps:
        return pd.DataFrame()
    names = parameter_names if parameter_names is not None else sorted(steps[0]["grad"].keys())
    epoch_delta = flatten_named_tensors(replay["params_after"], names) - flatten_named_tensors(
        replay["params_before"], names
    )
    cumulative = torch.zeros_like(epoch_delta)
    rows: list[dict[str, float | int]] = []
    for rec in steps:
        g = flatten_named_tensors(rec["grad"], names)
        u = flatten_named_tensors(rec["update"], names)
        cumulative = cumulative + u
        row: dict[str, float | int] = {
            "step": int(rec["step"]),
            "batch_loss": float(rec["batch_loss"]),
            "lr": float(replay["lr"]),
            "grad_norm": float(g.norm()),
            "update_norm": float(u.norm()),
            "cos_update_grad": _cosine(u, g),
            "cos_update_epoch_delta": _cosine(u, epoch_delta),
            "cos_grad_epoch_delta": _cosine(g, epoch_delta),
            "cum_update_norm": float(cumulative.norm()),
        }
        if "full_grad" in rec:
            fg = flatten_named_tensors(rec["full_grad"], names)
            row["full_grad_norm"] = float(fg.norm())
            row["cos_grad_full_grad"] = _cosine(g, fg)
            row["cos_update_full_grad"] = _cosine(u, fg)
        adam_names = [n for n in names if n in rec.get("adam_step_size", {})]
        if adam_names:
            ss = flatten_named_tensors(rec["adam_step_size"], adam_names)
            wd = flatten_named_tensors(rec["weight_decay_update"], adam_names)
            u_adam = flatten_named_tensors(rec["update"], adam_names)
            row["adam_step_size_min"] = float(ss.min())
            row["adam_step_size_median"] = float(ss.median())
            row["adam_step_size_max"] = float(ss.max())
            row["weight_decay_fraction"] = (
                float(wd.norm() / u_adam.norm()) if float(u_adam.norm()) > 0 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def epoch_replay_parameter_series(
    replay: dict[str, Any],
    key: str,
    parameter_name: str,
) -> torch.Tensor:
    """Stack one recorded quantity for one parameter across steps.

    ``key`` is a per-step entry such as ``"grad"``, ``"update"``,
    ``"adam_step_size"``, ``"adam_direction"``, ``"full_grad"`` or
    ``"params_before"``. Returns a tensor of shape ``(num_steps, *param_shape)``.
    """
    return torch.stack([rec[key][parameter_name] for rec in replay["steps"]])


# ---------------------------------------------------------------------------
# Checkpoint trajectories stored as weight arrays (e.g. the ``runs/*.npz``
# files written by the designed-init sweep scripts, keys ``w__<param name>``
# stacked over snapshot epochs).
# ---------------------------------------------------------------------------


def state_dict_from_weight_arrays(
    hidden_weights: list[np.ndarray],
    output_weight: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Assemble a bias-free ``ReLUMLP`` state dict from plain weight arrays.

    ``hidden_weights[i]`` becomes ``hidden_layers.i.weight`` and
    ``output_weight`` becomes ``output_layer.weight``, so the result can be fed
    to ``plot_layered_relu_graph`` or ``first_layer_odt_alignment`` directly.
    """
    state = {
        f"hidden_layers.{i}.weight": torch.as_tensor(np.asarray(w), dtype=torch.float32)
        for i, w in enumerate(hidden_weights)
    }
    out = torch.as_tensor(np.asarray(output_weight), dtype=torch.float32)
    state["output_layer.weight"] = out if out.ndim == 2 else out.reshape(1, -1)
    return state


def snapshot_state_dict(run: Any, epoch: int) -> dict[str, torch.Tensor]:
    """State dict at one snapshot of a stored trajectory.

    ``run`` is either a mapping (``np.load`` result or dict) with an ``epochs`` array and
    one ``w__<param name>`` array per parameter, or a dict with ``epochs`` and a ``W``
    sub-dict mapping parameter names to the same stacked arrays (the form returned by the
    notebooks' ``load_hier_run``). Arrays are stacked along axis 0 in the order of
    ``epochs``; ``epoch`` must be one of the stored epochs.
    """
    epochs = np.asarray(run["epochs"])
    hits = np.nonzero(epochs == epoch)[0]
    if hits.size == 0:
        raise ValueError(f"Epoch {epoch} is not a stored snapshot; stored epochs: {epochs.tolist()}")
    idx = int(hits[0])
    if "W" in run.keys() and isinstance(run["W"], dict):
        weights = {name: arr for name, arr in run["W"].items()}
    else:
        weights = {k[3:]: run[k] for k in run.keys() if k.startswith("w__")}
    hidden = sorted(
        (int(k.split(".")[1]), k) for k in weights if k.startswith("hidden_layers.") and k.endswith(".weight")
    )
    if not hidden or "output_layer.weight" not in weights:
        raise KeyError("run must hold 'w__hidden_layers.<i>.weight' / 'w__output_layer.weight' arrays, or a 'W' dict with those parameter names")
    return state_dict_from_weight_arrays(
        [weights[k][idx] for _, k in hidden], weights["output_layer.weight"][idx]
    )


def first_layer_alignment_series(
    first_layer_weights: np.ndarray,
    epochs: np.ndarray | list[int],
    tree: COBODTTree,
    *,
    neuron_ids: list[int],
    node_ids: list[int],
    fan_out_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Cosines of chosen first-layer neurons to chosen ODT nodes along a trajectory.

    ``first_layer_weights`` has shape (n_snapshots, width, dim) and ``epochs``
    labels its first axis. Returns one row per (epoch, neuron) with columns
    ``norm``, ``cos_<node>`` for each requested node, and ``fan_out_norm``
    (the norm of the neuron's outgoing column of ``fan_out_weights``, shape
    (n_snapshots, next_width, width)) when that array is given.
    """
    W = np.asarray(first_layer_weights, dtype=np.float64)
    epochs = np.asarray(epochs)
    if W.ndim != 3 or W.shape[0] != epochs.shape[0]:
        raise ValueError(f"first_layer_weights must be (n_snapshots, width, dim) matching epochs; got {W.shape} and {epochs.shape}")
    width, n_nodes = W.shape[1], tree.w_list.shape[0]
    for i in neuron_ids:
        if not 0 <= i < width:
            raise ValueError(f"neuron id {i} outside [0, {width - 1}]")
    for n in node_ids:
        if not 0 <= n < n_nodes:
            raise ValueError(f"ODT node id {n} outside [0, {n_nodes - 1}]")
    U = np.asarray(tree.w_list, dtype=np.float64)[node_ids]
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    rows = []
    for t, e in enumerate(epochs):
        for i in neuron_ids:
            w = W[t, i]
            nrm = float(np.linalg.norm(w))
            rec = {"epoch": int(e), "neuron": int(i), "norm": nrm}
            cos = U @ w / max(nrm, 1e-12)
            rec.update({f"cos_{n}": float(c) for n, c in zip(node_ids, cos)})
            if fan_out_weights is not None:
                rec["fan_out_norm"] = float(np.linalg.norm(np.asarray(fan_out_weights)[t][:, i]))
            rows.append(rec)
    return pd.DataFrame(rows)


def plot_first_layer_alignment_series(
    series: pd.DataFrame,
    *,
    neuron_ids: list[int] | None = None,
    node_ids: list[int] | None = None,
    show_norm: bool = True,
    show_fan_out: bool = True,
    norm_axis: bool = True,
    ylim: tuple[float, float] | None = None,
    figsize_per_row: float = 3.2,
    title: str | None = None,
):
    """One panel per neuron: cosine to each requested node over epochs.

    Cosines are drawn on the left axis in [-1, 1]. When ``norm_axis`` is true
    the neuron norm and fan-out norm go on a second (right) axis so they do not
    squash the cosines; otherwise everything shares one axis.
    """
    if neuron_ids is None:
        neuron_ids = sorted(series.neuron.unique().tolist())
    if node_ids is None:
        node_ids = [int(c[4:]) for c in series.columns if c.startswith("cos_")]
    fig, axes = plt.subplots(len(neuron_ids), 1, figsize=(9, figsize_per_row * len(neuron_ids)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, i in zip(axes, neuron_ids):
        g = series[series.neuron == i].sort_values("epoch")
        for k, n in enumerate(node_ids):
            ax.plot(g.epoch, g[f"cos_{n}"], color=f"C{k % 10}", lw=1.8, label=f"cos to node {n}")
        for yv in (-1.0, 0.0, 1.0):
            ax.axhline(yv, color="gray", ls="--", lw=0.8)
        ax.set_ylabel("cosine")
        ax.set_ylim(*(ylim if ylim is not None else (-1.05, 1.05)))
        ax.grid(alpha=0.25)
        ax2 = ax.twinx() if norm_axis else ax
        if show_norm:
            ax2.plot(g.epoch, g.norm, color="k", lw=1.4, ls="-", label="neuron norm")
        if show_fan_out and "fan_out_norm" in g:
            ax2.plot(g.epoch, g.fan_out_norm, color="k", lw=1.4, ls=":", label="fan-out norm")
        if norm_axis:
            ax2.set_ylabel("norm")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = (ax2.get_legend_handles_labels() if norm_axis else ([], []))
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", ncol=2)
        ax.set_title(f"first-layer neuron {i}", fontsize=9)
    axes[-1].set_xlabel("epoch")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig, axes


def population_gradient_at_snapshot(
    run: Any,
    epoch: int,
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden_dims: list[int] | None = None,
    device: str = "cpu",
    batch_size: int = 1_000_000,
) -> dict[str, np.ndarray]:
    """Gradient of the mean log loss over all of ``(x, y)`` at one stored snapshot.

    ``run`` is a ``w__``-keyed trajectory (see `snapshot_state_dict`). The model is rebuilt
    from the snapshot (bias-free ``ReLUMLP``; ``hidden_dims`` is inferred from the weights
    unless given) and the gradient is accumulated over batches of ``batch_size`` rows, so
    the whole set never has to sit on the device at once. Returns numpy arrays keyed like
    the state dict. With ``x`` the full training set this is the population (full-batch)
    gradient the optimiser would see at that checkpoint.
    """
    state = snapshot_state_dict(run, epoch)
    if hidden_dims is None:
        hidden_dims = [int(state[f"hidden_layers.{i}.weight"].shape[0]) for i in range(len(state) - 1)]
    model = ReLUMLP(input_dim=int(state["hidden_layers.0.weight"].shape[1]), hidden_dims=hidden_dims, bias=False)
    model.load_state_dict(state)
    n = x.shape[0]
    total: dict[str, np.ndarray] | None = None
    for start in range(0, n, batch_size):
        xb, yb = x[start : start + batch_size], y[start : start + batch_size]
        g = log_loss_gradients(model=model, x=np.ascontiguousarray(xb), y=np.ascontiguousarray(yb), device=device)
        w = xb.shape[0] / n
        if total is None:
            total = {k: v.numpy().astype(np.float64) * w for k, v in g.items()}
        else:
            for k, v in g.items():
                total[k] += v.numpy().astype(np.float64) * w
    assert total is not None
    return total


def first_layer_gradient_series(
    grads_by_epoch: dict[int, dict[str, np.ndarray] | np.ndarray],
    tree: COBODTTree,
    *,
    neuron_ids: list[int],
    node_ids: list[int],
    push: bool = True,
    normalize: bool = False,
) -> pd.DataFrame:
    """Components of the first-layer population gradient of chosen rows along chosen ODT nodes, per epoch.

    ``grads_by_epoch`` maps an epoch to either the full gradient dict (as returned by
    `population_gradient_at_snapshot`) or the first-layer gradient array ``(width, dim)``.
    With ``push`` (default) the series describes ``-grad``, the direction the row moves in;
    otherwise the raw gradient. Columns per (epoch, neuron): ``grad_norm``, ``comp_<node>``
    (signed component along the unit ODT normal), ``offspan_norm`` (norm of the projection
    onto the orthogonal complement of all ODT normals), and ``inspan_norm``. With
    ``normalize`` every component and norm is divided by ``grad_norm`` (components become
    cosines, the off-span norm the off-span fraction's square root).
    """
    U = np.asarray(tree.w_list, dtype=np.float64)
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-300)
    n_nodes = U.shape[0]
    for n in node_ids:
        if not 0 <= n < n_nodes:
            raise ValueError(f"ODT node id {n} outside [0, {n_nodes - 1}]")
    rows = []
    for e in sorted(grads_by_epoch):
        g_all = grads_by_epoch[e]
        G = np.asarray(g_all["hidden_layers.0.weight"] if isinstance(g_all, dict) else g_all, dtype=np.float64)
        if push:
            G = -G
        for i in neuron_ids:
            if not 0 <= i < G.shape[0]:
                raise ValueError(f"neuron id {i} outside [0, {G.shape[0] - 1}]")
            g = G[i]; gn = float(np.linalg.norm(g)); c = U @ g
            off = g - U.T @ c
            scale = max(gn, 1e-300) if normalize else 1.0
            rec = {"epoch": int(e), "neuron": int(i), "grad_norm": gn / scale}
            rec.update({f"comp_{n}": float(c[n]) / scale for n in node_ids})
            rec["inspan_norm"] = float(np.linalg.norm(c)) / scale
            rec["offspan_norm"] = float(np.linalg.norm(off)) / scale
            rows.append(rec)
    return pd.DataFrame(rows)


def plot_first_layer_gradient_series(
    series: pd.DataFrame,
    *,
    neuron_ids: list[int] | None = None,
    node_ids: list[int] | None = None,
    show_total_norm: bool = True,
    figsize_per_row: float = 3.2,
    title: str | None = None,
    ylabel: str = "component of push (-grad)",
):
    """One panel per neuron: gradient components along the chosen nodes, the off-span norm and the total norm vs epoch."""
    if neuron_ids is None:
        neuron_ids = sorted(series.neuron.unique().tolist())
    if node_ids is None:
        node_ids = [int(c[5:]) for c in series.columns if c.startswith("comp_")]
    fig, axes = plt.subplots(len(neuron_ids), 1, figsize=(9, figsize_per_row * len(neuron_ids)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, i in zip(axes, neuron_ids):
        g = series[series.neuron == i].sort_values("epoch")
        for k, n in enumerate(node_ids):
            ax.plot(g.epoch, g[f"comp_{n}"], color=f"C{k % 10}", lw=1.8, marker="o", ms=2.5, label=f"along u_{n}")
        ax.plot(g.epoch, g.offspan_norm, color="k", lw=1.6, ls="--", marker="o", ms=2.5, label="off-span norm")
        if show_total_norm:
            ax.plot(g.epoch, g.grad_norm, color="gray", lw=1.2, ls=":", label="total norm")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_ylabel(ylabel); ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.set_title(f"first-layer neuron {i}", fontsize=9)
    axes[-1].set_xlabel("epoch")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Replayable designed-init runs: compact on-disk format and multi-epoch replay.
# ---------------------------------------------------------------------------


def designed_run_seed_result(
    *,
    out: dict[str, Any],
    train_config: ReLUTrainConfig,
    model_spec: dict[str, Any],
    data_spec: dict[str, Any],
    tree: COBODTTree,
    design: dict[str, Any] | None = None,
    tag: str = "",
) -> dict[str, Any]:
    """Wrap the output of `train_relu_mlp_with_optimizer_snapshots` as a replayable result.

    The dict has the keys `replay_epoch_from_snapshot` / `replay_interval_from_snapshot`
    read (``out``, ``train_config``, ``model_spec``, ``data_spec``, ``tree``) plus the
    design description and a tag. ``data_spec`` must be understood by
    `regenerate_seed_result_data` (``kind="sweep_chunked"`` for the designed-init runs).
    """
    return {"tag": tag, "tree": tree, "data_spec": dict(data_spec), "model_spec": dict(model_spec),
            "train_config": train_config, "out": out, "design": dict(design or {})}


def save_designed_run(seed_result: dict[str, Any], folder: str | Path) -> Path:
    """Write a replayable designed-init run as small files (none above ~50 MB).

    ``weights.npz``: ``epochs`` and one ``w__<param>`` stack per parameter;
    ``optimizer_state.npz``: every tensor of the optimizer state at the same epochs, one
    stack per (state key, parameter) as ``s__<key>__<param>`` (SGD: ``momentum_buffer``;
    AdamW: ``exp_avg``, ``exp_avg_sq``, ``step``); ``rotation_lock.npz`` (only for runs with a
    first-layer rotation lock): the locked mask at every snapshot epoch plus the locked
    directions and lock steps, which never change once set; ``meta.pkl``: everything else
    (config, specs, tree, design, losses, lr history, final model state, the optimizer's
    param-group template) without the snapshots. `load_designed_run` reverses this.
    """
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    out = seed_result["out"]; ck = out["checkpoint_snapshots"]; epochs = sorted(ck)
    names = list(ck[epochs[0]].keys())
    np.savez(folder / "weights.npz", epochs=np.array(epochs), **{f"w__{n}": np.stack([ck[e][n].numpy() for e in epochs]) for n in names})
    opt = out["optimizer_snapshots"]
    first = _as_optimizer_state_list(opt[epochs[0]])[0]
    state_keys = sorted({k for st in first["state"].values() for k, v in st.items() if isinstance(v, torch.Tensor)})
    # a fresh optimizer has an empty state at epoch 0; look at the last snapshot for the keys instead
    if not state_keys:
        last = _as_optimizer_state_list(opt[epochs[-1]])[0]
        state_keys = sorted({k for st in last["state"].values() for k, v in st.items() if isinstance(v, torch.Tensor)})
    # a fresh optimizer has an empty state at epoch 0: fill with zeros of the shape the key has once it exists
    # (per-parameter for momentum / Adam moments, 0-dim for "step" and ScalarAdam's shared second moment)
    ref_shape: dict[tuple[str, str], tuple[int, ...]] = {}
    for e in epochs:
        state = _as_optimizer_state_list(opt[e])[0]["state"]
        for i, n in enumerate(names):
            for k in state_keys:
                val = state.get(i, {}).get(k)
                if val is not None and (k, n) not in ref_shape:
                    ref_shape[(k, n)] = tuple(val.shape)
    stacks: dict[str, list[np.ndarray]] = {}
    for e in epochs:
        state = _as_optimizer_state_list(opt[e])[0]["state"]
        for i, n in enumerate(names):
            for k in state_keys:
                val = state.get(i, {}).get(k)
                if val is None:
                    val = torch.zeros(ref_shape.get((k, n), () if k == "step" else tuple(ck[e][n].shape)), dtype=torch.float32)
                stacks.setdefault(f"s__{k}__{n}", []).append(val.detach().cpu().float().numpy())
    np.savez(folder / "optimizer_state.npz", epochs=np.array(epochs), **{k: np.stack(v) for k, v in stacks.items()})
    template = {k: v for k, v in _as_optimizer_state_list(opt[epochs[-1]])[0]["param_groups"][0].items() if k != "params"}
    if out.get("rotation_lock_snapshots"):
        rl = out["rotation_lock_snapshots"]; fin = out["rotation_lock_final"] if out.get("rotation_lock_final") is not None else rl[epochs[-1]]
        np.savez(folder / "rotation_lock.npz", epochs=np.array(epochs), locked=np.stack([rl[e]["locked"] for e in epochs]), step=np.array([int(rl[e]["step"]) for e in epochs]),
                 direction=np.asarray(fin["direction"]), lock_step=np.asarray(fin["lock_step"]))
    light_out = {k: v for k, v in out.items() if k not in ("checkpoint_snapshots", "optimizer_snapshots", "permutation_generator_snapshots", "rotation_lock_snapshots")}
    light_out["model_state_final"] = {k: v.detach().cpu().clone() for k, v in out["model"].state_dict().items()}
    light_out.pop("model", None)
    meta = {k: v for k, v in seed_result.items() if k != "out"}; meta["out"] = light_out; meta["parameter_names"] = names
    meta["optimizer_state_keys"] = state_keys; meta["optimizer_param_group"] = template
    with open(folder / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    return folder


def load_designed_run(folder: str | Path) -> dict[str, Any]:
    """Load a run written by `save_designed_run` back into a replayable seed_result.

    Reads the current format (``optimizer_state.npz``) and the earlier SGD-only one
    (``momentum.npz`` with ``m__<param>`` stacks).
    """
    folder = Path(folder)
    with open(folder / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    w = np.load(folder / "weights.npz")
    epochs = [int(e) for e in w["epochs"]]; names = meta["parameter_names"]
    ck = {e: {n: torch.from_numpy(w[f"w__{n}"][i].copy()) for n in names} for i, e in enumerate(epochs)}
    cfg: ReLUTrainConfig = meta["train_config"]
    lr_hist = meta["out"].get("lr_history") or meta["out"].get("lr_history_by_group", {}).get("sgd")
    if (folder / "optimizer_state.npz").exists():
        st_npz = np.load(folder / "optimizer_state.npz"); state_keys = meta["optimizer_state_keys"]; template = dict(meta["optimizer_param_group"])
        def state_at(i):
            return {j: {k: torch.as_tensor(np.asarray(st_npz[f"s__{k}__{n}"][i]).copy()) for k in state_keys} for j, n in enumerate(names)}
    else:
        m = np.load(folder / "momentum.npz"); template = {"momentum": cfg.momentum, "dampening": 0, "weight_decay": cfg.weight_decay, "nesterov": False, "maximize": False, "foreach": None, "differentiable": False, "fused": None}
        def state_at(i):
            return {j: {"momentum_buffer": torch.from_numpy(m[f"m__{n}"][i].copy())} for j, n in enumerate(names)}
    opt = {}
    for i, e in enumerate(epochs):
        lr = float(lr_hist[e]) if lr_hist is not None and e < len(lr_hist) and not np.isnan(lr_hist[e]) else template.get("lr", cfg.lr)
        group = dict(template); group["lr"] = lr; group["params"] = list(range(len(names)))
        if "initial_lr" in group:
            group["initial_lr"] = float(group["initial_lr"])
        opt[e] = [{"state": state_at(i), "param_groups": [group]}]
    spec = meta["model_spec"]
    model = ReLUMLP(input_dim=spec["input_dim"], hidden_dims=list(spec["hidden_dims"]), bias=spec.get("bias", False), activation=spec.get("activation", "relu"))
    model.load_state_dict(meta["out"]["model_state_final"])
    out = dict(meta["out"]); out["model"] = model; out["checkpoint_snapshots"] = ck; out["optimizer_snapshots"] = opt
    if (folder / "rotation_lock.npz").exists():
        rl = np.load(folder / "rotation_lock.npz"); direction = rl["direction"]; lock_step = rl["lock_step"]
        out["rotation_lock_snapshots"] = {e: {"locked": rl["locked"][i].copy(), "direction": np.where(rl["locked"][i][:, None], direction, 0.0).astype(direction.dtype),
                                              "lock_step": np.where(rl["locked"][i], lock_step, -1), "step": np.int64(rl["step"][i])} for i, e in enumerate(epochs)}
    result = {k: v for k, v in meta.items() if k not in ("out", "parameter_names", "optimizer_state_keys", "optimizer_param_group")}; result["out"] = out
    return result


def replay_interval_from_snapshot(
    seed_result: dict[str, Any],
    epoch_start: int,
    *,
    epoch_end: int | None = None,
    data: dict[str, np.ndarray] | None = None,
    device: str | None = None,
    record_full_data_gradient: bool = False,
    full_data_subsample: int | None = None,
    record_params: bool = False,
    neuron_rows: list[int] | None = None,
) -> dict[str, Any]:
    """Re-run training from the snapshot at ``epoch_start`` up to ``epoch_end`` step by step.

    ``epoch_end`` defaults to the next stored snapshot. Mini-batches follow the fixed
    (identity) order the trainer used, the learning rate of every epoch is taken from the
    stored ``lr_history``, and the momentum buffers come from the snapshot, so the replay
    is exact; the end state is checked against the snapshot at ``epoch_end`` when it exists.

    Per step this records the batch loss, the mini-batch gradient and the update of the
    first-layer rows in ``neuron_rows`` (all rows if None) and, with
    ``record_full_data_gradient``, the gradient of the mean loss over the first
    ``full_data_subsample`` training points (all of them if None) on those rows. With
    ``record_params`` the first-layer rows before every step are kept too. Returns the
    step records as stacked arrays, step-indexed from the start of ``epoch_start``. For a
    run with a first-layer rotation lock the lock state is restored from the snapshot and
    ``locked_rows`` (per step, before the step) and ``lock_step`` (per row) are returned.
    """
    out = seed_result["out"]; config: ReLUTrainConfig = seed_result["train_config"]
    if epoch_start not in out["optimizer_snapshots"]:
        raise KeyError(f"epoch {epoch_start} has no optimizer snapshot; available: {sorted(out['optimizer_snapshots'])[:5]} ...")
    stored = sorted(out["checkpoint_snapshots"])
    if epoch_end is None:
        later = [e for e in stored if e > epoch_start]
        if not later:
            raise ValueError(f"epoch {epoch_start} is the last snapshot; nothing to replay.")
        epoch_end = later[0]
    if not epoch_start < epoch_end <= config.epochs:
        raise ValueError(f"need epoch_start < epoch_end <= {config.epochs}, got {epoch_start}, {epoch_end}")
    dev = torch.device(device if device is not None else config.device)
    if data is None:
        data = regenerate_seed_result_data(seed_result)
    x = torch.from_numpy(data["x_train"]).float().to(dev)
    y_bin = _labels_pm1_to_binary(torch.from_numpy(data["y_train"]).float().to(dev))
    n = x.shape[0]
    n_full = n if full_data_subsample is None else min(int(full_data_subsample), n)
    x_full, y_full = data["x_train"][:n_full], data["y_train"][:n_full]

    model = checkpoint_model_from_state(out["model"], out["checkpoint_snapshots"][epoch_start]).to(dev); model.train()
    optimizers = build_relu_optimizers(model, config)
    for optimizer, st in zip(optimizers, _as_optimizer_state_list(out["optimizer_snapshots"][epoch_start])):
        optimizer.load_state_dict(copy.deepcopy(st))
    lr_hist = out.get("lr_history") or next(iter(out.get("lr_history_by_group", {}).values()), None)
    masks = _parameter_update_masks(model=model, config=config, device=dev)
    proj = _first_layer_projection_vector(model=model, config=config, device=dev)
    lock = None
    if config.first_layer_rotation_lock is not None:
        if epoch_start not in out.get("rotation_lock_snapshots", {}):
            raise KeyError(f"epoch {epoch_start} has no rotation-lock snapshot although the run uses a rotation lock")
        lock = rotation_lock_state(model=model, config=config, device=dev, snapshot=out["rotation_lock_snapshots"][epoch_start])
    criterion = torch.nn.BCEWithLogitsLoss()
    rows = list(range(model.hidden_layers[0].weight.shape[0])) if neuron_rows is None else list(neuron_rows)
    W1 = model.hidden_layers[0].weight
    rec = {"step": [], "epoch": [], "batch_loss": [], "grad_rows": [], "update_rows": [], "full_grad_rows": [], "params_rows": [], "lr": [], "locked_rows": []}
    step = 0
    for epoch in range(epoch_start, epoch_end):
        if lr_hist is not None and not np.isnan(lr_hist[epoch]):
            for optimizer in optimizers:
                for g in optimizer.param_groups:
                    g["lr"] = float(lr_hist[epoch])
        for start in range(0, n, config.batch_size):
            idx = slice(start, min(start + config.batch_size, n))
            before = W1.detach()[rows].cpu().clone()
            if record_params:
                rec["params_rows"].append(before.numpy())
            if record_full_data_gradient:
                fg = log_loss_gradients(model=model, x=x_full, y=y_full, device=str(dev)); model.train()
                rec["full_grad_rows"].append(fg["hidden_layers.0.weight"][rows].numpy())
            if lock is not None:
                rec["locked_rows"].append(lock.locked[rows].cpu().numpy().copy())
            loss = relu_training_step(model=model, optimizers=optimizers, criterion=criterion, x_batch=x[idx], y_batch_bin=y_bin[idx],
                                      first_layer_projection_vector=proj, parameter_update_masks=masks, rotation_lock=lock)
            rec["grad_rows"].append(W1.grad.detach()[rows].cpu().numpy().copy())
            rec["update_rows"].append((W1.detach()[rows].cpu() - before).numpy())
            rec["batch_loss"].append(float(loss.cpu())); rec["step"].append(step); rec["epoch"].append(epoch); rec["lr"].append(float(optimizers[0].param_groups[0]["lr"]))
            step += 1
    params_after = _named_parameter_tensors(model)
    verification = None
    if epoch_end in out["checkpoint_snapshots"]:
        nxt = out["checkpoint_snapshots"][epoch_end]
        diff = {k: float((params_after[k] - nxt[k].float()).abs().max()) for k in params_after}
        verification = {"epoch_end": epoch_end, "max_abs_param_diff": diff, "max_abs_param_diff_overall": max(diff.values())}
        if epoch_end in out["optimizer_snapshots"]:
            st_next = _as_optimizer_state_list(out["optimizer_snapshots"][epoch_end])[0]["state"]
            st_now = _state_dict_to_cpu(optimizers[0].state_dict())["state"]
            diffs = [float((st_now[k][key].float() - val.float()).abs().max()) for k in st_next for key, val in st_next[k].items()
                     if isinstance(val, torch.Tensor) and val.dtype.is_floating_point and k in st_now and key in st_now[k]]
            verification["max_abs_momentum_diff"] = max(diffs) if diffs else 0.0      # all optimizer-state tensors (momentum, Adam moments, step)
        if lock is not None and epoch_end in out.get("rotation_lock_snapshots", {}):
            verification["locked_rows_mismatch"] = int((lock.locked.cpu().numpy() != out["rotation_lock_snapshots"][epoch_end]["locked"]).sum())
    result = {"epoch_start": epoch_start, "epoch_end": epoch_end, "rows": rows, "num_steps": step, "verification": verification, "params_after": params_after,
              "step": np.array(rec["step"]), "epoch": np.array(rec["epoch"]), "lr": np.array(rec["lr"]), "batch_loss": np.array(rec["batch_loss"]),
              "grad_rows": np.stack(rec["grad_rows"]), "update_rows": np.stack(rec["update_rows"])}
    if record_full_data_gradient:
        result["full_grad_rows"] = np.stack(rec["full_grad_rows"]); result["full_data_subsample"] = n_full
    if record_params:
        result["params_rows"] = np.stack(rec["params_rows"])
    if lock is not None:
        result["locked_rows"] = np.stack(rec["locked_rows"]); result["lock_step"] = lock.lock_step.cpu().numpy().copy()
    return result


# ---------------------------------------------------------------------------------------------
# Step anatomy: full-parameter replay of a few steps, loss along an update, curvature
# ---------------------------------------------------------------------------------------------


def replay_steps_full(
    seed_result: dict[str, Any],
    epoch_start: int,
    *,
    step_from: int = 0,
    step_to: int | None = None,
    data: dict[str, np.ndarray] | None = None,
    device: str | None = None,
    probe_points: int | None = 20_000,
) -> dict[str, Any]:
    """Replay from the snapshot at ``epoch_start`` and record every parameter for steps ``step_from .. step_to-1``.

    Step indices count from the start of ``epoch_start`` (``step_to`` defaults to the next
    stored snapshot). Steps before ``step_from`` are replayed silently. For each recorded step
    ``t`` the lists hold, in order: ``params_before`` (dict name -> array, the parameters the
    step starts from), ``momentum_before`` (the SGD momentum buffers, same keys; empty for a
    fresh buffer), ``grads`` (the mini-batch gradient the optimizer consumed), ``updates``
    (``params_after - params_before``), ``batch_loss`` (on the step's mini-batch, before the
    step), ``probe_loss`` (mean loss on the first ``probe_points`` training points, before the
    step; None disables it), ``lr`` and ``batch_index``. ``params_after`` / ``momentum_after``
    are the state after the last recorded step.

    Works for every optimizer the config can name. ``optimizer_state_before`` holds all
    optimizer-state tensors per parameter (SGD: ``momentum_buffer``; AdamW / ScalarAdam:
    ``exp_avg``, ``exp_avg_sq``, ``step``), ``momentum_before`` only SGD's buffers. ``beta1`` is the
    momentum coefficient (SGD ``momentum``, ScalarAdam / AdamW beta1) and ``lr_sgd_equiv`` the
    learning rate of the plain heavy-ball step the optimizer's step amounts to: ``lr`` for SGD,
    ``lr (1 - beta1) / ((1 - beta1^t) (sqrt(v_hat) + eps))`` for ScalarAdam (so the stability
    products ``lr_sgd_equiv * lambda`` compare with ``2 (1 + beta1)``), NaN for AdamW.
    """
    out = seed_result["out"]; config: ReLUTrainConfig = seed_result["train_config"]
    if epoch_start not in out["optimizer_snapshots"]:
        raise KeyError(f"epoch {epoch_start} has no optimizer snapshot")
    dev = torch.device(device if device is not None else config.device)
    if data is None:
        data = regenerate_seed_result_data(seed_result)
    x = torch.from_numpy(data["x_train"]).float().to(dev); y_bin = _labels_pm1_to_binary(torch.from_numpy(data["y_train"]).float().to(dev)); n = x.shape[0]
    steps_per_epoch = -(-n // config.batch_size)
    later = [e for e in sorted(out["checkpoint_snapshots"]) if e > epoch_start]
    if step_to is None:
        if not later:
            raise ValueError(f"epoch {epoch_start} is the last snapshot")
        step_to = (later[0] - epoch_start) * steps_per_epoch
    if not 0 <= step_from < step_to <= (config.epochs - epoch_start) * steps_per_epoch:
        raise ValueError(f"need 0 <= step_from < step_to <= {(config.epochs - epoch_start) * steps_per_epoch}, got {step_from}, {step_to}")
    model = checkpoint_model_from_state(out["model"], out["checkpoint_snapshots"][epoch_start]).to(dev); model.train()
    optimizers = build_relu_optimizers(model, config)
    for optimizer, st in zip(optimizers, _as_optimizer_state_list(out["optimizer_snapshots"][epoch_start])):
        optimizer.load_state_dict(copy.deepcopy(st))
    lr_hist = out.get("lr_history") or next(iter(out.get("lr_history_by_group", {}).values()), None)
    masks = _parameter_update_masks(model=model, config=config, device=dev); proj = _first_layer_projection_vector(model=model, config=config, device=dev)
    lock = None
    if config.first_layer_rotation_lock is not None:
        lock = rotation_lock_state(model=model, config=config, device=dev, snapshot=out["rotation_lock_snapshots"][epoch_start])
    criterion = torch.nn.BCEWithLogitsLoss(); names = [nm for nm, _ in model.named_parameters()]
    kind = normalized_optimizer_name(config)
    beta1 = float(config.momentum) if kind == "sgd" else float(config.adam_betas[0]) if kind == "scalar_adam" else 0.9
    def momentum_now() -> dict[str, np.ndarray]:
        st = optimizers[0].state_dict()["state"]
        return {nm: st[i]["momentum_buffer"].detach().cpu().numpy().copy() for i, nm in enumerate(names) if i in st and st[i].get("momentum_buffer") is not None}
    def optimizer_state_now() -> dict[str, dict[str, np.ndarray]]:
        st = optimizers[0].state_dict()["state"]
        return {nm: {k: v.detach().cpu().numpy().copy() for k, v in st[i].items() if isinstance(v, torch.Tensor)} for i, nm in enumerate(names) if i in st}
    def lr_sgd_equiv_now() -> float:
        # the learning rate of the plain heavy-ball step (b = beta b + g; p -= lr_eq b) that this optimizer's next step amounts to
        lr_now = float(optimizers[0].param_groups[0]["lr"])
        if kind == "sgd":
            return lr_now
        if kind == "scalar_adam":
            st = optimizers[0].state_dict()["state"]
            if 0 not in st or "exp_avg_sq" not in st[0]:
                return float("nan")
            t = float(st[0]["step"]) + 1.0; b1, b2 = config.adam_betas; eps = float(optimizers[0].param_groups[0]["eps"])
            v_next_floor = float(st[0]["exp_avg_sq"]) * b2 / (1 - b2 ** t)          # the bias-corrected v the step will use, before this step's |g|^2 is added
            return lr_now * (1 - b1) / ((1 - b1 ** t) * (np.sqrt(v_next_floor) + eps))
        return float("nan")
    def probe_loss() -> float | None:
        if probe_points is None:
            return None
        with torch.no_grad():
            return float(criterion(model(x[:probe_points]), y_bin[:probe_points]).cpu())
    rec: dict[str, list] = {k: [] for k in ("step", "epoch", "batch_index", "lr", "lr_sgd_equiv", "batch_loss", "probe_loss", "params_before", "momentum_before", "optimizer_state_before", "grads", "updates")}
    for t in range(step_to):
        epoch = epoch_start + t // steps_per_epoch; b = t % steps_per_epoch
        if b == 0 and lr_hist is not None and not np.isnan(lr_hist[epoch]):
            for optimizer in optimizers:
                for g in optimizer.param_groups:
                    g["lr"] = float(lr_hist[epoch])
        idx = slice(b * config.batch_size, min((b + 1) * config.batch_size, n))
        record = t >= step_from
        if record:
            before = {nm: p.detach().cpu().numpy().copy() for nm, p in model.named_parameters()}
            rec["params_before"].append(before); rec["momentum_before"].append(momentum_now()); rec["optimizer_state_before"].append(optimizer_state_now()); rec["probe_loss"].append(probe_loss())
            rec["lr_sgd_equiv"].append(lr_sgd_equiv_now())
        loss = relu_training_step(model=model, optimizers=optimizers, criterion=criterion, x_batch=x[idx], y_batch_bin=y_bin[idx],
                                  first_layer_projection_vector=proj, parameter_update_masks=masks, rotation_lock=lock)
        if record:
            rec["grads"].append({nm: (p.grad.detach().cpu().numpy().copy() if p.grad is not None else np.zeros(tuple(p.shape), dtype=np.float32)) for nm, p in model.named_parameters()})
            rec["updates"].append({nm: p.detach().cpu().numpy() - before[nm] for nm, p in model.named_parameters()})
            rec["batch_loss"].append(float(loss.cpu())); rec["step"].append(t); rec["epoch"].append(epoch); rec["batch_index"].append(b); rec["lr"].append(float(optimizers[0].param_groups[0]["lr"]))
    result: dict[str, Any] = {k: (np.array(v) if k in ("step", "epoch", "batch_index", "lr", "lr_sgd_equiv", "batch_loss") else v) for k, v in rec.items()}
    result["probe_loss"] = np.array([np.nan if v is None else v for v in rec["probe_loss"]])
    result["params_after"] = {nm: p.detach().cpu().numpy().copy() for nm, p in model.named_parameters()}; result["momentum_after"] = momentum_now()
    result["probe_loss_after"] = probe_loss(); result["epoch_start"] = epoch_start; result["steps_per_epoch"] = steps_per_epoch; result["probe_points"] = probe_points
    result["momentum"] = beta1; result["beta1"] = beta1; result["optimizer"] = kind; result["parameter_names"] = names
    return result


def model_from_param_arrays(template_model: ReLUMLP, params: dict[str, np.ndarray], device: str = "cpu") -> ReLUMLP:
    """Copy of ``template_model`` with its parameters set from ``params`` (name -> array)."""
    model = copy.deepcopy(template_model).to(torch.device(device))
    with torch.no_grad():
        for nm, p in model.named_parameters():
            p.copy_(torch.as_tensor(np.asarray(params[nm]), dtype=p.dtype, device=p.device))
    return model


def loss_on_points(template_model: ReLUMLP, params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, *, device: str = "cpu", batch_size: int = 200_000) -> float:
    """Mean BCE log loss of the network with parameters ``params`` on ``(x, y)``, ``y`` in {-1, +1}."""
    model = model_from_param_arrays(template_model, params, device); model.eval(); dev = torch.device(device)
    crit = torch.nn.BCEWithLogitsLoss(reduction="sum"); total = 0.0
    with torch.no_grad():
        for s in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[s:s + batch_size]).float().to(dev); yb = _labels_pm1_to_binary(torch.from_numpy(y[s:s + batch_size]).float().to(dev))
            total += float(crit(model(xb), yb).cpu())
    return total / x.shape[0]


def loss_along_update(template_model: ReLUMLP, params: dict[str, np.ndarray], update: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray,
                      alphas: np.ndarray | list[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0), *, device: str = "cpu") -> pd.DataFrame:
    """Loss on ``(x, y)`` at ``params + alpha * update`` for each alpha (alpha = 1 is the step the optimizer took)."""
    rows = [{"alpha": float(a), "loss": loss_on_points(template_model, {k: params[k] + a * update[k] for k in params}, x, y, device=device)} for a in alphas]
    return pd.DataFrame(rows)


def layerwise_update_ablation(template_model: ReLUMLP, params: dict[str, np.ndarray], update: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, *, device: str = "cpu") -> pd.DataFrame:
    """Loss on ``(x, y)`` when the update is applied to one parameter tensor only, or to all but one.

    Rows: ``none`` (before the step), ``all`` (the step as taken), ``only <name>`` and ``all but <name>``
    for every parameter tensor; ``update_norm`` is the norm of that tensor's update.
    """
    rows = [{"applied": "none", "loss": loss_on_points(template_model, params, x, y, device=device), "update_norm": 0.0},
            {"applied": "all", "loss": loss_on_points(template_model, {k: params[k] + update[k] for k in params}, x, y, device=device), "update_norm": float(np.sqrt(sum((u ** 2).sum() for u in update.values())))}]
    for nm in params:
        only = {k: params[k] + (update[k] if k == nm else 0.0) for k in params}; rest = {k: params[k] + (0.0 if k == nm else update[k]) for k in params}
        rows.append({"applied": f"only {nm}", "loss": loss_on_points(template_model, only, x, y, device=device), "update_norm": float(np.linalg.norm(update[nm]))})
        rows.append({"applied": f"all but {nm}", "loss": loss_on_points(template_model, rest, x, y, device=device), "update_norm": float(np.sqrt(sum((u ** 2).sum() for k, u in update.items() if k != nm)))})
    return pd.DataFrame(rows)


def flatten_parameter_dict(d: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """Concatenate the arrays of ``d`` in the order ``names`` into one flat vector."""
    return np.concatenate([np.asarray(d[nm]).ravel() for nm in names])


_flat = flatten_parameter_dict


def hessian_quadratic_form(template_model: ReLUMLP, params: dict[str, np.ndarray], direction: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, *, device: str = "cpu") -> dict[str, float]:
    """Curvature of the mean loss on ``(x, y)`` along ``direction``: ``d^T H d / |d|^2``, plus the gradient component ``g . d / |d|``.

    Computed with one Hessian-vector product (double backward) through the ReLU network at ``params``.
    """
    model = model_from_param_arrays(template_model, params, device); model.train(); dev = torch.device(device)
    xb = torch.from_numpy(x).float().to(dev); yb = _labels_pm1_to_binary(torch.from_numpy(y).float().to(dev))
    plist = [p for _, p in model.named_parameters()]; dlist = [torch.as_tensor(np.asarray(direction[nm]), dtype=p.dtype, device=dev) for nm, p in model.named_parameters()]
    norm = float(torch.sqrt(sum((d ** 2).sum() for d in dlist)).cpu())
    if norm == 0.0:
        raise ValueError("direction must be nonzero")
    dlist = [d / norm for d in dlist]
    loss = torch.nn.BCEWithLogitsLoss()(model(xb), yb)
    grads = torch.autograd.grad(loss, plist, create_graph=True)
    gd = sum((g * d).sum() for g, d in zip(grads, dlist))
    hv = torch.autograd.grad(gd, plist)
    return {"curvature": float(sum((h * d).sum() for h, d in zip(hv, dlist)).cpu()), "grad_along": float(gd.detach().cpu()), "direction_norm": norm, "loss": float(loss.detach().cpu())}


def top_hessian_eigenvalue(template_model: ReLUMLP, params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, *, device: str = "cpu", iters: int = 30,
                           mask: dict[str, np.ndarray] | None = None, seed: int = 0, init_direction: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    """Largest Hessian eigenvalue of the mean loss on ``(x, y)`` at ``params`` by power iteration.

    ``mask`` (0/1 per parameter tensor, e.g. the training update mask) restricts the eigenproblem to the
    trainable coordinates. Power iteration finds the eigenvalue of largest magnitude; when that one is
    negative a second, shifted pass finds the largest algebraic eigenvalue instead. Returns ``lambda_max``,
    the unit eigenvector (dict name -> array), the fraction of its squared norm in each parameter tensor,
    and the Rayleigh quotients of every iteration.
    """
    model = model_from_param_arrays(template_model, params, device); model.train(); dev = torch.device(device)
    xb = torch.from_numpy(x).float().to(dev); yb = _labels_pm1_to_binary(torch.from_numpy(y).float().to(dev))
    named = list(model.named_parameters()); plist = [p for _, p in named]
    m = [torch.as_tensor(np.asarray(mask[nm]), dtype=p.dtype, device=dev) if mask is not None else torch.ones_like(p) for nm, p in named]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    v = [torch.as_tensor(np.asarray(init_direction[nm]), dtype=p.dtype, device=dev) if init_direction is not None else torch.randn(p.shape, generator=gen).to(dev) for nm, p in named]
    v = [vi * mi for vi, mi in zip(v, m)]; nv = torch.sqrt(sum((vi ** 2).sum() for vi in v)); v = [vi / nv for vi in v]
    loss = torch.nn.BCEWithLogitsLoss()(model(xb), yb); grads = torch.autograd.grad(loss, plist, create_graph=True)
    def power(v, shift):                      # power iteration on (H - shift I) restricted by the mask; returns Rayleigh quotients of H and the final vector
        hist = []
        for _ in range(iters):
            gd = sum((g * vi).sum() for g, vi in zip(grads, v)); hv = torch.autograd.grad(gd, plist, retain_graph=True)
            hv = [h * mi for h, mi in zip(hv, m)]; hist.append(float(sum((h * vi).sum() for h, vi in zip(hv, v)).cpu()))
            hv = [h - shift * vi for h, vi in zip(hv, v)]
            nh = torch.sqrt(sum((h ** 2).sum() for h in hv)); v = [h / nh for h in hv]
        return hist, v
    history, v = power(v, 0.0)
    if history[-1] < 0:                       # the dominant eigenvalue by magnitude is negative: shift and iterate again for the largest algebraic one
        shift = history[-1]; history2, v = power(v, shift); history = history + history2
    vec = {nm: vi.detach().cpu().numpy().copy() for (nm, _), vi in zip(named, v)}
    return {"lambda_max": history[-1], "eigenvector": vec, "layer_fraction": {nm: float((vi ** 2).sum()) for nm, vi in vec.items()}, "history": history}
