"""Analysis helpers for ReLU MLPs trained on COB-ODT data.

These utilities keep exploratory notebooks focused on plotting choices and
interpretation while preserving the reusable mechanics in importable code.
"""

from __future__ import annotations

import copy
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
    checkpoint_model_from_state,
    evaluate_relu_mlp,
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
