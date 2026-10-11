import copy
import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt

from layer_lenses.odt import generate_cob_odt_data
from layer_lenses.relu_analysis import (
    activation_tensor_by_layer_neuron_data,
    active_point_mask_for_neuron,
    count_active_points_for_neuron_leaf,
    first_layer_neuron_weighted_lens,
    log_loss_gradients,
    masked_relu_logits,
    neuron_leaf_responsibility_from_ablation,
    epoch_replay_summary_table,
    first_layer_alignment_series,
    first_layer_odt_alignment,
    plot_first_layer_alignment_series,
    snapshot_state_dict,
    state_dict_from_weight_arrays,
    flatten_named_tensors,
    plot_layered_relu_graph,
    regenerate_seed_result_data,
    replay_epoch_from_snapshot,
    run_single_relu_seed,
    run_single_relu_seed_replayable,
    summarize_neuron_leaf_activity,
)
from layer_lenses.relu_mlp import ReLUMLP
from layer_lenses.relu_training import ReLUTrainConfig, train_relu_mlp


def test_first_layer_weighted_lens_matches_autograd_for_one_to_three_layers() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 5)).astype(np.float32)

    for hidden_dims in ([4], [4, 3], [4, 3, 2]):
        model = ReLUMLP(input_dim=5, hidden_dims=list(hidden_dims), bias=False)
        activations = activation_tensor_by_layer_neuron_data(
            model=model,
            x_eval=x,
        )
        x_t = torch.from_numpy(x)
        first_preactivation = model.hidden_layers[0](x_t)
        first_preactivation.retain_grad()
        h = torch.relu(first_preactivation)
        for layer in model.hidden_layers[1:]:
            h = torch.relu(layer(h))
        model.output_layer(h).sum().backward()

        for neuron_id in range(hidden_dims[0]):
            lens = first_layer_neuron_weighted_lens(
                state_dict=model.state_dict(),
                activation_tensor=activations,
                first_layer_neuron_id=neuron_id,
            )
            expected = first_preactivation.grad[:, neuron_id].numpy()

            assert np.allclose(lens, expected, atol=1e-6)


def test_masked_relu_logits_accepts_different_width_per_layer_masks() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(12, 5)).astype(np.float32)
    model = ReLUMLP(input_dim=5, hidden_dims=[4, 3, 2], bias=False)
    node_masks = [
        np.ones(4, dtype=bool),
        np.ones(3, dtype=bool),
        np.ones(2, dtype=bool),
    ]

    masked_logits = masked_relu_logits(model, x, node_masks)
    expected_logits = model(torch.from_numpy(x)).detach().numpy()

    assert np.allclose(masked_logits, expected_logits, atol=1e-6)


def test_neuron_responsibility_accepts_per_layer_masks() -> None:
    x, y, tree, _ = generate_cob_odt_data(
        num_data=32,
        dim=5,
        depth=2,
        seed=4,
    )
    model = ReLUMLP(input_dim=5, hidden_dims=[3, 2], bias=False)
    base_node_mask = [
        np.ones(3, dtype=bool),
        np.ones(2, dtype=bool),
    ]

    _, detail, baseline = neuron_leaf_responsibility_from_ablation(
        model=model,
        x_eval=x,
        y_eval=y,
        tree=tree,
        base_node_mask=base_node_mask,
        min_log_loss_delta=-1e6,
    )

    assert set(detail["layer"]) == {0, 1}
    assert set(detail["neuron_id"]) == {0, 1, 2}
    assert not baseline.empty


def test_plot_layered_relu_graph_supports_one_to_three_hidden_layers() -> None:
    _, _, tree, _ = generate_cob_odt_data(num_data=32, dim=5, depth=2, seed=2)

    for hidden_dims in ([4], [4, 3], [4, 3, 2]):
        model = ReLUMLP(input_dim=5, hidden_dims=list(hidden_dims), bias=False)
        expected_edges = sum(
            src_width * dst_width
            for src_width, dst_width in zip(hidden_dims, hidden_dims[1:])
        ) + hidden_dims[-1]

        _, visible_edges, fig, _ = plot_layered_relu_graph(
            state_dict=model.state_dict(),
            tree=tree,
            depth=2,
            epoch=0,
            edge_abs_threshold=0.0,
        )

        assert len(visible_edges) == expected_edges
        plt.close(fig)


def test_plot_layered_relu_graph_accepts_per_layer_masks_with_different_widths() -> None:
    _, _, tree, _ = generate_cob_odt_data(num_data=32, dim=5, depth=2, seed=3)
    model = ReLUMLP(input_dim=5, hidden_dims=[4, 3, 2], bias=False)
    node_masks = [
        np.array([True, False, False, False]),
        np.array([True, True, False]),
        np.array([False, True]),
    ]

    _, visible_edges, fig, _ = plot_layered_relu_graph(
        state_dict=model.state_dict(),
        tree=tree,
        depth=2,
        epoch=0,
        node_mask=node_masks,
        edge_abs_threshold=0.0,
    )

    assert len(visible_edges) == 5
    plt.close(fig)


def test_summarize_neuron_leaf_activity_matches_point_queries() -> None:
    x, y, tree, _ = generate_cob_odt_data(num_data=128, dim=10, depth=3, seed=5)
    x_train, y_train = x[:64], y[:64]
    x_eval = x[64:]

    model = ReLUMLP(input_dim=10, hidden_dims=[8, 8], bias=False)
    trained = train_relu_mlp(
        model=model,
        x_train=x_train,
        y_train=y_train,
        config=ReLUTrainConfig(
            epochs=2,
            lr=1e-3,
            batch_size=32,
            seed=7,
            snapshot_epochs=(0, 2),
        ),
    )["model"]

    summary = summarize_neuron_leaf_activity(model=trained, x_eval=x_eval, tree=tree)

    # Spot-check multiple neurons/leaves against the scalar helper.
    neurons_to_check = [0, 3, 7]
    leaf_ids = sorted(summary["leaf_node_id"].unique().tolist())[:3]
    for neuron_id in neurons_to_check:
        for leaf_node_id in leaf_ids:
            scalar = count_active_points_for_neuron_leaf(
                model=trained,
                neuron_id=neuron_id,
                leaf_node_id=int(leaf_node_id),
                x_eval=x_eval,
                tree=tree,
            )
            row = summary[
                (summary["global_neuron_id"] == neuron_id)
                & (summary["leaf_node_id"] == int(leaf_node_id))
            ].iloc[0]
            assert int(row["active_points_in_leaf"]) == int(scalar["active_points_in_leaf"])
            assert int(row["total_points_in_leaf"]) == int(scalar["total_points_in_leaf"])


def test_activation_tensor_by_layer_neuron_data_shape_and_values() -> None:
    x, y, _, _ = generate_cob_odt_data(num_data=96, dim=10, depth=3, seed=17)
    x_train, y_train = x[:48], y[:48]
    x_eval = x[48:]

    model = ReLUMLP(input_dim=10, hidden_dims=[5, 7], bias=False)
    trained = train_relu_mlp(
        model=model,
        x_train=x_train,
        y_train=y_train,
        config=ReLUTrainConfig(
            epochs=2,
            lr=1e-3,
            batch_size=24,
            seed=19,
            snapshot_epochs=(0, 2),
        ),
    )["model"]

    tensor = activation_tensor_by_layer_neuron_data(model=trained, x_eval=x_eval)
    assert tensor.dtype == np.bool_
    assert tensor.shape == (2, 7, x_eval.shape[0])

    # Second layer has full width 7; first layer width is 5 so padded rows should be all False.
    assert np.all(~tensor[0, 5:, :])

    # Check consistency with scalar neuron masks.
    mask_l0_n3 = active_point_mask_for_neuron(model=trained, x_eval=x_eval, neuron_id=(0, 3))
    mask_l1_n6 = active_point_mask_for_neuron(model=trained, x_eval=x_eval, neuron_id=(1, 6))
    assert np.array_equal(tensor[0, 3, :], mask_l0_n3)
    assert np.array_equal(tensor[1, 6, :], mask_l1_n6)


def test_log_loss_gradients_keys_and_shapes() -> None:
    x = np.random.randn(32, 10).astype(np.float32)
    y = np.where(np.random.randn(32) > 0, 1, -1).astype(np.int8)
    model = ReLUMLP(input_dim=10, hidden_dims=[8, 4], bias=True)

    grads = log_loss_gradients(model=model, x=x, y=y, device="cpu")
    state = model.state_dict()

    assert set(grads.keys()) == set(state.keys())
    for name, tensor in state.items():
        assert grads[name].shape == tensor.shape
        assert torch.isfinite(grads[name]).all()


def test_run_single_relu_seed_uses_provided_model_init_without_mutating_source() -> None:
    torch.manual_seed(123)
    model_init = ReLUMLP(input_dim=6, hidden_dims=[5, 4], bias=False)
    initial_state = {k: v.detach().clone() for k, v in model_init.state_dict().items()}

    seed_result = run_single_relu_seed(
        11,
        dim=6,
        depth=2,
        n_train=32,
        hidden_dims=[5, 4],
        total_epochs=1,
        lr=1e-3,
        batch_size=16,
        weight_decay=0.0,
        snapshot_epochs=(0, 1),
        bias=False,
        model_init=model_init,
        show_progress=False,
    )

    start_snapshot = seed_result["out"]["checkpoint_snapshots"][0]
    for name, tensor in initial_state.items():
        assert torch.allclose(start_snapshot[name], tensor)
        assert torch.allclose(model_init.state_dict()[name], tensor)


def test_run_single_relu_seed_can_use_coordinate_axis_odt_hyperplanes() -> None:
    seed_result = run_single_relu_seed(
        13,
        dim=6,
        depth=2,
        n_train=32,
        hidden_dims=[5, 4],
        total_epochs=1,
        lr=1e-3,
        batch_size=16,
        weight_decay=0.0,
        snapshot_epochs=(0, 1),
        odt_hyperplanes="coordinate_axes",
        show_progress=False,
    )

    tree = seed_result["tree"]
    expected_w_list = np.eye(6, dtype=np.float64)[:3]
    assert np.array_equal(tree.w_list, expected_w_list)


def test_run_single_relu_seed_threads_parameter_update_mask() -> None:
    torch.manual_seed(456)
    model_init = ReLUMLP(input_dim=6, hidden_dims=[5, 4], bias=False)
    initial_state = {k: v.detach().clone() for k, v in model_init.state_dict().items()}
    mask = {name: torch.zeros_like(param) for name, param in model_init.named_parameters()}

    seed_result = run_single_relu_seed(
        17,
        dim=6,
        depth=2,
        n_train=32,
        hidden_dims=[5, 4],
        total_epochs=1,
        lr=1e-2,
        batch_size=16,
        weight_decay=0.1,
        snapshot_epochs=(0, 1),
        bias=False,
        model_init=model_init,
        parameter_update_mask=mask,
        show_progress=False,
    )

    end_snapshot = seed_result["out"]["checkpoint_snapshots"][1]
    for name, tensor in initial_state.items():
        assert torch.allclose(end_snapshot[name], tensor)


def test_run_single_relu_seed_projects_first_layer_gradient_to_selected_odt_node() -> None:
    seed_result = run_single_relu_seed(
        23,
        dim=6,
        depth=2,
        n_train=32,
        hidden_dims=[5, 4],
        total_epochs=1,
        lr=1e-2,
        batch_size=32,
        weight_decay=0.0,
        snapshot_epochs=(0, 1),
        odt_hyperplanes="coordinate_axes",
        optimizer="sgd",
        project_first_layer_grad_orthogonal=2,
        show_progress=False,
    )

    snapshots = seed_result["out"]["checkpoint_snapshots"]
    first_layer_update = (
        snapshots[1]["hidden_layers.0.weight"] - snapshots[0]["hidden_layers.0.weight"]
    )

    assert torch.allclose(
        first_layer_update[:, 2],
        torch.zeros_like(first_layer_update[:, 2]),
        atol=1e-7,
    )


def test_run_single_relu_seed_rejects_invalid_projection_node() -> None:
    try:
        run_single_relu_seed(
            29,
            dim=6,
            depth=2,
            n_train=32,
            hidden_dims=[5, 4],
            total_epochs=1,
            lr=1e-3,
            batch_size=16,
            weight_decay=0.0,
            snapshot_epochs=(0, 1),
            project_first_layer_grad_orthogonal=3,
            show_progress=False,
        )
    except ValueError as exc:
        assert "ODT internal node" in str(exc)
    else:
        raise AssertionError("Expected invalid projection node to raise ValueError.")


@pytest.mark.parametrize(
    ("optimizer", "extra"),
    [
        ("adam", {}),
        ("sgd", {"momentum": 0.9}),
        ("sgd_adam", {"momentum": 0.9, "sgd_adam_split_layer": 1, "sgd_lr": 0.5}),
    ],
)
def test_replay_epoch_reproduces_the_next_snapshot(optimizer, extra) -> None:
    seed_result = run_single_relu_seed_replayable(
        5,
        dim=6,
        depth=2,
        n_train=200,
        hidden_dims=[4, 4],
        total_epochs=3,
        lr=2e-3,
        batch_size=50,
        weight_decay=0.01,
        snapshot_epochs=[0, 1, 2, 3],
        show_progress=False,
        optimizer=optimizer,
        **extra,
    )
    assert "data" not in seed_result

    replay = replay_epoch_from_snapshot(seed_result, 1)
    # Replaying the epoch must land exactly on the stored next checkpoint, including
    # the optimizer's own state (Adam moments / SGD momentum buffers).
    assert replay["verification"]["max_abs_param_diff_overall"] == 0.0
    assert replay["verification"]["max_abs_optimizer_state_diff"] == 0.0
    assert replay["num_steps"] == 4

    # The per-step updates must sum to the epoch's net parameter change.
    total = sum(flatten_named_tensors(s["update"]) for s in replay["steps"])
    delta = flatten_named_tensors(replay["params_after"]) - flatten_named_tensors(
        replay["params_before"]
    )
    assert float((total - delta).abs().max()) < 1e-9

    summary = epoch_replay_summary_table(replay)
    assert len(summary) == replay["num_steps"]
    assert {"grad_norm", "update_norm", "cos_update_grad"} <= set(summary.columns)


def test_replay_final_snapshot_is_rejected() -> None:
    seed_result = run_single_relu_seed_replayable(
        5, dim=6, depth=2, n_train=200, hidden_dims=[4], total_epochs=2, lr=2e-3,
        batch_size=100, weight_decay=0.0, snapshot_epochs=[0, 1, 2], show_progress=False,
    )
    with pytest.raises(ValueError):
        replay_epoch_from_snapshot(seed_result, 2)
    with pytest.raises(KeyError):
        replay_epoch_from_snapshot(seed_result, 99)


def test_regenerate_seed_result_data_matches_the_training_data() -> None:
    kwargs = dict(
        dim=6, depth=2, n_train=200, hidden_dims=[4, 4], total_epochs=2, lr=2e-3,
        batch_size=50, weight_decay=0.0, snapshot_epochs=[0, 1, 2], show_progress=False,
    )
    legacy = run_single_relu_seed(5, **kwargs)
    replayable = run_single_relu_seed_replayable(5, **kwargs)

    data = regenerate_seed_result_data(replayable)
    for key, value in legacy["data"].items():
        assert np.array_equal(data[key], value)

    # The two runners must agree bit for bit on the trained weights.
    for epoch, state in legacy["out"]["checkpoint_snapshots"].items():
        for name, tensor in state.items():
            assert torch.equal(tensor, replayable["out"]["checkpoint_snapshots"][epoch][name])


def test_hybrid_replay_separates_sgd_and_adam_parameters() -> None:
    seed_result = run_single_relu_seed_replayable(
        5, dim=6, depth=2, n_train=200, hidden_dims=[4, 4], total_epochs=2, lr=2e-3,
        batch_size=50, weight_decay=0.0, sgd_weight_decay=0.0, snapshot_epochs=[0, 1, 2],
        show_progress=False, optimizer="sgd_adam", sgd_adam_split_layer=1, sgd_lr=0.5,
        momentum=0.0,
    )
    replay = replay_epoch_from_snapshot(seed_result, 0)
    assert replay["parameter_groups"]["sgd"] == ["hidden_layers.0.weight"]
    assert "output_layer.weight" in replay["parameter_groups"]["adamw"]
    assert replay["lr_by_group"]["sgd"] == pytest.approx(0.5)
    assert replay["lr_by_group"]["adamw"] == pytest.approx(2e-3)

    # The Adam decomposition must cover the AdamW parameters and only those.
    step = replay["steps"][0]
    assert set(step["adam_step_size"]) == set(replay["parameter_groups"]["adamw"])

    # Layer 0 follows plain SGD exactly; the Adam layers do not.
    grad0 = step["grad"]["hidden_layers.0.weight"]
    update0 = step["update"]["hidden_layers.0.weight"]
    assert torch.allclose(update0, -0.5 * grad0, atol=1e-7)

    # Restricting the summary table to one group must work and change the numbers.
    sgd_table = epoch_replay_summary_table(replay, parameter_names=replay["parameter_groups"]["sgd"])
    adam_table = epoch_replay_summary_table(replay, parameter_names=replay["parameter_groups"]["adamw"])
    assert sgd_table["cos_update_grad"].iloc[0] == pytest.approx(-1.0, abs=1e-4)
    assert adam_table["cos_update_grad"].iloc[0] > -0.99
    assert "adam_step_size_median" not in sgd_table.columns
    assert "adam_step_size_median" in adam_table.columns


def test_snapshot_state_dict_round_trips_weight_arrays_and_plots() -> None:
    _, _, tree, _ = generate_cob_odt_data(num_data=32, dim=5, depth=2, seed=4)
    model = ReLUMLP(input_dim=5, hidden_dims=[4, 3, 2], bias=False)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs = np.array([0, 10, 20])
    run = {"epochs": epochs, **{f"w__{k}": np.stack([v.numpy() * (1 + t) for t in range(3)]) for k, v in sd.items()}}

    rebuilt = snapshot_state_dict(run, 10)
    assert set(rebuilt) == set(sd)
    for k in sd:
        assert torch.allclose(rebuilt[k], 2 * sd[k])
    assert rebuilt["output_layer.weight"].shape == (1, 2)
    with pytest.raises(ValueError):
        snapshot_state_dict(run, 5)

    direct = state_dict_from_weight_arrays([sd["hidden_layers.0.weight"].numpy()], np.ones(4, dtype=np.float32))
    assert direct["output_layer.weight"].shape == (1, 4)

    _, visible_edges, fig, _ = plot_layered_relu_graph(state_dict=rebuilt, tree=tree, depth=2, epoch=10, edge_abs_threshold=0.0)
    assert len(visible_edges) == 4 * 3 + 3 * 2 + 2
    plt.close(fig)


def test_first_layer_alignment_series_matches_direct_cosines() -> None:
    _, _, tree, _ = generate_cob_odt_data(num_data=32, dim=6, depth=2, seed=7)
    rng = np.random.default_rng(0)
    epochs = np.array([0, 5, 10, 15])
    W0 = rng.normal(size=(4, 5, 6)).astype(np.float32)
    W1 = rng.normal(size=(4, 3, 5)).astype(np.float32)

    series = first_layer_alignment_series(W0, epochs, tree, neuron_ids=[1, 3], node_ids=[0, 2], fan_out_weights=W1)
    assert len(series) == 8
    assert list(series.columns) == ["epoch", "neuron", "norm", "cos_0", "cos_2", "fan_out_norm"]

    row = series[(series.epoch == 10) & (series.neuron == 3)].iloc[0]
    w = W0[2, 3].astype(np.float64)
    u2 = tree.w_list[2] / np.linalg.norm(tree.w_list[2])
    assert row.cos_2 == pytest.approx(float(w @ u2 / np.linalg.norm(w)), abs=1e-6)
    assert row.norm == pytest.approx(float(np.linalg.norm(w)), abs=1e-6)
    assert row.fan_out_norm == pytest.approx(float(np.linalg.norm(W1[2][:, 3])), abs=1e-6)

    # agrees with first_layer_odt_alignment on the same snapshot
    sd = state_dict_from_weight_arrays([W0[2]], np.ones((1, 5), dtype=np.float32))
    cos = first_layer_odt_alignment(state_dict=sd, tree=tree)["cos"]
    assert row.cos_0 == pytest.approx(float(cos[3, 0]), abs=1e-5)

    with pytest.raises(ValueError):
        first_layer_alignment_series(W0, epochs, tree, neuron_ids=[9], node_ids=[0])
    with pytest.raises(ValueError):
        first_layer_alignment_series(W0, epochs, tree, neuron_ids=[0], node_ids=[3])

    fig, axes = plot_first_layer_alignment_series(series, title="t")
    assert len(axes) == 2
    plt.close(fig)


def test_population_gradient_at_snapshot_matches_one_shot_gradient() -> None:
    from layer_lenses.relu_analysis import population_gradient_at_snapshot
    rng = np.random.default_rng(3)
    model = ReLUMLP(input_dim=6, hidden_dims=[5, 4], bias=False)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    run = {"epochs": np.array([0, 10]), **{f"w__{k}": np.stack([v.numpy(), 2 * v.numpy()]) for k, v in sd.items()}}
    x = rng.normal(size=(23, 6)).astype(np.float32); y = np.where(rng.random(23) < 0.5, -1, 1).astype(np.int8)
    g_batched = population_gradient_at_snapshot(run, 10, x, y, batch_size=7)
    with torch.no_grad():
        for k, v in model.state_dict().items():
            v.copy_(2 * sd[k])
    g_once = log_loss_gradients(model=model, x=x, y=y)
    assert set(g_batched) == set(g_once)
    for k in g_once:
        assert np.allclose(g_batched[k], g_once[k].numpy(), atol=1e-6)


def test_snapshot_state_dict_accepts_load_hier_run_style_dict() -> None:
    model = ReLUMLP(input_dim=5, hidden_dims=[4, 3], bias=False)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stacks = {k: np.stack([v.numpy(), 3 * v.numpy()]) for k, v in sd.items()}
    flat = {"epochs": np.array([0, 10]), **{f"w__{k}": v for k, v in stacks.items()}}
    nested = {"epochs": np.array([0, 10]), "W": stacks, "tag": "x", "dims": [4, 3]}
    a, b = snapshot_state_dict(flat, 10), snapshot_state_dict(nested, 10)
    assert set(a) == set(b) == set(sd)
    for k in sd:
        assert torch.allclose(a[k], b[k]) and torch.allclose(a[k], 3 * sd[k])
    with pytest.raises(KeyError):
        snapshot_state_dict({"epochs": np.array([0]), "w__hidden_layers.0.weight": stacks["hidden_layers.0.weight"][:1]}, 0)


def test_first_layer_gradient_series_components_and_offspan() -> None:
    from layer_lenses.relu_analysis import first_layer_gradient_series, plot_first_layer_gradient_series
    _, _, tree, _ = generate_cob_odt_data(num_data=16, dim=8, depth=2, seed=9)
    U = tree.w_list / np.linalg.norm(tree.w_list, axis=1, keepdims=True)          # 3 orthonormal rows in R^8
    rng = np.random.default_rng(0)
    off = rng.normal(size=8); off -= U.T @ (U @ off); off *= 2.0 / np.linalg.norm(off)   # off-span vector of norm 2
    G = np.zeros((4, 8)); G[1] = -(3 * U[0] - 1 * U[2] + off)                        # push on row 1 = 3 u_0 - u_2 + off
    series = first_layer_gradient_series({0: G, 10: {"hidden_layers.0.weight": 2 * G}}, tree, neuron_ids=[1], node_ids=[0, 2])
    assert list(series.columns) == ["epoch", "neuron", "grad_norm", "comp_0", "comp_2", "inspan_norm", "offspan_norm"]
    r0, r10 = series.iloc[0], series.iloc[1]
    assert np.isclose(r0.comp_0, 3.0) and np.isclose(r0.comp_2, -1.0) and np.isclose(r0.offspan_norm, 2.0) and np.isclose(r0.inspan_norm, np.sqrt(10))
    assert np.isclose(r0.grad_norm, np.sqrt(14)) and np.isclose(r10.comp_0, 6.0) and np.isclose(r10.offspan_norm, 4.0)
    raw = first_layer_gradient_series({0: G}, tree, neuron_ids=[1], node_ids=[0], push=False)
    assert np.isclose(raw.iloc[0].comp_0, -3.0)
    norm = first_layer_gradient_series({0: G}, tree, neuron_ids=[1], node_ids=[0], normalize=True)
    assert np.isclose(norm.iloc[0].comp_0, 3 / np.sqrt(14)) and np.isclose(norm.iloc[0].grad_norm, 1.0)
    with pytest.raises(ValueError):
        first_layer_gradient_series({0: G}, tree, neuron_ids=[7], node_ids=[0])
    fig, axes = plot_first_layer_gradient_series(series, title="t")
    assert len(axes) == 1
    plt.close(fig)


def test_designed_run_save_load_and_interval_replay_are_exact(tmp_path) -> None:
    from layer_lenses.init_sweep import SweepRunSpec, build_data_chunked, build_tree
    from layer_lenses.relu_analysis import (designed_run_seed_result, load_designed_run, replay_interval_from_snapshot, save_designed_run,
                                            train_relu_mlp_with_optimizer_snapshots, regenerate_seed_result_data)
    tree = build_tree(SweepRunSpec(tree_seed=5, data_seed=0, init_seed=0, batch_size=1, dim=12, depth=2))
    x, y = build_data_chunked(SweepRunSpec(tree_seed=5, data_seed=7, init_seed=0, batch_size=1, dim=12, depth=2, n_train=600), tree, chunk_size=300)
    torch.manual_seed(0)
    model = ReLUMLP(input_dim=12, hidden_dims=[6, 4], bias=False)
    mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}; mask["hidden_layers.0.weight"][2:4] = 0.0
    cfg = ReLUTrainConfig(epochs=8, lr=0.05, batch_size=100, seed=1, device="cpu", snapshot_epochs=(0, 2, 4, 6, 8), lr_scheduler_eta_min_ratio=0.1,
                          optimizer="sgd", momentum=0.9, parameter_update_mask=mask)
    out = train_relu_mlp_with_optimizer_snapshots(model=model, x_train=x, y_train=y, config=cfg)
    data_spec = {"kind": "sweep_chunked", "tree_seed": 5, "data_seed": 7, "n_train": 600, "chunk_size": 300, "eval_seed": 99, "n_eval": 50, "dim": 12, "depth": 2}
    sr = designed_run_seed_result(out=out, train_config=cfg, model_spec={"input_dim": 12, "hidden_dims": [6, 4], "bias": False}, data_spec=data_spec, tree=tree, tag="t")
    regen = regenerate_seed_result_data(sr)
    assert np.allclose(regen["x_train"], x) and np.array_equal(regen["y_train"], y) and regen["x_eval"].shape == (50, 12)
    folder = save_designed_run(sr, tmp_path / "run")
    assert {p.name for p in folder.iterdir()} == {"weights.npz", "optimizer_state.npz", "meta.pkl"}
    lo = load_designed_run(folder)
    assert sorted(lo["out"]["checkpoint_snapshots"]) == [0, 2, 4, 6, 8]
    for e in (0, 4, 8):
        for k, v in out["checkpoint_snapshots"][e].items():
            assert torch.equal(lo["out"]["checkpoint_snapshots"][e][k], v)
    # replay 2 epochs from the loaded run and land exactly on the next snapshot (params and momentum)
    rep = replay_interval_from_snapshot(lo, 2, data=regen, record_full_data_gradient=True, full_data_subsample=200, record_params=True, neuron_rows=[0, 1, 2])
    assert rep["epoch_end"] == 4 and rep["num_steps"] == 12 and rep["grad_rows"].shape == (12, 3, 12) and rep["full_grad_rows"].shape == (12, 3, 12)
    assert rep["verification"]["max_abs_param_diff_overall"] == 0.0 and rep["verification"]["max_abs_momentum_diff"] == 0.0
    assert np.all(rep["update_rows"][:, 2] == 0.0)                      # frozen row never moves
    assert np.allclose(rep["params_rows"][0], out["checkpoint_snapshots"][2]["hidden_layers.0.weight"].numpy()[[0, 1, 2]])
    assert np.allclose(rep["lr"][:6], out["lr_history"][2]) and np.allclose(rep["lr"][6:], out["lr_history"][3])
    rep2 = replay_interval_from_snapshot(lo, 6, data=regen)                # last interval, no explicit end
    assert rep2["epoch_end"] == 8 and rep2["verification"]["max_abs_param_diff_overall"] == 0.0
    with pytest.raises(ValueError):
        replay_interval_from_snapshot(lo, 8, data=regen)


def test_resume_from_snapshot_reproduces_the_run_and_adam_runs_save_load_replay(tmp_path) -> None:
    from layer_lenses.init_sweep import SweepRunSpec, build_data_chunked, build_tree
    from layer_lenses.relu_analysis import (designed_run_seed_result, load_designed_run, replay_interval_from_snapshot, save_designed_run,
                                            train_relu_mlp_with_optimizer_snapshots, checkpoint_model_from_state)
    tree = build_tree(SweepRunSpec(tree_seed=5, data_seed=0, init_seed=0, batch_size=1, dim=12, depth=2))
    x, y = build_data_chunked(SweepRunSpec(tree_seed=5, data_seed=7, init_seed=0, batch_size=1, dim=12, depth=2, n_train=600), tree, chunk_size=600)
    data_spec = {"kind": "sweep_chunked", "tree_seed": 5, "data_seed": 7, "n_train": 600, "chunk_size": 600, "eval_seed": 99, "n_eval": 50, "dim": 12, "depth": 2}
    torch.manual_seed(0); model = ReLUMLP(input_dim=12, hidden_dims=[6, 4], bias=False)
    cfg = ReLUTrainConfig(epochs=8, lr=0.05, batch_size=100, seed=1, device="cpu", snapshot_epochs=(0, 2, 4, 6, 8), lr_scheduler_eta_min_ratio=0.1, optimizer="sgd", momentum=0.9)
    full = train_relu_mlp_with_optimizer_snapshots(model=model, x_train=x, y_train=y, config=cfg)
    # resume at epoch 4 from the stored weights and momentum: identical to the original run from there on
    m4 = checkpoint_model_from_state(full["model"], full["checkpoint_snapshots"][4])
    res = train_relu_mlp_with_optimizer_snapshots(model=m4, x_train=x, y_train=y, config=cfg, start_epoch=4, initial_optimizer_states=full["optimizer_snapshots"][4])
    assert res["start_epoch"] == 4 and sorted(res["checkpoint_snapshots"]) == [4, 6, 8] and len(res["epoch_losses"]) == 8 and np.isnan(res["epoch_losses"][:4]).all()
    assert np.allclose(res["lr_history"][4:], full["lr_history"][4:]) and np.allclose(res["epoch_losses"][4:], full["epoch_losses"][4:])
    for k, v in full["checkpoint_snapshots"][8].items():
        assert torch.equal(res["checkpoint_snapshots"][8][k], v)
    # a branch with a frozen first layer and a fresh AdamW optimizer from the same epoch; save / load / replay exactly
    mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}; mask["hidden_layers.0.weight"][:] = 0.0
    cfg_adam = ReLUTrainConfig(epochs=8, lr=1e-3, batch_size=100, seed=1, device="cpu", snapshot_epochs=(4, 6, 8), lr_scheduler_eta_min_ratio=0.1, optimizer="adamw", parameter_update_mask=mask)
    m4b = checkpoint_model_from_state(full["model"], full["checkpoint_snapshots"][4])
    adam = train_relu_mlp_with_optimizer_snapshots(model=m4b, x_train=x, y_train=y, config=cfg_adam, start_epoch=4)
    assert torch.equal(adam["checkpoint_snapshots"][8]["hidden_layers.0.weight"], full["checkpoint_snapshots"][4]["hidden_layers.0.weight"])   # first layer frozen
    sr = designed_run_seed_result(out=adam, train_config=cfg_adam, model_spec={"input_dim": 12, "hidden_dims": [6, 4], "bias": False}, data_spec=data_spec, tree=tree, tag="adam-branch")
    folder = save_designed_run(sr, tmp_path / "adam"); lo = load_designed_run(folder)
    assert {p.name for p in folder.iterdir()} == {"weights.npz", "optimizer_state.npz", "meta.pkl"}
    st = lo["out"]["optimizer_snapshots"][6][0]["state"][1]; assert {"exp_avg", "exp_avg_sq", "step"} <= set(st)
    rep = replay_interval_from_snapshot(lo, 6, data={"x_train": x, "y_train": y}, record_full_data_gradient=True, full_data_subsample=100, neuron_rows=[0, 1])
    assert rep["epoch_end"] == 8 and rep["verification"]["max_abs_param_diff_overall"] == 0.0 and rep["verification"]["max_abs_momentum_diff"] == 0.0
    assert np.all(rep["update_rows"] == 0.0)                                   # frozen first layer never moves
    # the SGD run still round-trips through the new format
    sr2 = designed_run_seed_result(out=full, train_config=cfg, model_spec={"input_dim": 12, "hidden_dims": [6, 4], "bias": False}, data_spec=data_spec, tree=tree, tag="sgd")
    lo2 = load_designed_run(save_designed_run(sr2, tmp_path / "sgd"))
    rep2 = replay_interval_from_snapshot(lo2, 2, data={"x_train": x, "y_train": y}); assert rep2["verification"]["max_abs_param_diff_overall"] == 0.0 and rep2["verification"]["max_abs_momentum_diff"] == 0.0


def test_rotation_lock_run_snapshots_save_load_and_replay_exactly(tmp_path) -> None:
    from layer_lenses.init_sweep import SweepRunSpec, build_data_chunked, build_tree
    from layer_lenses.relu_analysis import (designed_run_seed_result, load_designed_run, replay_interval_from_snapshot, save_designed_run,
                                            train_relu_mlp_with_optimizer_snapshots)
    from layer_lenses.relu_training import FirstLayerRotationLock
    tree = build_tree(SweepRunSpec(tree_seed=5, data_seed=0, init_seed=0, batch_size=1, dim=12, depth=2))
    x, y = build_data_chunked(SweepRunSpec(tree_seed=5, data_seed=7, init_seed=0, batch_size=1, dim=12, depth=2, n_train=600), tree, chunk_size=600)
    data_spec = {"kind": "sweep_chunked", "tree_seed": 5, "data_seed": 7, "n_train": 600, "chunk_size": 600, "eval_seed": 99, "n_eval": 50, "dim": 12, "depth": 2}
    base_cfg = dict(epochs=8, lr=0.5, batch_size=100, seed=1, device="cpu", snapshot_epochs=tuple(range(0, 9)), lr_scheduler_eta_min_ratio=0.1, optimizer="sgd", momentum=0.9)
    torch.manual_seed(0); model = ReLUMLP(input_dim=12, hidden_dims=[6, 4], bias=False); init_state = copy.deepcopy(model.state_dict())
    free = train_relu_mlp_with_optimizer_snapshots(model=model, x_train=x, y_train=y, config=ReLUTrainConfig(**base_cfg))
    # pick a threshold that one row crosses in the middle of the free run, so the locked run has a genuine lock event
    def max_cos(state):
        W = state["hidden_layers.0.weight"].numpy(); return np.abs(W / np.linalg.norm(W, axis=1, keepdims=True) @ tree.w_list.T).max(1)
    c = np.stack([max_cos(free["checkpoint_snapshots"][e]) for e in range(9)])          # (epochs, rows)
    row = int((c.max(0) - c[0]).argmax()); thr = float(0.5 * (c[0, row] + c[:, row].max()))
    assert c[:, row].max() > c[0, row] + 1e-3
    model.load_state_dict(init_state)
    cfg = ReLUTrainConfig(**base_cfg, first_layer_rotation_lock=FirstLayerRotationLock(targets=tree.w_list, threshold=thr, rows=tuple(range(6))))
    out = train_relu_mlp_with_optimizer_snapshots(model=model, x_train=x, y_train=y, config=cfg)
    rl = out["rotation_lock_snapshots"]; assert sorted(rl) == list(range(9)) and out["rotation_lock_final"]["locked"][row]
    lock_step = int(out["rotation_lock_final"]["lock_step"][row]); assert 0 < lock_step < 48
    e_lock = lock_step // 6 + 1                                                           # first snapshot epoch at which the row is locked
    assert not rl[e_lock - 1]["locked"][row] or lock_step % 6 == 0
    d = out["rotation_lock_final"]["direction"][row]
    for e in range(e_lock, 9):                                                            # locked: direction fixed to 1e-6, norm free
        w = out["checkpoint_snapshots"][e]["hidden_layers.0.weight"].numpy()[row]
        assert abs(abs(w @ d) / np.linalg.norm(w) - 1) < 1e-6
    sr = designed_run_seed_result(out=out, train_config=cfg, model_spec={"input_dim": 12, "hidden_dims": [6, 4], "bias": False}, data_spec=data_spec, tree=tree, tag="lock")
    folder = save_designed_run(sr, tmp_path / "lock"); assert (folder / "rotation_lock.npz").exists()
    lo = load_designed_run(folder)
    for e in range(9):
        assert np.array_equal(lo["out"]["rotation_lock_snapshots"][e]["locked"], rl[e]["locked"])
        assert np.array_equal(lo["out"]["rotation_lock_snapshots"][e]["lock_step"], rl[e]["lock_step"])
        assert np.array_equal(lo["out"]["rotation_lock_snapshots"][e]["direction"], rl[e]["direction"])
    # replay the interval that contains the lock event (and the one after it): bitwise exact, lock state reproduced
    for e0 in (max(0, e_lock - 1), e_lock):
        rep = replay_interval_from_snapshot(lo, e0, data={"x_train": x, "y_train": y}, neuron_rows=[row])
        assert rep["verification"]["max_abs_param_diff_overall"] == 0.0 and rep["verification"]["max_abs_momentum_diff"] == 0.0 and rep["verification"]["locked_rows_mismatch"] == 0
        assert rep["locked_rows"].shape == (6, 1) and int(rep["lock_step"][row]) == lock_step
    rep = replay_interval_from_snapshot(lo, e_lock, data={"x_train": x, "y_train": y}, neuron_rows=[row])
    assert rep["locked_rows"].all()
    W0 = lo["out"]["checkpoint_snapshots"][e_lock]["hidden_layers.0.weight"].numpy()[row]
    for upd in rep["update_rows"][:, 0]:                                                  # every update of a locked row is radial
        assert np.linalg.norm(upd - (upd @ d) * d) < 1e-6 * max(1.0, np.linalg.norm(W0))


def test_step_anatomy_helpers_replay_curvature_and_ablation(tmp_path) -> None:
    from layer_lenses.init_sweep import SweepRunSpec, build_data_chunked, build_tree
    from layer_lenses.relu_analysis import (designed_run_seed_result, hessian_quadratic_form, layerwise_update_ablation, loss_along_update, loss_on_points,
                                            replay_interval_from_snapshot, replay_steps_full, top_hessian_eigenvalue, train_relu_mlp_with_optimizer_snapshots, _flat)
    tree = build_tree(SweepRunSpec(tree_seed=5, data_seed=0, init_seed=0, batch_size=1, dim=12, depth=2))
    x, y = build_data_chunked(SweepRunSpec(tree_seed=5, data_seed=7, init_seed=0, batch_size=1, dim=12, depth=2, n_train=600), tree, chunk_size=600)
    data_spec = {"kind": "sweep_chunked", "tree_seed": 5, "data_seed": 7, "n_train": 600, "chunk_size": 600, "eval_seed": 99, "n_eval": 50, "dim": 12, "depth": 2}
    torch.manual_seed(0); model = ReLUMLP(input_dim=12, hidden_dims=[6, 4], bias=False)
    mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}; mask["hidden_layers.0.weight"][2:4] = 0.0
    cfg = ReLUTrainConfig(epochs=6, lr=0.3, batch_size=100, seed=1, device="cpu", snapshot_epochs=(0, 2, 4, 6), lr_scheduler_eta_min_ratio=0.1, optimizer="sgd", momentum=0.9, parameter_update_mask=mask)
    out = train_relu_mlp_with_optimizer_snapshots(model=model, x_train=x, y_train=y, config=cfg)
    sr = designed_run_seed_result(out=out, train_config=cfg, model_spec={"input_dim": 12, "hidden_dims": [6, 4], "bias": False}, data_spec=data_spec, tree=tree, tag="t")
    d = {"x_train": x, "y_train": y}
    R = replay_steps_full(sr, 2, step_from=3, step_to=9, data=d, probe_points=200)          # steps 3..8 of the 12-step interval 2 -> 4
    assert R["step"].tolist() == [3, 4, 5, 6, 7, 8] and R["epoch"].tolist() == [2, 2, 2, 3, 3, 3] and len(R["params_before"]) == 6
    rep = replay_interval_from_snapshot(sr, 2, data=d)                                          # same steps seen by the row-level replay
    for i, t in enumerate(R["step"]):
        assert np.allclose(R["updates"][i]["hidden_layers.0.weight"], rep["update_rows"][t]) and np.allclose(R["grads"][i]["hidden_layers.0.weight"], rep["grad_rows"][t])
        assert np.isclose(R["batch_loss"][i], rep["batch_loss"][t])
    for i in range(5):                                                                           # params_before chain through the updates
        for nm in R["parameter_names"]:
            assert np.allclose(R["params_before"][i][nm] + R["updates"][i][nm], R["params_before"][i + 1][nm], atol=1e-6)
    assert np.all(R["updates"][0]["hidden_layers.0.weight"][2:4] == 0.0) and set(R["momentum_before"][0]) == set(R["parameter_names"])
    tmpl = sr["out"]["model"]; p, u = R["params_before"][0], R["updates"][0]
    la = loss_along_update(tmpl, p, u, x[:200], y[:200], alphas=[0.0, 1.0])
    assert np.isclose(la.loss[0], R["probe_loss"][0]) and np.isclose(la.loss[1], R["probe_loss"][1]) and np.isclose(la.loss[0], loss_on_points(tmpl, p, x[:200], y[:200]))
    ab = layerwise_update_ablation(tmpl, p, u, x[:200], y[:200]); assert ab.applied.tolist()[:2] == ["none", "all"] and len(ab) == 2 + 2 * len(R["parameter_names"]) and np.isclose(ab.loss[1], la.loss[1])
    q = hessian_quadratic_form(tmpl, p, u, x[:200], y[:200]); assert np.isclose(q["loss"], loss_on_points(tmpl, p, x[:200], y[:200])) and np.isclose(q["direction_norm"], np.linalg.norm(_flat(u, R["parameter_names"])))
    ev = top_hessian_eigenvalue(tmpl, p, x[:200], y[:200], iters=40, mask=mask)
    assert ev["lambda_max"] >= q["curvature"] - 1e-4 and abs(sum(ev["layer_fraction"].values()) - 1) < 1e-5 and np.all(ev["eigenvector"]["hidden_layers.0.weight"][2:4] == 0.0)
    assert abs(hessian_quadratic_form(tmpl, p, ev["eigenvector"], x[:200], y[:200])["curvature"] - ev["lambda_max"]) < 1e-3 * max(1.0, ev["lambda_max"])
    # the curvature along the top eigenvector matches the finite-difference second derivative of the loss along it (float32 loss, so a step of 0.02)
    v = ev["eigenvector"]; f = lambda a: loss_on_points(tmpl, {k: p[k] + a * v[k] for k in p}, x[:200], y[:200]); h = 0.02
    fd = (f(h) - 2 * f(0.0) + f(-h)) / h ** 2
    assert abs(ev["lambda_max"] - fd) < 0.15 * max(1.0, abs(fd))
    with pytest.raises(ValueError):
        replay_steps_full(sr, 2, step_from=5, step_to=5, data=d)


def test_scalar_adam_run_saves_loads_and_replays_exactly(tmp_path) -> None:
    from layer_lenses.init_sweep import SweepRunSpec, build_data_chunked, build_tree
    from layer_lenses.relu_analysis import designed_run_seed_result, load_designed_run, replay_interval_from_snapshot, save_designed_run, train_relu_mlp_with_optimizer_snapshots
    tree = build_tree(SweepRunSpec(tree_seed=5, data_seed=0, init_seed=0, batch_size=1, dim=12, depth=2))
    x, y = build_data_chunked(SweepRunSpec(tree_seed=5, data_seed=7, init_seed=0, batch_size=1, dim=12, depth=2, n_train=600), tree, chunk_size=600)
    data_spec = {"kind": "sweep_chunked", "tree_seed": 5, "data_seed": 7, "n_train": 600, "chunk_size": 600, "eval_seed": 99, "n_eval": 50, "dim": 12, "depth": 2}
    torch.manual_seed(0); model = ReLUMLP(input_dim=12, hidden_dims=[6, 4], bias=False)
    mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}; mask["hidden_layers.0.weight"][2:4] = 0.0
    cfg = ReLUTrainConfig(epochs=8, lr=0.01, batch_size=100, seed=1, device="cpu", snapshot_epochs=(0, 2, 4, 6, 8), lr_scheduler_eta_min_ratio=0.1, optimizer="scalar_adam", adam_betas=(0.9, 0.999), parameter_update_mask=mask)
    out = train_relu_mlp_with_optimizer_snapshots(model=model, x_train=x, y_train=y, config=cfg)
    assert out["optimizer_group_names"] == ["scalar_adam"] and len(out["checkpoint_snapshots"]) == 5
    sr = designed_run_seed_result(out=out, train_config=cfg, model_spec={"input_dim": 12, "hidden_dims": [6, 4], "bias": False}, data_spec=data_spec, tree=tree, tag="scalar")
    lo = load_designed_run(save_designed_run(sr, tmp_path / "scalar"))
    st = lo["out"]["optimizer_snapshots"][4][0]["state"]; assert {"exp_avg", "exp_avg_sq", "step"} <= set(st[0]) and st[0]["exp_avg_sq"].ndim == 0 and float(st[0]["step"]) == 24
    for e0 in (0, 2, 6):
        rep = replay_interval_from_snapshot(lo, e0, data={"x_train": x, "y_train": y}, neuron_rows=[0, 2])
        assert rep["verification"]["max_abs_param_diff_overall"] == 0.0 and rep["verification"]["max_abs_momentum_diff"] == 0.0 and np.all(rep["update_rows"][:, 1] == 0.0)
    # every step moves the parameters by at most ~lr (normalised step) once the normaliser is warm
    rep = replay_interval_from_snapshot(lo, 6, data={"x_train": x, "y_train": y}); W = lo["out"]["checkpoint_snapshots"]
    assert np.linalg.norm(rep["update_rows"].reshape(rep["num_steps"], -1), axis=1).max() <= 1.05 * out["lr_history"][6]


def test_replay_steps_full_reports_sgd_equivalent_rate_for_scalar_adam() -> None:
    from layer_lenses.init_sweep import SweepRunSpec, build_data_chunked, build_tree
    from layer_lenses.relu_analysis import designed_run_seed_result, replay_steps_full, train_relu_mlp_with_optimizer_snapshots
    tree = build_tree(SweepRunSpec(tree_seed=5, data_seed=0, init_seed=0, batch_size=1, dim=12, depth=2))
    x, y = build_data_chunked(SweepRunSpec(tree_seed=5, data_seed=7, init_seed=0, batch_size=1, dim=12, depth=2, n_train=600), tree, chunk_size=600)
    data_spec = {"kind": "sweep_chunked", "tree_seed": 5, "data_seed": 7, "n_train": 600, "chunk_size": 600, "eval_seed": 99, "n_eval": 50, "dim": 12, "depth": 2}
    for kind, extra in (("scalar_adam", dict(adam_betas=(0.8, 0.99))), ("sgd", dict(momentum=0.7)), ("adamw", {})):
        torch.manual_seed(0); model = ReLUMLP(input_dim=12, hidden_dims=[6, 4], bias=False)
        cfg = ReLUTrainConfig(epochs=4, lr=0.02, batch_size=100, seed=1, device="cpu", snapshot_epochs=(0, 2, 4), lr_scheduler_eta_min_ratio=1.0, optimizer=kind, **extra)
        out = train_relu_mlp_with_optimizer_snapshots(model=model, x_train=x, y_train=y, config=cfg)
        sr = designed_run_seed_result(out=out, train_config=cfg, model_spec={"input_dim": 12, "hidden_dims": [6, 4], "bias": False}, data_spec=data_spec, tree=tree, tag=kind)
        R = replay_steps_full(sr, 2, step_from=2, step_to=6, data={"x_train": x, "y_train": y}, probe_points=100)
        assert R["optimizer"] == kind and len(R["optimizer_state_before"]) == 4
        if kind == "sgd":
            assert R["beta1"] == 0.7 and np.allclose(R["lr_sgd_equiv"], 0.02) and set(R["optimizer_state_before"][0]["hidden_layers.0.weight"]) == {"momentum_buffer"}
        elif kind == "scalar_adam":
            assert R["beta1"] == 0.8 and set(R["optimizer_state_before"][0]["hidden_layers.0.weight"]) == {"step", "exp_avg", "exp_avg_sq"}
            st = R["optimizer_state_before"][1]["hidden_layers.0.weight"]; t = float(st["step"]) + 1; g2 = sum((g ** 2).sum() for g in R["grads"][1].values())
            v_hat = (0.99 * float(st["exp_avg_sq"]) + 0.01 * g2) / (1 - 0.99 ** t)
            # the update equals -lr_eq * (beta1 * m_prev + (1 - beta1) g) / (1 - beta1) with the true v (the reported rate uses v before this step's |g|^2 is added, hence approx)
            names = R["parameter_names"]; mprev = np.concatenate([R["optimizer_state_before"][1][n]["exp_avg"].ravel() for n in names]); g = np.concatenate([R["grads"][1][n].ravel() for n in names])
            upd = np.concatenate([R["updates"][1][n].ravel() for n in names]); pred = -0.02 * (0.8 * mprev + 0.2 * g) / ((1 - 0.8 ** t) * (np.sqrt(v_hat) + 1e-8))
            assert np.allclose(upd, pred, atol=1e-6)
            lr_eq_true = 0.02 * 0.2 / ((1 - 0.8 ** t) * (np.sqrt(v_hat) + 1e-8)); assert abs(R["lr_sgd_equiv"][1] / lr_eq_true - 1) < 0.05
        else:
            assert R["beta1"] == 0.9 and np.isnan(R["lr_sgd_equiv"]).all()


def test_plot_layered_relu_graph_layer_orders_rearranges_positions_only() -> None:
    from layer_lenses.relu_analysis import _layer_y_positions
    from layer_lenses.init_sweep import SweepRunSpec, build_tree
    tree = build_tree(SweepRunSpec(tree_seed=5, data_seed=0, init_seed=0, batch_size=1, dim=6, depth=2))
    torch.manual_seed(0); model = ReLUMLP(input_dim=6, hidden_dims=[5, 3, 4], bias=False); sd = model.state_dict()
    order3 = [0, 2, 1, 3]
    _, edges_a, fig_a, ax_a = plot_layered_relu_graph(state_dict=sd, tree=tree, depth=2, epoch=0, edge_abs_threshold=0.0, figsize=(4, 4))
    _, edges_b, fig_b, ax_b = plot_layered_relu_graph(state_dict=sd, tree=tree, depth=2, epoch=0, edge_abs_threshold=0.0, figsize=(4, 4), layer_orders={2: order3})
    assert edges_a == edges_b                                                   # same edges, only drawn at other heights
    # the third hidden layer's labels (texts "0".."3" at x = 1.6 * 2) sit at the slots given by the order
    ys = _layer_y_positions(4); x3 = 1.6 * 2
    pos = {t.get_text(): t.get_position()[1] for t in ax_b.texts if abs(t.get_position()[0] - x3) < 1e-9 and t.get_text() in {"0", "1", "2", "3"}}
    assert all(abs(pos[str(n)] - ys[slot]) < 1e-9 for slot, n in enumerate(order3))
    pos_a = {t.get_text(): t.get_position()[1] for t in ax_a.texts if abs(t.get_position()[0] - x3) < 1e-9 and t.get_text() in {"0", "1", "2", "3"}}
    assert all(abs(pos_a[str(n)] - ys[n]) < 1e-9 for n in range(4))
    with pytest.raises(ValueError):
        plot_layered_relu_graph(state_dict=sd, tree=tree, depth=2, epoch=0, figsize=(4, 4), layer_orders={2: [0, 0, 1, 2]})
    plt.close(fig_a); plt.close(fig_b)
