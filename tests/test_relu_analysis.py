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
