import numpy as np
import pytest
import torch

from layer_lenses.relu_mlp import ReLUMLP
from layer_lenses.relu_training import (
    ReLUTrainConfig,
    build_relu_optimizers,
    evaluate_relu_mlp,
    normalized_optimizer_name,
    optimizer_group_names,
    relu_training_step,
    split_named_parameters_sgd_adam,
    suggest_sgd_lr_for_adam_parity,
    train_relu_mlp,
)


def test_train_relu_mlp_smoke() -> None:
    x = np.random.randn(128, 10).astype(np.float32)
    y = np.where(np.random.randn(128) > 0, 1, -1).astype(np.int8)
    model = ReLUMLP(input_dim=10, hidden_dims=[16, 16], bias=False)

    out = train_relu_mlp(
        model=model,
        x_train=x,
        y_train=y,
        config=ReLUTrainConfig(
            epochs=4,
            lr=1e-3,
            batch_size=32,
            seed=123,
            snapshot_epochs=(0, 2, 4),
        ),
    )
    assert len(out["epoch_losses"]) == 4
    assert set(out["checkpoint_snapshots"].keys()) == {0, 2, 4}


def test_relu_train_config_sgd_momentum_default_is_zero() -> None:
    assert ReLUTrainConfig().momentum == 0.0
    assert ReLUTrainConfig(optimizer="sgd").momentum == 0.0


def test_train_relu_mlp_sgd_smoke() -> None:
    x = np.random.randn(128, 10).astype(np.float32)
    y = np.where(np.random.randn(128) > 0, 1, -1).astype(np.int8)
    model = ReLUMLP(input_dim=10, hidden_dims=[16, 16], bias=False)

    out = train_relu_mlp(
        model=model,
        x_train=x,
        y_train=y,
        config=ReLUTrainConfig(
            epochs=4,
            lr=1e-2,
            batch_size=32,
            seed=456,
            snapshot_epochs=(0, 2, 4),
            optimizer="sgd",
        ),
    )
    assert len(out["epoch_losses"]) == 4
    assert set(out["checkpoint_snapshots"].keys()) == {0, 2, 4}


def test_train_relu_mlp_sgd_momentum_changes_updates() -> None:
    x = np.random.randn(128, 10).astype(np.float32)
    y = np.where(np.random.randn(128) > 0, 1, -1).astype(np.int8)
    model_plain = ReLUMLP(input_dim=10, hidden_dims=[16, 16], bias=False)
    model_momentum = ReLUMLP(input_dim=10, hidden_dims=[16, 16], bias=False)
    model_momentum.load_state_dict(model_plain.state_dict())

    shared_kwargs = dict(
        epochs=4,
        lr=1e-2,
        batch_size=32,
        seed=456,
        snapshot_epochs=(0, 4),
        optimizer="sgd",
    )
    out_plain = train_relu_mlp(
        model=model_plain,
        x_train=x,
        y_train=y,
        config=ReLUTrainConfig(**shared_kwargs, momentum=0.0),
    )
    out_momentum = train_relu_mlp(
        model=model_momentum,
        x_train=x,
        y_train=y,
        config=ReLUTrainConfig(**shared_kwargs, momentum=0.9),
    )

    w_plain = out_plain["checkpoint_snapshots"][4]["hidden_layers.0.weight"]
    w_momentum = out_momentum["checkpoint_snapshots"][4]["hidden_layers.0.weight"]
    assert not torch.allclose(w_plain, w_momentum)


def test_first_layer_gradient_projection_keeps_sgd_update_orthogonal() -> None:
    x = np.random.randn(128, 10).astype(np.float32)
    y = np.where(np.random.randn(128) > 0, 1, -1).astype(np.int8)
    model = ReLUMLP(input_dim=10, hidden_dims=[16, 16], bias=False)
    projection_vector = np.zeros(10, dtype=np.float32)
    projection_vector[0] = 1.0

    out = train_relu_mlp(
        model=model,
        x_train=x,
        y_train=y,
        config=ReLUTrainConfig(
            epochs=1,
            lr=1e-2,
            batch_size=x.shape[0],
            seed=789,
            snapshot_epochs=(0, 1),
            optimizer="sgd",
            first_layer_grad_orthogonal_to=projection_vector,
        ),
    )
    w0_start = out["checkpoint_snapshots"][0]["hidden_layers.0.weight"]
    w0_end = out["checkpoint_snapshots"][1]["hidden_layers.0.weight"]
    delta = w0_end - w0_start

    assert torch.allclose(delta @ torch.from_numpy(projection_vector), torch.zeros(16), atol=1e-7)


def test_parameter_update_mask_freezes_individual_entries() -> None:
    x = np.ones((64, 4), dtype=np.float32)
    y = np.ones(64, dtype=np.int8)
    model = ReLUMLP(input_dim=4, hidden_dims=[3], bias=False)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(0.1)

    mask = {name: torch.ones_like(param) for name, param in model.named_parameters()}
    mask["hidden_layers.0.weight"][0, 0] = 0

    out = train_relu_mlp(
        model=model,
        x_train=x,
        y_train=y,
        config=ReLUTrainConfig(
            epochs=1,
            lr=1e-1,
            batch_size=x.shape[0],
            seed=101,
            snapshot_epochs=(0, 1),
            optimizer="sgd",
            parameter_update_mask=mask,
        ),
    )
    w0_start = out["checkpoint_snapshots"][0]["hidden_layers.0.weight"]
    w0_end = out["checkpoint_snapshots"][1]["hidden_layers.0.weight"]

    assert w0_end[0, 0] == w0_start[0, 0]
    assert not torch.allclose(w0_end[0, 1], w0_start[0, 1])


def test_evaluate_relu_mlp_metrics_shape() -> None:
    x = np.random.randn(64, 10).astype(np.float32)
    y = np.where(np.random.randn(64) > 0, 1, -1).astype(np.int8)
    model = ReLUMLP(input_dim=10, hidden_dims=[16, 16], bias=False)
    with torch.no_grad():
        _ = model(torch.from_numpy(x))
    metrics = evaluate_relu_mlp(model=model, x_eval=x, y_eval=y)
    assert set(metrics.keys()) == {"log_loss", "zero_one_loss", "accuracy", "num_samples"}
    assert metrics["num_samples"] == 64


def test_sgd_adam_split_named_parameters_and_validation() -> None:
    model = ReLUMLP(input_dim=6, hidden_dims=[4, 4, 4], bias=True)

    sgd_named, adam_named = split_named_parameters_sgd_adam(model, 1)
    assert [n for n, _ in sgd_named] == ["hidden_layers.0.weight", "hidden_layers.0.bias"]
    assert [n for n, _ in adam_named][:2] == ["hidden_layers.1.weight", "hidden_layers.1.bias"]
    assert "output_layer.weight" in [n for n, _ in adam_named]

    sgd_named, adam_named = split_named_parameters_sgd_adam(model, 3)
    assert all(n.startswith("hidden_layers.") for n, _ in sgd_named)
    assert [n for n, _ in adam_named] == ["output_layer.weight", "output_layer.bias"]

    # Every parameter is assigned exactly once, for every valid split.
    all_names = {n for n, _ in model.named_parameters()}
    for split in (1, 2, 3):
        sgd_named, adam_named = split_named_parameters_sgd_adam(model, split)
        sgd_names = {n for n, _ in sgd_named}
        adam_names = {n for n, _ in adam_named}
        assert sgd_names | adam_names == all_names
        assert not sgd_names & adam_names

    for bad in (0, 4, -1):
        with pytest.raises(ValueError):
            split_named_parameters_sgd_adam(model, bad)
    with pytest.raises(TypeError):
        split_named_parameters_sgd_adam(model, 1.0)


def test_normalized_optimizer_name_and_group_names() -> None:
    assert normalized_optimizer_name(ReLUTrainConfig(optimizer="adam")) == "adamw"
    assert normalized_optimizer_name(ReLUTrainConfig(optimizer="AdamW")) == "adamw"
    assert normalized_optimizer_name(ReLUTrainConfig(optimizer="sgd")) == "sgd"
    assert normalized_optimizer_name(ReLUTrainConfig(optimizer="sgd_adam")) == "sgd_adam"
    with pytest.raises(ValueError):
        normalized_optimizer_name(ReLUTrainConfig(optimizer="rmsprop"))

    assert optimizer_group_names(ReLUTrainConfig(optimizer="adam")) == ["adamw"]
    assert optimizer_group_names(ReLUTrainConfig(optimizer="sgd_adam")) == ["sgd", "adamw"]


def test_train_relu_mlp_sgd_adam_smoke() -> None:
    x = np.random.randn(128, 10).astype(np.float32)
    y = np.where(np.random.randn(128) > 0, 1, -1).astype(np.int8)
    model = ReLUMLP(input_dim=10, hidden_dims=[16, 16], bias=False)

    out = train_relu_mlp(
        model=model,
        x_train=x,
        y_train=y,
        config=ReLUTrainConfig(
            epochs=4,
            lr=1e-3,
            batch_size=32,
            seed=7,
            snapshot_epochs=(0, 2, 4),
            optimizer="sgd_adam",
            sgd_adam_split_layer=1,
            sgd_lr=1.0,
            momentum=0.9,
        ),
    )
    assert len(out["epoch_losses"]) == 4
    assert set(out["checkpoint_snapshots"].keys()) == {0, 2, 4}
    assert set(out["lr_history_by_group"].keys()) == {"sgd", "adamw"}
    # Each group anneals from its own base LR, and lr_history tracks the AdamW part.
    assert out["lr_history_by_group"]["sgd"][0] == pytest.approx(1.0)
    assert out["lr_history_by_group"]["adamw"][0] == pytest.approx(1e-3)
    assert out["lr_history"] == out["lr_history_by_group"]["adamw"]
    assert out["lr_history_by_group"]["sgd"][-1] < out["lr_history_by_group"]["sgd"][0]


def test_sgd_adam_applies_sgd_to_early_layers_and_adam_to_later() -> None:
    torch.manual_seed(0)
    x = np.random.randn(64, 10).astype(np.float32)
    y = np.where(np.random.randn(64) > 0, 1, -1).astype(np.int8)
    model = ReLUMLP(input_dim=10, hidden_dims=[8, 8], bias=False)
    config = ReLUTrainConfig(
        optimizer="sgd_adam",
        sgd_adam_split_layer=1,
        lr=1e-3,
        sgd_lr=0.5,
        momentum=0.0,
        weight_decay=0.0,
        sgd_weight_decay=0.0,
    )
    optimizers = build_relu_optimizers(model, config)
    assert isinstance(optimizers[0], torch.optim.SGD)
    assert isinstance(optimizers[1], torch.optim.AdamW)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    loss = relu_training_step(
        model=model,
        optimizers=optimizers,
        criterion=torch.nn.BCEWithLogitsLoss(),
        x_batch=torch.from_numpy(x),
        y_batch_bin=torch.from_numpy((y + 1.0) / 2.0).float(),
        first_layer_projection_vector=None,
        parameter_update_masks={},
    )
    assert torch.isfinite(loss)
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    after = {n: p.detach().clone() for n, p in model.named_parameters()}

    # Layer 0 is plain SGD: the update is exactly -sgd_lr * grad.
    update0 = after["hidden_layers.0.weight"] - before["hidden_layers.0.weight"]
    assert torch.allclose(update0, -0.5 * grads["hidden_layers.0.weight"], atol=1e-7)

    # Layer 1 is AdamW at t=1: the update is the sign step -lr * g / (|g| + eps),
    # so its magnitude is ~lr per coordinate and it is NOT a multiple of the gradient.
    g1 = grads["hidden_layers.1.weight"]
    update1 = after["hidden_layers.1.weight"] - before["hidden_layers.1.weight"]
    assert torch.allclose(update1, -1e-3 * g1 / (g1.abs() + 1e-8), atol=1e-7)
    assert not torch.allclose(update1, -1e-3 * g1, atol=1e-7)

    # The optimizers hold the state types their algorithms imply. Note that SGD with
    # momentum=0 keeps no per-parameter state at all, so the layer-0 state is empty.
    named = dict(model.named_parameters())
    state0 = optimizers[0].state.get(named["hidden_layers.0.weight"], {})
    state1 = optimizers[1].state[named["hidden_layers.1.weight"]]
    assert "exp_avg" not in state0
    assert "exp_avg" in state1 and "exp_avg_sq" in state1


def test_sgd_adam_keeps_momentum_buffer_only_on_the_sgd_layers() -> None:
    torch.manual_seed(0)
    x = np.random.randn(64, 10).astype(np.float32)
    y = np.where(np.random.randn(64) > 0, 1, -1).astype(np.int8)
    model = ReLUMLP(input_dim=10, hidden_dims=[8, 8], bias=False)
    config = ReLUTrainConfig(
        optimizer="sgd_adam", sgd_adam_split_layer=1, lr=1e-3, sgd_lr=0.5, momentum=0.9
    )
    optimizers = build_relu_optimizers(model, config)
    relu_training_step(
        model=model,
        optimizers=optimizers,
        criterion=torch.nn.BCEWithLogitsLoss(),
        x_batch=torch.from_numpy(x),
        y_batch_bin=torch.from_numpy((y + 1.0) / 2.0).float(),
        first_layer_projection_vector=None,
        parameter_update_masks={},
    )
    named = dict(model.named_parameters())
    assert "momentum_buffer" in optimizers[0].state[named["hidden_layers.0.weight"]]
    assert "exp_avg" in optimizers[1].state[named["hidden_layers.1.weight"]]


def test_suggest_sgd_lr_for_adam_parity_matches_adam_step_scale() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(256, 10)).astype(np.float32)
    y = np.where(rng.normal(size=256) > 0, 1, -1).astype(np.int8)
    model = ReLUMLP(input_dim=10, hidden_dims=[8, 8], bias=False)
    config = ReLUTrainConfig(
        optimizer="sgd_adam", sgd_adam_split_layer=1, lr=2e-3, weight_decay=0.01
    )

    report = suggest_sgd_lr_for_adam_parity(
        model=model, x_train=x, y_train=y, config=config, num_samples=None
    )
    assert report["num_samples_used"] == 256
    assert report["adam_per_coordinate_step"] == pytest.approx(2e-3)
    assert report["suggested_sgd_lr"] == pytest.approx(
        2e-3 / report["median_abs_grad"], rel=1e-6
    )
    assert report["decay_fraction_per_step"] == pytest.approx(
        report["suggested_sgd_lr"] * 0.01, rel=1e-6
    )
    # The model must be left clean: no leftover gradients from the probe.
    assert all(p.grad is None for p in model.parameters())
