import numpy as np
import pytest
import torch

from layer_lenses.designed_init import (
    GROUP_ORDER,
    PROBE,
    WIDTH,
    DesignedInitSpec,
    build_designed_model,
    build_designed_weights,
    first_layer_index,
    first_layer_labels,
    neuron_trajectory_table,
    odt_coordinates,
    run_designed_flow,
    sign_flip_variants,
)
from layer_lenses.init_sweep import SweepRunSpec, build_data, build_tree


def _tree():
    return build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1, dim=100, depth=5))


def test_first_layer_layout_and_norms() -> None:
    tree = _tree()
    spec = DesignedInitSpec(eps=0.01, seed=3)
    W = build_designed_weights(spec, tree)
    W1 = W["hidden_layers.0.weight"]
    assert W1.shape == (WIDTH, 100)
    assert np.allclose(np.linalg.norm(W1, axis=1), 0.01)
    U = tree.w_list
    # probe is orthogonal to every ODT normal; level-4 rows are exactly +-u_j
    assert np.abs(U @ W1[PROBE]).max() < 1e-12
    for node in range(15, 31):
        assert np.allclose(W1[first_layer_index(node, -1)], -0.01 * U[node])
        assert np.allclose(W1[first_layer_index(node, +1)], +0.01 * U[node])
    labels = first_layer_labels()
    assert len(labels) == WIDTH and labels[0] == "probe" and labels[1] == "-u15" and labels[32] == "+u30"
    assert labels[33].startswith("rand")
    coords, off = odt_coordinates(W1, tree)
    assert off[PROBE] == pytest.approx(0.01, rel=1e-9) and np.allclose(off[1:33], 0, atol=1e-12)


def test_second_layer_groups_wire_the_right_neurons() -> None:
    tree = _tree()
    for mode in ("random", "all"):
        W2 = build_designed_weights(DesignedInitSpec(subset_mode=mode, seed=1), tree)["hidden_layers.1.weight"]
        assert set(np.unique(W2)) <= {-1.0, 0.0, 1.0}
        expected_sign = {"A": 1, "B": 1, "C": -1, "D": -1}
        allowed = {
            "A": {first_layer_index(n, -1) for n in range(15, 23)},
            "B": {first_layer_index(n, +1) for n in range(23, 31)},
            "C": {first_layer_index(n, -1) for n in range(23, 31)},
            "D": {first_layer_index(n, +1) for n in range(15, 23)},
        }
        for g, name in enumerate(GROUP_ORDER):
            rows = W2[g * 10:(g + 1) * 10]
            assert (rows[:, PROBE] == expected_sign[name]).all()
            others = {j for j in np.nonzero(rows[:, 1:].any(axis=0))[0] + 1}
            assert others <= allowed[name] and len(others) >= 1
            assert (rows[:, 1:] >= 0).all() and (rows[:, 1:].sum(axis=1) >= 1).all()
            if mode == "all":
                assert others == allowed[name] and (rows[:, sorted(allowed[name])] == 1).all()
    flipped = build_designed_weights(DesignedInitSpec(group_sign=(-1, 1, 1, -1)), tree)["hidden_layers.1.weight"]
    assert (flipped[0:10, PROBE] == -1).all() and (flipped[20:30, PROBE] == 1).all()


def test_third_layer_output_and_model_roundtrip() -> None:
    tree = _tree()
    spec = DesignedInitSpec(layer3_density=0.5, output_scale=2.0, seed=2)
    W = build_designed_weights(spec, tree)
    W3 = W["hidden_layers.2.weight"]
    assert set(np.unique(W3)) <= {0.0, 1.0} and 0.3 < W3.mean() < 0.7
    assert np.all(W["output_layer.weight"] == 2.0) and W["output_layer.weight"].shape == (1, WIDTH)
    model = build_designed_model(spec, tree)
    for k, v in model.state_dict().items():
        assert np.allclose(v.numpy(), W[k], atol=1e-7)
    assert len(sign_flip_variants()) == 16 and (1, 1, -1, -1) in sign_flip_variants()


def test_flow_runner_snapshots_and_trajectory_table() -> None:
    tree = _tree()
    spec = DesignedInitSpec(eps=1e-3, seed=0)
    x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=11, init_seed=0, batch_size=1, n_train=2000), tree)
    model = build_designed_model(spec, tree)
    out = run_designed_flow(model, x, y, lr=1e-4, steps=6, snapshot_every=3, device="cpu")
    assert sorted(out["checkpoint_snapshots"]) == [0, 3, 6] and len(out["epoch_losses"]) == 6
    assert len(out["lr_history"]) == 6 and all(l == pytest.approx(1e-4) for l in out["lr_history"])
    t = neuron_trajectory_table(out, tree)
    assert list(t.step) == [0, 3, 6] and t.loc[0, "off_span_fraction"] == pytest.approx(1.0, abs=1e-9)
    assert abs(t.loc[0, "cos_u0"]) < 1e-6 and t.loc[0, "norm"] == pytest.approx(1e-3, rel=1e-4)
