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


def test_two_probe_design_structure_and_linear_cues() -> None:
    from layer_lenses.designed_init import (
        CueGroup, MultiProbeSpec, build_multiprobe_model, build_multiprobe_weights, cue_index,
        multiprobe_labels, two_probe_spec, LEFT_NEG, RIGHT_NEG,
    )
    tree = _tree()
    spec = two_probe_spec(cue_mode="relu", eps=0.01, seed=4, layer3="identity", output_scale=20.0)
    W = build_multiprobe_weights(spec, tree)
    W1, W2, W3, Wo = (W[k] for k in ("hidden_layers.0.weight", "hidden_layers.1.weight", "hidden_layers.2.weight", "output_layer.weight"))
    assert W1.shape == (40, 100) and np.allclose(np.linalg.norm(W1, axis=1), 0.01)
    assert np.abs(tree.w_list @ W1[0]).max() < 1e-12 and np.abs(tree.w_list @ W1[1]).max() < 1e-12
    assert abs(W1[0] @ W1[1]) / (np.linalg.norm(W1[0]) * np.linalg.norm(W1[1])) < 0.4  # independent random probes
    assert np.allclose(W1[cue_index(15, -1, 2)], -0.01 * tree.w_list[15]) and np.allclose(W1[cue_index(30, 1, 2)], 0.01 * tree.w_list[30])
    labels = multiprobe_labels(spec)
    assert labels[:2] == ["probe0", "probe1"] and labels[2] == "-u15" and labels[-1] == "rand5" and len(labels) == 40
    # groups: rows 0-19 belong to probe 0, 20-39 to probe 1, all probe weights -1, no cross-wiring
    assert (W2[0:20, 0] == -1).all() and (W2[0:20, 1] == 0).all()
    assert (W2[20:40, 1] == -1).all() and (W2[20:40, 0] == 0).all()
    assert set(np.unique(W2)) <= {-1.0, 0.0, 1.0}
    c_rows = W2[0:10]; assert (c_rows[:, [cue_index(n, -1, 2) for n in range(23, 31)]] == 1).all()
    assert np.allclose(W3, np.eye(40)) and (Wo == 20.0).all()
    # linear cue mode puts -1 on the opposite-sign twin
    lin = build_multiprobe_weights(two_probe_spec(cue_mode="linear", seed=4), tree)["hidden_layers.1.weight"]
    assert (lin[0:10, [cue_index(n, -1, 2) for n in range(23, 31)]] == 1).all()
    assert (lin[0:10, [cue_index(n, +1, 2) for n in range(23, 31)]] == -1).all()
    model = build_multiprobe_model(spec, tree)
    assert all(np.allclose(v.numpy(), W[k], atol=1e-7) for k, v in model.state_dict().items())
    with pytest.raises(ValueError):
        build_multiprobe_weights(MultiProbeSpec(groups=(CueGroup(probe=2, probe_weight=-1, cues=LEFT_NEG),), n_probes=2), tree)
    with pytest.raises(ValueError):
        build_multiprobe_weights(MultiProbeSpec(groups=(CueGroup(probe=0, probe_weight=-1, cues=RIGHT_NEG, size=41),), n_probes=2), tree)


def test_level4_descendants_and_node_probe_groups() -> None:
    from layer_lenses.designed_init import (
        LEFT_NEG, LEFT_POS, RIGHT_NEG, RIGHT_POS, SIX_PROBE_TARGETS, level4_descendants,
        node_probe_groups, six_probe_spec, build_multiprobe_weights, cue_index,
    )
    assert level4_descendants(0) == list(range(15, 31))
    assert level4_descendants(1) == list(range(15, 23)) and level4_descendants(2) == list(range(23, 31))
    assert level4_descendants(3) == [15, 16, 17, 18] and level4_descendants(6) == [27, 28, 29, 30]
    assert level4_descendants(20) == [20]
    # root reproduces the C/D and A/B pairs
    plus = node_probe_groups(0, 0, +1); minus = node_probe_groups(0, 0, -1)
    assert plus[0].cues == RIGHT_NEG and plus[1].cues == LEFT_POS
    assert minus[0].cues == LEFT_NEG and minus[1].cues == RIGHT_POS
    # node 1, pole +1: negative-side cues under node 4 (19-22), positive-side under node 3 (15-18)
    g = node_probe_groups(2, 1, +1)
    assert g[0].cues == tuple((-1, n) for n in (19, 20, 21, 22)) and g[1].cues == tuple((+1, n) for n in (15, 16, 17, 18))
    assert all(gg.probe == 2 and gg.probe_weight == -1.0 for gg in g)
    with pytest.raises(ValueError):
        node_probe_groups(0, 15, +1)

    tree = _tree()
    spec = six_probe_spec(eps=1e-3, seed=1)
    assert spec.n_probes == 6 and sum(gr.size for gr in spec.groups) == 40
    W = build_multiprobe_weights(spec, tree)
    W1, W2 = W["hidden_layers.0.weight"], W["hidden_layers.1.weight"]
    assert np.abs(tree.w_list @ W1[:6].T).max() < 1e-12          # six probes orthogonal to every ODT normal
    assert np.allclose(np.linalg.norm(W1, axis=1), 1e-3) and W1.shape[0] == 6 + 32 + 2
    # each row of layer 2 reads exactly one probe with -1; rows for probe p only read probe p
    probe_cols = W2[:, :6]
    assert ((probe_cols == -1).sum(axis=1) == 1).all() and (probe_cols[probe_cols != -1] == 0).all()
    owner = np.argmax(probe_cols == -1, axis=1)
    assert list(np.bincount(owner)) == [8, 8, 6, 6, 6, 6]
    # the +u_1 probe (index 2) never reads a cue outside node 1's subtree
    rows2 = W2[owner == 2]
    outside = [cue_index(n, s, 6) for n in range(23, 31) for s in (-1, 1)]
    assert (rows2[:, outside] == 0).all() and rows2[:, cue_index(19, -1, 6)].max() == 1


def test_flow_runner_can_freeze_cue_rows() -> None:
    from layer_lenses.designed_init import build_multiprobe_model, cue_rows, two_probe_spec
    tree = _tree()
    x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=11, init_seed=0, batch_size=1, n_train=2000), tree)
    spec = two_probe_spec(eps=1e-3, seed=0)
    rows = cue_rows(spec)
    assert rows == list(range(2, 34))
    out = run_designed_flow(build_multiprobe_model(spec, tree), x, y, lr=1e-3, steps=4, snapshot_every=4, device="cpu",
                            freeze_first_layer_rows=rows)
    W0 = out["checkpoint_snapshots"][0]["hidden_layers.0.weight"]; W4 = out["checkpoint_snapshots"][4]["hidden_layers.0.weight"]
    assert torch.equal(W0[rows], W4[rows])
    assert not torch.equal(W0[:2], W4[:2])
