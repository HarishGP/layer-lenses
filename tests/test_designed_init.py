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


def test_hierarchy_design_structure() -> None:
    from layer_lenses.designed_init import (
        HIERARCHY_TARGETS, HierarchyDesignSpec, build_hierarchy_model, build_hierarchy_weights,
        hierarchy_cue_index, hierarchy_probe_table, hierarchy_unit_cues,
    )
    tree = _tree()
    assert len(HIERARCHY_TARGETS) == 30 and HIERARCHY_TARGETS[0] == (0, 1) and HIERARCHY_TARGETS[-1] == (14, -1)
    left, right = hierarchy_unit_cues(0, +1)
    assert left == tuple((+1, n) for n in range(15, 23)) and right == tuple((-1, n) for n in range(23, 31))
    assert hierarchy_unit_cues(7, -1) == (((-1, 15),), ((+1, 16),))
    spec = HierarchyDesignSpec(eps=0.01, seed=3, cue_mode="linear", subset_mode="random")
    W, meta = build_hierarchy_weights(spec, tree)
    W1, W2, W3, Wo = (W[k] for k in ("hidden_layers.0.weight", "hidden_layers.1.weight", "hidden_layers.2.weight", "output_layer.weight"))
    assert W1.shape == (80, 100) and W2.shape == (40, 80) and W3.shape == (40, 40) and Wo.shape == (1, 40)
    assert np.allclose(np.linalg.norm(W1, axis=1), 0.01)
    assert np.abs(tree.w_list @ W1[:30].T).max() < 1e-12                 # probes off the ODT span
    c = tree.w_list @ W1[30:62].T / 0.01                                    # cues: ODT part plus a 0.2 off-span component
    assert np.allclose(np.abs(c).max(axis=0), 1 / np.sqrt(1 + 0.2**2), atol=1e-6)
    assert spec.n_random == 18 and meta["cue_rows"] == list(range(30, 62))
    # every unit reads exactly one probe with -1 * layer2_scale; counts 4/2/1/1 by level
    probe_cols = W2[:, :30]
    assert ((probe_cols != 0).sum(axis=1) == 1).all() and np.allclose(probe_cols[probe_cols != 0], -spec.layer2_scale)
    counts = np.bincount(meta["unit_probe"], minlength=30)
    assert list(counts[:2]) == [4, 4] and list(counts[2:6]) == [2] * 4 and list(counts[6:]) == [1] * 24
    # a level-3 unit in linear mode reads its two cues and their twins: +1 on (-1,15), -1 on (+1,15), +1 on (+1,16), -1 on (-1,16)
    u = int(np.nonzero(meta["unit_probe"] == 14 + 1)[0][0])               # probe 15 -> (7, -1)
    assert W2[u, hierarchy_cue_index(15, -1, spec)] > 0 and W2[u, hierarchy_cue_index(15, +1, spec)] < 0
    assert W2[u, hierarchy_cue_index(16, +1, spec)] > 0 and W2[u, hierarchy_cue_index(16, -1, spec)] < 0
    assert set(np.unique(np.round(W3 / spec.layer3_scale, 9))) <= {0.0, 1.0}
    model, _ = build_hierarchy_model(spec, tree)
    t = hierarchy_probe_table(model.hidden_layers[0].weight.detach().numpy(), tree)
    assert len(t) == 30 and (t.cos_target.abs() < 1e-6).all() and (t.norm - 0.01).abs().max() < 1e-6
    with pytest.raises(ValueError):
        build_hierarchy_weights(HierarchyDesignSpec(units_per_level=(8, 8, 8, 8)), tree)


def test_train_sweep_run_accepts_a_model_and_freezes_rows() -> None:
    from layer_lenses.designed_init import HierarchyDesignSpec, build_hierarchy_model
    from layer_lenses.init_sweep import train_sweep_run
    tree = _tree()
    x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=11, init_seed=0, batch_size=1, n_train=1024), tree)
    model, meta = build_hierarchy_model(HierarchyDesignSpec(eps=0.01, seed=0), tree)
    sp = SweepRunSpec(tree_seed=5178, data_seed=11, init_seed=0, batch_size=256, n_train=1024, hidden_dims=(80, 40, 40),
                      lr=1e-3, momentum=0.9, device="cpu", snapshot_every_steps=4, max_steps=8, stop_loss=0.0, store_all_layers=True)
    out = train_sweep_run(sp, tree=tree, x_train=x, y_train=y, model=model, freeze_first_layer_rows=meta["cue_rows"])
    W0, W8 = out["weights"]["hidden_layers.0.weight"][0], out["weights"]["hidden_layers.0.weight"][-1]
    assert out["final_step"] == 8 and np.array_equal(W0[30:62], W8[30:62]) and not np.array_equal(W0[:30], W8[:30])
    assert out["weights"]["hidden_layers.1.weight"].shape[1:] == (40, 80)


def test_train_sweep_run_unfreezes_rows_at_a_step() -> None:
    from layer_lenses.designed_init import HierarchyDesignSpec, build_hierarchy_model
    from layer_lenses.init_sweep import train_sweep_run
    tree = _tree()
    x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=11, init_seed=0, batch_size=1, n_train=1024), tree)
    model, meta = build_hierarchy_model(HierarchyDesignSpec(eps=0.01, seed=0), tree)
    sp = SweepRunSpec(tree_seed=5178, data_seed=11, init_seed=0, batch_size=256, n_train=1024, hidden_dims=(80, 40, 40),
                      lr=1e-3, momentum=0.9, device="cpu", snapshot_every_steps=4, max_steps=12, stop_loss=0.0, store_all_layers=True)
    out = train_sweep_run(sp, tree=tree, x_train=x, y_train=y, model=model, freeze_first_layer_rows=meta["cue_rows"], unfreeze_at_step=8)
    W = out["weights"]["hidden_layers.0.weight"]; steps = list(out["snapshot_steps"])
    i0, i8, i12 = steps.index(0), steps.index(8), steps.index(12)
    assert np.array_equal(W[i0][30:62], W[i8][30:62])          # frozen through step 8
    assert not np.array_equal(W[i8][30:62], W[i12][30:62])     # moving afterwards
    assert not np.array_equal(W[i0][:30], W[i8][:30])


def test_standard_scale_spec_normalizes_rows_and_keeps_signs() -> None:
    from layer_lenses.designed_init import HierarchyDesignSpec, build_hierarchy_weights, standard_row_norms, standard_scale_spec
    tree = _tree()
    base = HierarchyDesignSpec(seed=2, cue_mode="linear")
    assert np.allclose(standard_row_norms("default", base), 1 / np.sqrt(3))
    assert np.allclose(standard_row_norms("kaiming", base), np.sqrt(2))
    xa = standard_row_norms("xavier", base)
    assert xa[0] == pytest.approx(np.sqrt(200 / 180)) and xa[3] == pytest.approx(np.sqrt(80 / 41))
    with pytest.raises(ValueError):
        standard_row_norms("lecun", base)
    Wb, _ = build_hierarchy_weights(base, tree)
    spec = standard_scale_spec("default", seed=2, cue_mode="linear")
    Ws, _ = build_hierarchy_weights(spec, tree)
    for k in Wb:
        norms = np.linalg.norm(Ws[k], axis=1)
        assert np.allclose(norms[norms > 0], 1 / np.sqrt(3))
        assert np.array_equal(np.sign(Ws[k]), np.sign(Wb[k]))        # sign structure untouched
    assert np.abs(tree.w_list @ Ws["hidden_layers.0.weight"][:30].T).max() < 1e-9


def test_hierarchy_layer3_exact_count_and_all_cues_widths() -> None:
    from layer_lenses.designed_init import HierarchyDesignSpec, build_hierarchy_weights
    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1))
    spec = HierarchyDesignSpec(width1=62, width2=30, width3=40, units_per_level=(1, 1, 1, 1), subset_mode="all",
                               cue_mode="linear", layer3_exact=True, seed=0)
    assert spec.n_random == 0
    w, meta = build_hierarchy_weights(spec, tree)
    W1, W2, W3 = w["hidden_layers.0.weight"], w["hidden_layers.1.weight"], w["hidden_layers.2.weight"]
    assert W1.shape == (62, 100) and W2.shape == (30, 62) and W3.shape == (40, 30)
    assert np.all((W3 != 0).sum(axis=1) == 15)
    assert list(meta["unit_probe"]) == list(range(30))
    # unit for the root probe reads every level-4 pair (16 pairs x 2 signs) plus its probe
    assert (W2[0] != 0).sum() == 1 + 32
    # unit for a level-3 probe reads its two level-4 children (2 pairs x 2 signs) plus its probe
    assert (W2[29] != 0).sum() == 1 + 4
    dense = build_hierarchy_weights(HierarchyDesignSpec(width1=62, width2=30, units_per_level=(1, 1, 1, 1), subset_mode="all", seed=0), tree)[0]
    assert not np.all((dense["hidden_layers.2.weight"] != 0).sum(axis=1) == 15)


def test_hierarchy_balanced_output_signs() -> None:
    from layer_lenses.designed_init import HierarchyDesignSpec, build_hierarchy_weights, standard_scale_spec
    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1))
    w = build_hierarchy_weights(HierarchyDesignSpec(width1=62, width2=30, units_per_level=(1, 1, 1, 1), subset_mode="all", output_signs="balanced"), tree)[0]
    out = w["output_layer.weight"].ravel()
    assert out.shape == (40,) and (out > 0).sum() == 20 and (out < 0).sum() == 20
    assert np.allclose(np.abs(out), np.abs(out[0]))
    ks = build_hierarchy_weights(standard_scale_spec("kaiming", width1=62, width2=30, units_per_level=(1, 1, 1, 1), subset_mode="all", output_signs="balanced"), tree)[0]
    ko = ks["output_layer.weight"].ravel()
    assert (ko < 0).sum() == 20 and np.isclose(np.linalg.norm(ko), np.sqrt(2.0))
    ones = build_hierarchy_weights(HierarchyDesignSpec(width1=62, width2=30, units_per_level=(1, 1, 1, 1), subset_mode="all"), tree)[0]["output_layer.weight"].ravel()
    assert (ones > 0).all()
    with pytest.raises(ValueError):
        build_hierarchy_weights(HierarchyDesignSpec(width1=62, width2=30, units_per_level=(1, 1, 1, 1), subset_mode="all", output_signs="random"), tree)


def test_probe_node_outcomes_and_run_summary_on_a_constructed_first_layer() -> None:
    from layer_lenses.designed_init import HIERARCHY_TARGETS, hierarchy_run_summary, probe_node_outcomes, plot_probe_target_cosines
    from matplotlib import pyplot as plt
    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1))
    U = tree.w_list
    rng = np.random.default_rng(0)
    W1 = rng.normal(size=(62, 100)) * 0.01          # unaligned filler
    for p, (node, pole) in enumerate(HIERARCHY_TARGETS):
        W1[p] = pole * U[node]                       # every probe exactly on target
    W1[1] = -U[0] * -1                               # probe 1 (-u_0) at the opposite pole (+u_0)
    W1[3] = U[20]                                    # probe 3 (-u_1) captured by a level-4 node
    W1[40] = U[1]                                    # a cue row sits on u_1
    po = probe_node_outcomes(W1, tree)
    assert len(po) == 15
    n0, n1, n2 = po.iloc[0], po.iloc[1], po.iloc[2]
    assert n0.own == 2 and n0.intended and n0.opposite
    assert n1.own == 1 and n1.intended and not n1.opposite and n1.others == 1 and n1.probe_minus.startswith("+u20")
    assert n2.own == 2 and n2.intended and not n2.opposite
    s = hierarchy_run_summary(W1, tree)
    assert s["own_probe_nodes"] == 15 and s["intended_pole"] == 15 and s["opposite_pole_only"] == 0
    assert s["missed_by_own_probes"] == [] and 1 not in s["both_poles"] and 0 in s["both_poles"]
    assert s["nodes_covered"] == 16 and s["missing_nodes"] == [n for n in range(15, 31) if n != 20]
    W1b = W1.copy(); W1b[[0, 1]] = W1b[[0, 1]] @ np.eye(100) * 0 + rng.normal(size=(2, 100))
    s2 = hierarchy_run_summary(W1b, tree)
    assert s2["missed_by_own_probes"] == [0] and s2["own_probe_nodes"] == 14
    fig, axes = plot_probe_target_cosines(np.stack([W1, W1b]), np.array([0, 10]), tree, title="t")
    assert len(axes) == 4
    plt.close(fig)


def test_path_sign_cues_restore_first_order_push_under_balanced_output() -> None:
    import torch
    from layer_lenses.designed_init import (HIERARCHY_TARGETS, build_hierarchy_model, build_hierarchy_weights, hierarchy_unit_cues,
                                            hierarchy_cue_index, standard_scale_spec)
    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1))
    base = dict(width1=80, width2=30, width3=40, units_per_level=(1, 1, 1, 1), subset_mode="all", cue_mode="linear", layer3_exact=True, cue_offspan_ratio=0.2, seed=0, output_signs="balanced")
    plain = standard_scale_spec("kaiming", **base)
    aware = standard_scale_spec("kaiming", **base, path_sign_cues=True, layer3_nonzero_paths=True)
    w0, m0 = build_hierarchy_weights(plain, tree)
    w1, m1 = build_hierarchy_weights(aware, tree)
    W3 = w1["hidden_layers.2.weight"]
    assert np.all((W3 != 0).sum(axis=1) == 15)                      # row counts preserved by the repair
    assert m1["zero_path_units"] == [] and len(m0["zero_path_units"]) > 0
    assert len(m1["flipped_units"]) > 0 and m0["flipped_units"] == []
    assert np.array_equal(w1["hidden_layers.0.weight"], w0["hidden_layers.0.weight"])
    W2 = w1["hidden_layers.1.weight"]
    for u, (node, pole) in enumerate(HIERARCHY_TARGETS):
        assert W2[u, u] < 0                                          # probe weight untouched
        eff = pole * int(m1["unit_path_sign"][u])
        left, right = hierarchy_unit_cues(node, eff)
        for s, n in left + right:                                    # cue signs follow the effective pole
            assert np.sign(W2[u, hierarchy_cue_index(n, s, aware)]) > 0
            assert np.sign(W2[u, hierarchy_cue_index(n, -s, aware)]) < 0
    # first-order push at init: -grad of the population loss on each probe row vs its designed target
    x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=11, init_seed=0, batch_size=1, n_train=100_000), tree)
    xt, yt = torch.from_numpy(x).float(), torch.from_numpy(y).float()
    def pushes(spec):
        model, _ = build_hierarchy_model(spec, tree); model.zero_grad()
        torch.nn.functional.softplus(-yt * model(xt).reshape(-1)).mean().backward()
        g = -model.hidden_layers[0].weight.grad.numpy()
        return np.array([pole * (g[p] @ tree.w_list[node]) / (np.linalg.norm(g[p]) * np.linalg.norm(tree.w_list[node]) + 1e-30) for p, (node, pole) in enumerate(HIERARCHY_TARGETS)])
    c_plain, c_aware = pushes(plain), pushes(aware)
    assert (c_aware > 0).sum() >= 27 and (c_aware > 0).sum() > (c_plain > 0).sum()
    with pytest.raises(ValueError):
        build_hierarchy_weights(standard_scale_spec("kaiming", **{**base, "layer3_exact": False}, layer3_nonzero_paths=True), tree)


def test_cue_block_scale_equalises_probe_entries_across_levels() -> None:
    from layer_lenses.designed_init import HIERARCHY_TARGETS, NODE_LEVEL, build_hierarchy_weights, standard_scale_spec
    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1))
    base = dict(width1=80, width2=30, width3=40, units_per_level=(1, 1, 1, 1), subset_mode="all", cue_mode="linear", layer3_exact=True, cue_offspan_ratio=0.2, seed=0,
                output_signs="balanced", path_sign_cues=True, layer3_nonzero_paths=True)
    w_none = build_hierarchy_weights(standard_scale_spec("kaiming", **base), tree)[0]["hidden_layers.1.weight"]
    w_eq = build_hierarchy_weights(standard_scale_spec("kaiming", **base, cue_block_scale="inv_sqrt"), tree)[0]["hidden_layers.1.weight"]
    probe_none = np.array([abs(w_none[u, u]) for u in range(30)]); probe_eq = np.array([abs(w_eq[u, u]) for u in range(30)])
    assert probe_none[0] < probe_none[29] * 0.5                       # root probe entry much smaller without the scaling
    assert np.allclose(probe_eq, 1.0, atol=1e-6)                     # equal (and = cue-block norm) with it
    for u in range(30):
        cue = np.delete(w_eq[u], u)
        assert np.isclose(np.linalg.norm(cue), 1.0, atol=1e-6) and np.isclose(np.linalg.norm(w_eq[u]), np.sqrt(2), atol=1e-6)
        assert np.array_equal(np.sign(cue), np.sign(np.delete(w_none[u], u)))   # signs (incl. path-sign flips) unchanged
    with pytest.raises(ValueError):
        build_hierarchy_weights(standard_scale_spec("kaiming", **base, cue_block_scale="sqrt"), tree)


def test_probe_gradient_stats_fractions_and_direction() -> None:
    from layer_lenses.designed_init import HIERARCHY_TARGETS, level4_descendants, probe_gradient_stats
    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); U = tree.w_list
    rng = np.random.default_rng(0)
    W1 = rng.normal(size=(62, 100)); G = np.zeros((62, 100))
    G[0] = -(3 * U[0] + 4 * U[15])                       # push = 3 u_0 + 4 u_15 for probe 0 (+u_0; 15 is a descendant)
    G[3] = -(-U[1] + U[29])                             # push = -u_1 + u_29 for probe 3 (-u_1; 29 is not a descendant of 1)
    st = probe_gradient_stats(G, W1, tree)
    r0, r3 = st.iloc[0], st.iloc[3]
    assert r0.target == "+u0" and np.isclose(r0.cos_push_target, 0.6) and np.isclose(r0.frac_target, 0.36) and np.isclose(r0.frac_desc_l4, 0.64)
    assert np.isclose(r0.frac_offspan, 0.0, atol=1e-9) and r0.push_top == "+u15" and np.isclose(r0.grad_norm, 5.0)
    assert r3.target == "-u1" and np.isclose(r3.cos_push_target, np.sqrt(0.5)) and np.isclose(r3.frac_other_l4, 0.5) and np.isclose(r3.frac_desc_l4, 0.0)
    assert 29 not in level4_descendants(1)
    fr = st[["frac_target", "frac_desc_l4", "frac_other_l4", "frac_other_internal", "frac_offspan"]].iloc[[0, 3]].sum(axis=1)
    assert np.allclose(fr, 1.0)


def test_layer3_balanced_paths_gives_equal_alternating_unit_paths() -> None:
    from layer_lenses.designed_init import HierarchyDesignSpec, build_hierarchy_weights, standard_scale_spec
    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1))
    base = dict(width1=80, width2=30, width3=40, units_per_level=(1, 1, 1, 1), subset_mode="all", cue_mode="linear", cue_offspan_ratio=0.2, seed=0,
                output_signs="balanced", path_sign_cues=True, cue_block_scale="inv_sqrt", layer3_mode="balanced_paths")
    w, meta = build_hierarchy_weights(standard_scale_spec("kaiming", **base), tree)
    W3, wout = w["hidden_layers.2.weight"], w["output_layer.weight"].ravel()
    assert np.all((W3 != 0).sum(axis=1) == 15) and np.all((W3 != 0).sum(axis=0) == 20)
    path = W3.T @ wout; quantum = abs(W3[W3 != 0][0]) * abs(wout[0])
    assert np.allclose(np.abs(path), 10 * quantum) and np.array_equal(np.sign(path), np.where(np.arange(30) % 2 == 0, 1, -1))
    assert meta["zero_path_units"] == [] and meta["flipped_units"] == list(range(1, 30, 2))
    pos = wout > 0
    for j in range(15):
        a, b = 2 * j, 2 * j + 1
        assert (W3[pos, a] != 0).sum() == 15 and (W3[pos, b] != 0).sum() == 5 and (W3[~pos, a] != 0).sum() == 5 and (W3[~pos, b] != 0).sum() == 15
        assert not np.any((W3[:, a] != 0) & (W3[:, b] != 0))          # complementary subsets within a pair
    with pytest.raises(ValueError):
        build_hierarchy_weights(HierarchyDesignSpec(width1=62, width2=30, units_per_level=(1, 1, 1, 1), subset_mode="all", layer3_mode="balanced_paths"), tree)
    with pytest.raises(ValueError):
        build_hierarchy_weights(HierarchyDesignSpec(width1=62, width2=30, units_per_level=(1, 1, 1, 1), subset_mode="all", output_signs="balanced", layer3_mode="balanced_paths", layer3_majority=10), tree)


def test_probe_scale_applies_after_the_row_rescale() -> None:
    from layer_lenses.designed_init import build_hierarchy_weights, standard_scale_spec
    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1))
    base = dict(width1=80, width2=30, width3=40, units_per_level=(1, 1, 1, 1), subset_mode="all", cue_mode="linear", cue_offspan_ratio=0.2, seed=0,
                output_signs="balanced", path_sign_cues=True, layer3_mode="balanced_paths", cue_block_scale="inv_sqrt")
    w1 = build_hierarchy_weights(standard_scale_spec("kaiming", **base), tree)[0]
    w2 = build_hierarchy_weights(standard_scale_spec("kaiming", **base, probe_scale=0.01), tree)[0]
    A, B = w1["hidden_layers.0.weight"], w2["hidden_layers.0.weight"]
    assert np.allclose(np.linalg.norm(A, axis=1), np.sqrt(2))
    assert np.allclose(B[:30], 0.01 * A[:30]) and np.allclose(np.linalg.norm(B[:30], axis=1), 0.01 * np.sqrt(2))
    assert np.array_equal(B[30:], A[30:])
    for k in ("hidden_layers.1.weight", "hidden_layers.2.weight", "output_layer.weight"):
        assert np.array_equal(w1[k], w2[k])


def test_node_lineage_and_probe_deviation_lineage() -> None:
    from layer_lenses.designed_init import HIERARCHY_TARGETS, lineage_chance, node_ancestors, node_lineage, probe_deviation_lineage
    from layer_lenses.init_sweep import SweepRunSpec, build_tree

    assert node_ancestors(0) == [] and node_ancestors(2) == [0] and node_ancestors(16) == [7, 3, 1, 0]
    assert node_lineage(3, 3)["relation"] == "same"
    assert node_lineage(3, 1) == {"relation": "ancestor", "level_gap": -1, "side": "-"}         # node 3 is the left (negative-side) child of node 1
    assert node_lineage(4, 1) == {"relation": "ancestor", "level_gap": -1, "side": "+"}
    assert node_lineage(1, 16) == {"relation": "descendant", "level_gap": 3, "side": "-"}       # 16 is under 7, under 3, the negative child of 1
    assert node_lineage(1, 20) == {"relation": "descendant", "level_gap": 3, "side": "+"}       # 20 is under 9, under 4, the positive child of 1
    assert node_lineage(3, 4)["relation"] == "unrelated" and node_lineage(7, 16)["relation"] == "descendant" and node_lineage(7, 30)["relation"] == "unrelated"
    assert lineage_chance(0) == 1.0 and lineage_chance(1) == 15 / 30 and lineage_chance(3) == 8 / 30 and lineage_chance(7) == 5 / 30
    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1))
    W1 = np.zeros((62, 100)); W1[:30] = np.array([pole * tree.w_list[node] for node, pole in HIERARCHY_TARGETS])
    W1[2] = -tree.w_list[0]; W1[14] = tree.w_list[16]; W1[8] = 0.5 * tree.w_list[4] + 0.5 * tree.w_list[1]           # a deviation up, one down (16 = 2*7+2 is the positive-side child of 7), one undecided
    t = probe_deviation_lineage(W1, tree)
    assert len(t) == 30 and t.deviated.sum() == 2 and t.relation[t.probe == 2].item() == "ancestor" and t.relation[t.probe == 14].item() == "descendant"
    assert t.side[t.probe == 14].item() == "+" and t.level_gap[t.probe == 14].item() == 1 and t.relation[t.probe == 8].item() == "none" and not t.captured[t.probe == 8].item()
    assert t.end_cos[t.probe == 2].item() < -0.99 and (t[~t.deviated & t.captured].relation == "same").all()


def test_leaf_wrong_side_detectors_and_unit_leaf_anatomy() -> None:
    from layer_lenses.designed_init import leaf_wrong_side_detectors, unit_leaf_anatomy
    from layer_lenses.init_sweep import SweepRunSpec, build_tree

    assert leaf_wrong_side_detectors(0) == [(0, 1), (1, 1), (3, 1), (7, 1), (15, 1)]           # leftmost leaf: every split went left (w.x <= 0), so +u is the wrong side
    assert leaf_wrong_side_detectors(31) == [(0, -1), (2, -1), (6, -1), (14, -1), (30, -1)]
    assert leaf_wrong_side_detectors(30) == [(0, -1), (2, -1), (6, -1), (14, -1), (30, 1)]
    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); U = tree.w_list
    rng = np.random.default_rng(0); x = rng.standard_normal((4000, 100)); x /= np.linalg.norm(x, axis=1, keepdims=True)
    # first layer: +-u_j detectors for all 31 nodes (62 rows); a unit that vetoes leaf 30's four internal wrong-side detectors and reads -u30 positively
    W1 = np.array([s * U[j] for j in range(31) for s in (1, -1)]); row = {(j, s): 2 * j + (0 if s > 0 else 1) for j in range(31) for s in (1, -1)}
    W2 = np.zeros((2, 62)); 
    for n, s in leaf_wrong_side_detectors(30)[:4]: W2[0, row[(n, s)]] = -5.0
    W2[0, row[(30, -1)]] = 0.1; W2[1, row[(0, 1)]] = 1.0                                    # unit 1: fires on the whole right half
    W3 = np.ones((3, 2)); Wout = np.array([1.0, -1.0, 2.0])
    t = unit_leaf_anatomy(W1, W2, W3, Wout, tree, x)
    assert t.top_leaf[0] == 30 and t.share[0] > 0.9 and t.vetoes_on_chain[0] == 4 and t.veto_levels[0] == "0123" and t.positive_inputs[0] == "-u30:0.10" and abs(t.active[0] - 1 / 32) < 0.02
    assert abs(t.active[1] - 0.5) < 0.05 and t.n_veto[1] == 0 and t.vetoes_on_chain[1] <= 1 and t.path[1] == 2.0


def test_mirror_node_and_tree_flip_matrix_are_label_symmetries() -> None:
    from layer_lenses.designed_init import mirror_node, subtree_nodes, tree_flip_matrix
    from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data

    assert [mirror_node(n, 0) for n in (0, 1, 2, 3, 4, 5, 6, 15, 23, 30)] == [0, 2, 1, 5, 6, 3, 4, 23, 15, 22]
    assert [mirror_node(n, 1) for n in (0, 1, 2, 3, 4, 7, 9, 15, 19, 22)] == [0, 1, 2, 4, 3, 9, 7, 19, 15, 18]
    assert subtree_nodes(7) == [15, 16] and subtree_nodes(3) == [7, 8, 15, 16, 17, 18]
    assert all(mirror_node(mirror_node(n, j), j) == n for j in range(15) for n in range(31))
    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1))
    x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=11, init_seed=0, batch_size=1, n_train=4000), tree)
    from layer_lenses.odt import _traverse_tree
    def labels(pts):
        leaves = _traverse_tree(pts @ tree.w_list.T + tree.b_list, depth=5); return tree.leaf_labels[leaves - 31]
    for j in (0, 1, 6, 14):
        Q = tree_flip_matrix(tree, j)
        assert np.allclose(Q @ Q.T, np.eye(100), atol=1e-10)
        assert np.allclose(Q @ tree.w_list[j], -tree.w_list[j]) and np.allclose(Q @ tree.w_list[2 * j + 1], tree.w_list[2 * j + 2])
        assert np.array_equal(labels(x @ Q.T), labels(x)) and np.array_equal(labels(x), y)


def test_equivariant_quadruple_init_is_fixed_by_every_flip_and_the_30_unit_design_is_not() -> None:
    from layer_lenses.designed_init import (build_hierarchy_model, standard_scale_spec, equivariant_quadruple_weights, flip_equivariance_report, QUAD_TYPES)
    from layer_lenses.init_sweep import SweepRunSpec, build_tree

    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); U = tree.w_list
    design = dict(seed=0, cue_mode="linear", subset_mode="all", units_per_level=(1, 1, 1, 1), width1=80, width2=30, width3=40, cue_offspan_ratio=0.2,
                  output_signs="balanced", path_sign_cues=True, layer3_mode="balanced_paths", layer3_majority=15, cue_block_scale="inv_sqrt")
    model, meta = build_hierarchy_model(standard_scale_spec("kaiming", **design), tree)
    W1 = model.hidden_layers[0].weight.detach().numpy().astype(np.float64).copy(); W2 = model.hidden_layers[1].weight.detach().numpy().astype(np.float64)
    W3 = model.hidden_layers[2].weight.detach().numpy().astype(np.float64); Wout = model.output_layer.weight.detach().numpy().astype(np.float64)
    v = U[15:31].sum(0)
    for r in meta["cue_rows"]:                                                   # run 26's cues
        c = W1[r] / np.linalg.norm(W1[r]) @ U.T; m = int(np.abs(c).argmax()); pole = int(np.sign(c[m])); new = pole * (v + U[m]); W1[r] = new / np.linalg.norm(new) * np.linalg.norm(W1[r])
    q = equivariant_quadruple_weights(W1, W2, W3, Wout, cue_rows=meta["cue_rows"], tree=tree, orth_seed=3)
    rows = q["probe_rows"] + q["spare_rows"]; R = q["W1"][rows]
    assert np.abs(R @ U.T).max() < 1e-12 and np.allclose(R @ R.T, np.diag(np.linalg.norm(W1[rows], axis=1) ** 2), atol=1e-10)   # orthogonal, off-span, norms kept
    assert q["W2"].shape == (60, 80) and q["W3"].shape == (40, 60) and (q["W3"] != 0).sum(1).tolist() == [30] * 40
    for jn in range(15):
        for t, (bsign, s) in enumerate(QUAD_TYPES):
            u = 4 * jn + t; probe = 2 * jn + (0 if bsign * s > 0 else 1)
            assert q["W2"][u, probe] == -1 and np.array_equal(q["W2"][u, meta["cue_rows"]], bsign * W2[2 * jn, meta["cue_rows"]])
            assert np.sign(q["W3"][:, u] @ Wout.ravel()) == s
    for jn in range(15):
        rep = flip_equivariance_report(q["W1"], q["W2"], q["W3"], q["Wout"], tree, jn, cue_rows=meta["cue_rows"], units_per_node=4)
        assert max(rep.values()) < 1e-6, (jn, rep)                                           # float32 normals: orthonormal to ~1e-7
    # the 30-unit design with the same orthogonal rows fails only through its cue blocks (the chirality) and W3's random pattern
    W1o = W1.copy(); W1o[rows] = q["W1"][rows]
    rep = flip_equivariance_report(W1o, W2, W3, Wout, tree, 0, cue_rows=meta["cue_rows"], units_per_node=2)
    assert rep["in_span"] < 1e-6 and rep["off_span_gram"] < 1e-6 and rep["w2"] > 0.1
    W3c = np.stack([W3[:, 0] if u % 2 == 0 else W3[:, 1] for u in range(30)], axis=1)      # common pattern: W3 passes, W2 still fails
    rep = flip_equivariance_report(W1o, W2, W3c, Wout, tree, 0, cue_rows=meta["cue_rows"], units_per_node=2)
    assert rep["w3"] < 1e-6 and rep["w2"] > 0.1


def test_symmetric_coordinates_vanish_on_the_equivariant_init_and_flag_breaking() -> None:
    from layer_lenses.designed_init import build_hierarchy_model, standard_scale_spec, equivariant_quadruple_weights, symmetric_coordinates, symmetric_class_summary
    from layer_lenses.init_sweep import SweepRunSpec, build_tree

    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); U = tree.w_list
    design = dict(seed=0, cue_mode="linear", subset_mode="all", units_per_level=(1, 1, 1, 1), width1=80, width2=30, width3=40, cue_offspan_ratio=0.2,
                  output_signs="balanced", path_sign_cues=True, layer3_mode="balanced_paths", layer3_majority=15, cue_block_scale="inv_sqrt")
    model, meta = build_hierarchy_model(standard_scale_spec("kaiming", **design), tree)
    W1 = model.hidden_layers[0].weight.detach().numpy().astype(np.float64).copy(); W2 = model.hidden_layers[1].weight.detach().numpy().astype(np.float64)
    W3 = model.hidden_layers[2].weight.detach().numpy().astype(np.float64); Wout = model.output_layer.weight.detach().numpy().astype(np.float64)
    v = U[15:31].sum(0)
    for r in meta["cue_rows"]:
        c = W1[r] / np.linalg.norm(W1[r]) @ U.T; m = int(np.abs(c).argmax()); pole = int(np.sign(c[m])); new = pole * (v + U[m]); W1[r] = new / np.linalg.norm(new) * np.linalg.norm(W1[r])
    q = equivariant_quadruple_weights(W1, W2, W3, Wout, cue_rows=meta["cue_rows"], tree=tree, orth_seed=1)
    co = symmetric_coordinates(q["W1"], tree, cue_rows=meta["cue_rows"]); summ = symmetric_class_summary(co)
    assert co.residual.max() < 1e-6 and set(summ.cls) == {"probe_L0", "probe_L1", "probe_L2", "probe_L3", "cue_+", "cue_-", "spare"}
    assert summ.filter(like="_spread").abs().max().max() < 1e-6
    cue_plus = summ[summ.cls == "cue_+"].iloc[0]; assert cue_plus.own > 0 and abs(cue_plus.own - 2 * cue_plus.unrel4_b0) < 1e-6 and abs(cue_plus.anc_1) < 1e-6   # (v + u_m): own 2x the others, no ancestors
    assert (co[co.kind == "probe"].in_span_norm < 1e-9).all() and abs(summ[summ.cls == "spare"].iloc[0].lvl4) < 1e-9
    # the original design's random spare rows (in-span parts up to norm 1) are off the symmetric subspace; so is a probe with a component on an unrelated internal normal
    co0 = symmetric_coordinates(W1, tree, cue_rows=meta["cue_rows"]); assert co0[co0.kind == "spare"].residual.max() > 0.1 and co0[co0.kind == "probe"].in_span_norm.max() < 1e-6
    W1p = q["W1"].copy(); W1p[4] += 0.3 * U[2] + 0.2 * U[7]                      # probe of node 2 (+): u_2 is its own normal (free), u_7 is unrelated internal (must vanish)
    cop = symmetric_coordinates(W1p, tree, cue_rows=meta["cue_rows"]); assert abs(cop.residual[4] - 0.2) < 1e-6 and abs(cop.own[4] - 0.3) < 1e-6 and cop.residual.drop(4).max() < 1e-6
    W1a = q["W1"].copy(); W1a[8] += 0.5 * U[1] * (-1)                            # probe of node 4 (+): ancestor u_1, node 4 = 2*1 + 2 is on u_1's positive side -> anc_1 = -0.5
    coa = symmetric_coordinates(W1a, tree, cue_rows=meta["cue_rows"]); assert abs(coa.anc_1[8] + 0.5) < 1e-6 and coa.residual[8] < 1e-6


def test_reduced_third_layer_with_gradient_scales_reproduces_the_40_neuron_dynamics() -> None:
    """Four third-layer neurons with multiplicities folded into the output weights and gradient scales (m on the output weight, 1/m on
    the W3 row) train exactly like the 40 identical-copy neurons of the equivariant design."""
    import torch
    from layer_lenses.designed_init import build_hierarchy_model, standard_scale_spec, equivariant_quadruple_weights
    from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data
    from layer_lenses.relu_mlp import ReLUMLP
    from layer_lenses.relu_training import ReLUTrainConfig
    from layer_lenses.relu_analysis import train_relu_mlp_with_optimizer_snapshots

    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); U = tree.w_list
    x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=3, init_seed=0, batch_size=1, n_train=4000), tree)
    design = dict(seed=0, cue_mode="linear", subset_mode="all", units_per_level=(1, 1, 1, 1), width1=80, width2=30, width3=40, cue_offspan_ratio=0.2,
                  output_signs="balanced", path_sign_cues=True, layer3_mode="balanced_paths", layer3_majority=15, cue_block_scale="inv_sqrt")
    model, meta = build_hierarchy_model(standard_scale_spec("kaiming", **design), tree)
    W1 = model.hidden_layers[0].weight.detach().numpy().astype(np.float64).copy(); W2 = model.hidden_layers[1].weight.detach().numpy().astype(np.float64)
    W3 = model.hidden_layers[2].weight.detach().numpy().astype(np.float64); Wout = model.output_layer.weight.detach().numpy().astype(np.float64); v = U[15:31].sum(0)
    for r in meta["cue_rows"]:
        c = W1[r] / np.linalg.norm(W1[r]) @ U.T; m = int(np.abs(c).argmax()); pole = int(np.sign(c[m])); new = pole * (v + U[m]); W1[r] = new / np.linalg.norm(new) * np.linalg.norm(W1[r])
    q = equivariant_quadruple_weights(W1, W2, W3, Wout, cue_rows=meta["cue_rows"], tree=tree, orth_seed=0)
    q["W1"] = q["W1"][:62]; q["W2"] = q["W2"][:, :62]
    inS = q["W3"][:, 0] != 0; wo = q["Wout"].ravel(); classes = [inS & (wo > 0), inS & (wo < 0), ~inS & (wo > 0), ~inS & (wo < 0)]
    mult = np.array([int(c.sum()) for c in classes]); reps = [int(np.nonzero(c)[0][0]) for c in classes]
    W3r = np.stack([q["W3"][r] for r in reps]); Wor = (mult * np.array([wo[r] for r in reps]))[None, :]
    def train(dims, W3_, Wo_, mask_extra):
        m = ReLUMLP(input_dim=100, hidden_dims=dims, bias=False)
        m.load_state_dict({"hidden_layers.0.weight": torch.as_tensor(q["W1"], dtype=torch.float32), "hidden_layers.1.weight": torch.as_tensor(q["W2"], dtype=torch.float32),
                           "hidden_layers.2.weight": torch.as_tensor(W3_, dtype=torch.float32), "output_layer.weight": torch.as_tensor(Wo_, dtype=torch.float32)})
        mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in m.named_parameters()}; mask.update(mask_extra)
        cfg = ReLUTrainConfig(epochs=6, lr=0.1, batch_size=1000, seed=7, device="cpu", snapshot_epochs=(0, 6), lr_scheduler_eta_min_ratio=1.0, optimizer="sgd", momentum=0.9, parameter_update_mask=mask, show_progress=False)
        out = train_relu_mlp_with_optimizer_snapshots(model=m, x_train=x, y_train=y, config=cfg); return {k: v.numpy().astype(np.float64) for k, v in out["checkpoint_snapshots"][6].items()}, np.array(out["epoch_losses"])
    full, loss_full = train([62, 60, 40], q["W3"], q["Wout"], {})
    red, loss_red = train([62, 60, 4], W3r, Wor, {"hidden_layers.2.weight": np.repeat(1.0 / mult[:, None], 60, axis=1).astype(np.float32), "output_layer.weight": mult[None, :].astype(np.float32)})
    assert np.allclose(loss_full, loss_red, rtol=1e-4) and np.abs(loss_full - 0.69).max() > 1e-3     # the same losses, and training happened
    assert np.abs(full["hidden_layers.0.weight"] - red["hidden_layers.0.weight"]).max() < 3e-5 and np.abs(full["hidden_layers.1.weight"] - red["hidden_layers.1.weight"]).max() < 3e-5
    for k, r in enumerate(reps):
        assert np.abs(full["hidden_layers.2.weight"][classes[k]] - red["hidden_layers.2.weight"][k]).max() < 3e-5            # every copy equals the reduced row
        assert abs(mult[k] * full["output_layer.weight"][0, r] - red["output_layer.weight"][0, k]) < 3e-4                   # output weight = m x the copy's
    assert np.abs(full["hidden_layers.2.weight"] - q["W3"]).max() > 5e-4 and np.abs(full["output_layer.weight"] - q["Wout"]).max() > 5e-4   # and the third layer moved
