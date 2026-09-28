import numpy as np
import pandas as pd
import pytest

from layer_lenses.init_sweep import (
    FIRST_LAYER_KEY,
    SweepRunSpec,
    build_data,
    build_init_model,
    build_tree,
    capture_summary,
    first_layer_agreement,
    load_run,
    part_a_specs,
    part_b_specs,
    run_specs,
    save_run,
    train_sweep_run,
)


def _tiny(**kw) -> SweepRunSpec:
    base = dict(
        tree_seed=3, data_seed=4, init_seed=5, batch_size=64, dim=8, depth=2,
        n_train=512, hidden_dims=(6, 5), lr=0.05, momentum=0.9, device="cpu",
        snapshot_every_steps=4, max_steps=40, stop_loss=0.0,
    )
    base.update(kw)
    return SweepRunSpec(**base)


def test_tree_depends_only_on_tree_seed_and_data_on_data_seed() -> None:
    a, b = _tiny(data_seed=4), _tiny(data_seed=9)
    ta, tb = build_tree(a), build_tree(b)
    assert np.array_equal(ta.w_list, tb.w_list)
    xa, ya = build_data(a, ta)
    xb, yb = build_data(b, tb)
    assert xa.shape == (512, 8) and not np.allclose(xa, xb)
    # same data seed on the same tree reproduces exactly
    xa2, _ = build_data(a, ta)
    assert np.array_equal(xa, xa2)
    # a different tree seed gives a different ODT
    assert not np.allclose(build_tree(_tiny(tree_seed=8)).w_list, ta.w_list)


def test_init_depends_only_on_init_seed() -> None:
    m1 = build_init_model(_tiny(init_seed=5, data_seed=1))
    m2 = build_init_model(_tiny(init_seed=5, data_seed=2, batch_size=8))
    m3 = build_init_model(_tiny(init_seed=6))
    w1, w2, w3 = (m.hidden_layers[0].weight.detach().numpy() for m in (m1, m2, m3))
    assert np.array_equal(w1, w2) and not np.allclose(w1, w3)


def test_train_sweep_run_snapshots_by_step_and_stops_on_loss() -> None:
    spec = _tiny(store_all_layers=True)
    tree = build_tree(spec)
    x, y = build_data(spec, tree)
    res = train_sweep_run(spec, tree=tree, x_train=x, y_train=y)
    steps = res["snapshot_steps"]
    assert steps[0] == 0 and steps[-1] == res["final_step"] == 40
    assert res["stopped_by"] == "max_steps"
    assert np.array_equal(np.diff(steps), np.full(len(steps) - 1, 4))
    assert set(res["weights"]) == {
        "hidden_layers.0.weight", "hidden_layers.1.weight", "output_layer.weight"
    }
    assert res["weights"][FIRST_LAYER_KEY].shape == (len(steps), 6, 8)
    # trailing loss is NaN until one epoch (8 steps) has elapsed, then finite
    assert np.isnan(res["snapshot_loss"][0]) and np.isnan(res["snapshot_loss"][1])
    assert np.isfinite(res["snapshot_loss"][2:]).all()
    # snapshot 0 is the init from init_seed alone
    assert np.array_equal(
        res["weights"][FIRST_LAYER_KEY][0],
        build_init_model(spec).hidden_layers[0].weight.detach().numpy(),
    )

    # early stopping: a threshold above the initial loss stops at the first full epoch
    spec2 = _tiny(stop_loss=10.0, store_all_layers=False)
    res2 = train_sweep_run(spec2, tree=tree, x_train=x, y_train=y)
    assert res2["stopped_by"] == "stop_loss" and res2["final_step"] == 8
    assert set(res2["weights"]) == {FIRST_LAYER_KEY}


def test_save_and_load_run_round_trip(tmp_path) -> None:
    spec = _tiny(store_all_layers=True, max_steps=8)
    tree = build_tree(spec)
    x, y = build_data(spec, tree)
    res = train_sweep_run(spec, tree=tree, x_train=x, y_train=y)
    path = save_run(res, tmp_path / "r.npz")
    back = load_run(path)
    assert back["spec"] == spec
    assert back["final_step"] == res["final_step"] and back["stopped_by"] == res["stopped_by"]
    for k in res["weights"]:
        assert np.array_equal(back["weights"][k], res["weights"][k])
    assert np.array_equal(back["snapshot_steps"], res["snapshot_steps"])


def test_capture_summary_orders_crossers_and_reports_gap() -> None:
    d = 4
    u = np.array([1.0, 0, 0, 0])
    S, K = 5, 3
    W = np.zeros((S, K, d)); W[..., 1] = 1.0            # everyone starts orthogonal to u
    W[2:, 0, :] = [-1.0, 0.05, 0, 0]                    # neuron 0 crosses at snapshot 2, negative sign
    W[4:, 2, :] = [1.0, 0.0, 0, 0]                      # neuron 2 crosses at snapshot 4
    steps = np.array([0, 10, 20, 30, 40])
    fan = np.zeros((S, 2, K)); fan[-1, :, 0] = [0.3, 0.4]
    s = capture_summary(W, steps, u, threshold=0.95, fanout_weights=fan)
    assert s["winner"] == 0 and s["winner_step"] == 20 and s["winner_cos_at_cross"] < -0.95
    assert s["runner_up"] == 2 and s["runner_up_step"] == 40 and s["gap_steps"] == 20
    assert s["n_on_direction_end"] == 2 and s["on_direction_end"] == "0,2"
    assert s["winner_fanout_norm_end"] == pytest.approx(0.5)
    none = capture_summary(np.zeros((2, 2, d)) + np.array([0, 1.0, 0, 0]), steps[:2], u)
    assert none["winner"] == -1 and none["n_crossed_ever"] == 0


def test_first_layer_agreement_is_one_for_identical_runs_and_uses_common_steps() -> None:
    spec = _tiny(max_steps=16)
    tree = build_tree(spec)
    x, y = build_data(spec, tree)
    a = train_sweep_run(spec, tree=tree, x_train=x, y_train=y)
    b = train_sweep_run(spec, tree=tree, x_train=x, y_train=y)
    df = first_layer_agreement(a, b)
    assert df["min_cos"].min() > 0.999999
    # a run with a different snapshot cadence shares only some steps
    c = train_sweep_run(_tiny(max_steps=16, snapshot_every_steps=8), tree=tree, x_train=x, y_train=y)
    assert list(first_layer_agreement(a, c)["step"]) == [0, 8, 16]


def test_spec_grids_and_run_specs_write_summary(tmp_path) -> None:
    specs = part_a_specs(
        tree_seed=3, init_seeds=[5, 6], data_seeds=[4], batch_sizes=[64, 128],
        dim=8, depth=2, n_train=512, hidden_dims=(6, 5), device="cpu", max_steps=8, stop_loss=0.0,
    )
    assert len(specs) == 4 and all(not s.store_all_layers for s in specs)
    assert len({s.run_id for s in specs}) == 4
    b = part_b_specs(tree_seed=3, init_seeds=[5], data_seed=4, batch_size=64,
                     dim=8, depth=2, n_train=512, hidden_dims=(6, 5), device="cpu", max_steps=8)
    assert b[0].store_all_layers and b[0].train_seed == 1005

    df = run_specs(specs, tmp_path, log=lambda s: None)
    assert len(df) == 4 and (tmp_path / "summary.csv").exists() and (tmp_path / "tree.npz").exists()
    assert {"winner", "winner_step", "runner_up", "gap_steps", "final_loss", "stopped_by"} <= set(df.columns)
    # resume: nothing is re-run
    calls = []
    run_specs(specs, tmp_path, log=calls.append)
    assert all("skip" in c for c in calls)


def test_leading_set_groups_near_ties() -> None:
    d = 4
    u = np.array([1.0, 0, 0, 0])
    W = np.zeros((6, 4, d)); W[..., 1] = 1.0
    W[1:, 0] = [1, 0, 0, 0]; W[2:, 1] = [-1, 0, 0, 0]; W[5:, 2] = [1, 0, 0, 0]
    steps = np.array([0, 100, 200, 300, 400, 500])
    from layer_lenses.init_sweep import leading_set
    assert leading_set(W, steps, u, tie_window_steps=100) == [0, 1]
    assert leading_set(W, steps, u, tie_window_steps=500) == [0, 1, 2]
    assert leading_set(W, steps, u, tie_window_steps=0) == [0]
    assert leading_set(np.zeros((2, 2, d)) + np.array([0, 1.0, 0, 0]), steps[:2], u) == []


def test_init_features_and_rank_table_on_tiny_sweep(tmp_path) -> None:
    from layer_lenses.init_sweep import (
        init_neuron_features, load_tree, sweep_capture_table, winner_rank_table,
        part_a_invariance_table,
    )
    specs = part_b_specs(
        tree_seed=3, init_seeds=[5, 6, 7], data_seed=4, batch_size=64,
        dim=8, depth=2, n_train=512, hidden_dims=(6, 5), device="cpu", max_steps=24, stop_loss=0.0,
    )
    run_specs(specs, tmp_path, log=lambda s: None)
    table = sweep_capture_table(tmp_path)
    assert len(table) == 3 and "leading_set" in table.columns
    tree = load_tree(tmp_path / "tree.npz")
    xf, yf = build_data(_tiny(data_seed=77), tree)
    frames = []
    for r in table.itertuples():
        run = load_run(r.path)
        f = init_neuron_features(run, tree, xf, yf, device="cpu", plateau_loss=10.0)
        assert len(f) == 6 and f["is_winner"].sum() <= 1
        assert np.allclose(f["abs_cos_u0_init"], f["cos_u0_init"].abs())
        assert ((f["grad_level4_fraction_init"] >= 0) & (f["grad_level4_fraction_init"] <= 1 + 1e-9)).all()
        frames.append(f)
    feats = pd.concat(frames, ignore_index=True)
    # force a winner in every init so the rank table has rows, then check its arithmetic
    feats["is_winner"] = 0
    feats.loc[feats.groupby("init_seed")["norm_init"].idxmax(), "is_winner"] = 1
    rt = winner_rank_table(feats)
    row = rt[rt.feature == "norm_init"].iloc[0]
    assert row["mean_percentile"] == pytest.approx(1.0) and row["frac_top_decile"] == 1.0
    assert row["frac_above_median"] == 1.0 and row["p_sign_test"] == pytest.approx(2 / 8)

    # invariance table degenerate case: one run per init, the reference is itself
    per_init, per_run = part_a_invariance_table(tmp_path, reference_data_seed=4, reference_batch_size=64)
    assert len(per_init) == 3 and (per_run["min_row_cos_vs_ref_end"] > 0.99999).all()
