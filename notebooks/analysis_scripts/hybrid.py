import numpy as np, torch
from layer_lenses.relu_mlp import ReLUMLP
from layer_lenses.relu_training import (
    ReLUTrainConfig, build_relu_optimizers, optimizer_group_names,
    optimizer_parameter_names, split_named_parameters_sgd_adam,
    suggest_sgd_lr_for_adam_parity, normalized_optimizer_name,
)
from layer_lenses.relu_analysis import (
    run_single_relu_seed_replayable, replay_epoch_from_snapshot,
    regenerate_seed_result_data, epoch_replay_summary_table, flatten_named_tensors,
    save_seed_result_pickle, load_seed_result_pickle,
)

common = dict(dim=20, depth=3, n_train=4000, hidden_dims=[8, 8, 8], total_epochs=4, lr=2e-3,
              batch_size=500, weight_decay=0.01, snapshot_epochs=[0, 1, 2, 3, 4], bias=False,
              show_progress=False, device="cpu")

# --- 1. the split -----------------------------------------------------------
m = ReLUMLP(input_dim=20, hidden_dims=[8, 8, 8], bias=True)
for k in [1, 2, 3]:
    sgd, adam = split_named_parameters_sgd_adam(m, k)
    print(f"split={k}: sgd={[n for n,_ in sgd]}  adam={[n for n,_ in adam]}")
for bad in [0, 4, -1]:
    try: split_named_parameters_sgd_adam(m, bad)
    except ValueError as e: print(f"  split={bad} rejected: {str(e)[:70]}...")
cfg = ReLUTrainConfig(optimizer="sgd_adam", sgd_adam_split_layer=1)
print("group names:", optimizer_group_names(cfg), "| normalized:", normalized_optimizer_name(cfg))
print("param names by group:", optimizer_parameter_names(m, cfg))

# --- 2. suggested lr --------------------------------------------------------
x, y = np.random.randn(2000, 20).astype(np.float32), np.where(np.random.randn(2000) > 0, 1, -1).astype(np.int8)
m2 = ReLUMLP(input_dim=20, hidden_dims=[8, 8, 8], bias=False)
print("\nsuggest_sgd_lr_for_adam_parity:",
      {k: round(v, 6) for k, v in suggest_sgd_lr_for_adam_parity(
          model=m2, x_train=x, y_train=y, config=ReLUTrainConfig(optimizer="sgd_adam", lr=2e-3, weight_decay=0.01)).items()})

# --- 3. hybrid run + replay -------------------------------------------------
SGD_LR = 20.0
sr = run_single_relu_seed_replayable(11, optimizer="sgd_adam", momentum=0.9,
                                     sgd_adam_split_layer=1, sgd_lr=SGD_LR,
                                     sgd_weight_decay=0.0, **common)
out = sr["out"]
print("\nhybrid run:", out["optimizer_group_names"], "| param groups:", out["optimizer_parameter_names"])
print("lr_history_by_group:", {k: [round(v, 5) for v in vs] for k, vs in out["lr_history_by_group"].items()})
print("losses:", [round(v, 5) for v in out["epoch_losses"]])
print("optimizer snapshot is a list of", len(out["optimizer_snapshots"][0]), "state dicts")

data = regenerate_seed_result_data(sr)
r = replay_epoch_from_snapshot(sr, 1, data=data, record_full_data_gradient=True)
print("\nreplay verification:", r["verification"]["max_abs_param_diff_overall"],
      r["verification"]["max_abs_optimizer_state_diff"])
assert r["verification"]["max_abs_param_diff_overall"] == 0.0
assert r["verification"]["max_abs_optimizer_state_diff"] == 0.0
print("lr_by_group:", r["lr_by_group"], "| parameter_groups:", r["parameter_groups"])

# --- 4. each layer is governed by the right optimizer ----------------------
st = r["steps"][0]["optimizer_state_after"]
print("\nstate keys per parameter:")
for n in sorted(st): print(f"  {n:28s} -> {sorted(st[n].keys())}")
assert "momentum_buffer" in st["hidden_layers.0.weight"]
assert "exp_avg" in st["hidden_layers.1.weight"] and "exp_avg" in st["output_layer.weight"]
assert "exp_avg" not in st["hidden_layers.0.weight"]
assert "adam_step_size" in r["steps"][0]
assert set(r["steps"][0]["adam_step_size"].keys()) == set(r["parameter_groups"]["adamw"])
print("adam decomposition covers exactly:", sorted(r["steps"][0]["adam_step_size"].keys()))

# --- 5. per-group behaviour in the summary table ---------------------------
for grp in ["sgd", "adamw"]:
    df = epoch_replay_summary_table(r, parameter_names=r["parameter_groups"][grp])
    cols = ["step", "grad_norm", "update_norm", "cos_update_grad", "cos_update_full_grad"]
    print(f"\n--- group {grp} ---")
    print(df[cols].head(3).to_string(index=False))

# --- 6. first-layer SGD arithmetic is exactly -lr*g at a fresh momentum buffer
sr0 = run_single_relu_seed_replayable(11, optimizer="sgd_adam", momentum=0.0,
                                      sgd_adam_split_layer=1, sgd_lr=SGD_LR,
                                      sgd_weight_decay=0.0, **common)
r0 = replay_epoch_from_snapshot(sr0, 0, data=data)
s0 = r0["steps"][0]
g0 = s0["grad"]["hidden_layers.0.weight"]; u0 = s0["update"]["hidden_layers.0.weight"]
print("\nlayer 0, first hybrid step: max|update + sgd_lr*g| =",
      float((u0 + SGD_LR * g0).abs().max()))
print("  cos(update, -grad) layer 0 =",
      float((-u0.reshape(-1) @ g0.reshape(-1)) / (u0.norm() * g0.norm())))
g1 = s0["grad"]["hidden_layers.1.weight"]; u1 = s0["update"]["hidden_layers.1.weight"]
print("  cos(update, -grad) layer 1 =",
      float((-u1.reshape(-1) @ g1.reshape(-1)) / (u1.norm() * g1.norm())), "(Adam sign step)")

# --- 7. pickle round trip ---------------------------------------------------
pth = save_seed_result_pickle(sr, f"{__import__('os').environ.get('SP','/tmp')}/hybrid.pkl") if False else save_seed_result_pickle(sr, "/tmp/claude-1000/-home-harish-researchcodes-layer-lenses/8b3631ed-dc7e-41c4-af0e-3b349a6ef0e2/scratchpad/hybrid.pkl")
r2 = replay_epoch_from_snapshot(load_seed_result_pickle(pth), 2)
assert r2["verification"]["max_abs_param_diff_overall"] == 0.0
print("\npickle round trip + replay: exact,", pth.stat().st_size, "bytes")
print("\nALL HYBRID CHECKS PASSED")
