"""Run 19: run 17 with base lr 0.01 and NO decay (eta_min ratio 1.0 makes the cosine schedule constant). Run 17: run 11's init, data and protocol (80-30-40 hierarchy design, cues frozen; 1M points, batch 10k, 10 000 epochs of
100 steps, cosine schedule to 0.05x) trained with `ScalarAdam`: SGD-momentum direction (beta1 0.9, EMA form) with the step
length normalised by the running RMS of the whole gradient norm (one scalar, beta2 0.999), base lr 0.001. The step is at most
~0.001 in parameter space whatever the gradient scale, so the effective SGD learning rate lr/rms|g| is 0.0003-0.05 along run
11's path and lr_eff * lambda_max would stay below 0.4 (run 11 sits at 3.5-3.8, the heavy-ball limit). Weights and the full
optimizer state every 10 epochs; replays exactly. After training, lambda_max every 100 epochs (20k points) and lr_eff per epoch."""
import json, os, time, numpy as np, pandas as pd, torch
from layer_lenses.designed_init import build_hierarchy_model, hierarchy_probe_table, logit_scale, standard_scale_spec
from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data, build_data_chunked, save_tree
from layer_lenses.relu_training import ReLUTrainConfig, evaluate_relu_mlp
from layer_lenses.relu_analysis import train_relu_mlp_with_optimizer_snapshots, designed_run_seed_result, save_designed_run, top_hessian_eigenvalue
OUT = "results/designed_init/hierarchy_scalaradam_lr01"; T0 = time.time(); os.makedirs(OUT, exist_ok=True)
N_TRAIN, EPOCHS, BS, LR, BETAS, ETA_MIN_RATIO, SNAP = 1_000_000, 10_000, 10_000, 0.01, (0.9, 0.999), 1.0, 10
DESIGN = dict(seed=0, cue_mode="linear", subset_mode="all", units_per_level=(1, 1, 1, 1), width1=80, width2=30, width3=40, cue_offspan_ratio=0.2,
              output_signs="balanced", path_sign_cues=True, layer3_mode="balanced_paths", layer3_majority=15, cue_block_scale="inv_sqrt")
DATA = dict(kind="sweep_chunked", tree_seed=5178, data_seed=5178, n_train=N_TRAIN, chunk_size=1_000_000, eval_seed=999_777, n_eval=200_000, dim=100, depth=5)
tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); save_tree(tree, f"{OUT}/tree.npz")
x, y = build_data_chunked(SweepRunSpec(tree_seed=5178, data_seed=5178, init_seed=0, batch_size=1, n_train=N_TRAIN), tree)
xe, ye = build_data(SweepRunSpec(tree_seed=5178, data_seed=999_777, init_seed=0, batch_size=1, n_train=200_000), tree)
tag = "linear_frozen_kaiming_const_scalaradam01_80_seed0"; t0 = time.time()
hs = standard_scale_spec("kaiming", **DESIGN); model, meta = build_hierarchy_model(hs, tree); lg = logit_scale(model, x, device="cuda")
print(f"{tag}: data {x.shape} {x.dtype}; flipped units {meta['flipped_units']}, zero-path units {meta['zero_path_units']}, logit rms {lg['rms_logit']:.3f}", flush=True)
mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}; mask["hidden_layers.0.weight"][meta["cue_rows"]] = 0.0
cfg = ReLUTrainConfig(epochs=EPOCHS, lr=LR, batch_size=BS, seed=1000, device="cuda", snapshot_epochs=tuple(range(0, EPOCHS + 1, SNAP)),
                      lr_scheduler_eta_min_ratio=ETA_MIN_RATIO, optimizer="scalar_adam", adam_betas=BETAS, parameter_update_mask=mask, show_progress=False)
out = train_relu_mlp_with_optimizer_snapshots(model=model, x_train=x, y_train=y, config=cfg)
print(f"trained in {(time.time() - t0) / 60:.1f} min; snapshots {len(out['checkpoint_snapshots'])}", flush=True)
sr = designed_run_seed_result(out=out, train_config=cfg, model_spec={"input_dim": 100, "hidden_dims": [80, 30, 40], "bias": False, "activation": "relu"},
                              data_spec=DATA, tree=tree, design={**{k: (list(v) if isinstance(v, tuple) else v) for k, v in DESIGN.items()}, "scale": "kaiming", "cues": "frozen",
                                                                   "cue_rows": meta["cue_rows"], "flipped_units": meta["flipped_units"], "unit_path_sign": meta["unit_path_sign"].tolist(),
                                                                   "optimizer": {"kind": "scalar_adam", "lr": LR, "betas": list(BETAS)}}, tag=tag)
save_designed_run(sr, f"{OUT}/replay")
ck = out["checkpoint_snapshots"]; epochs = sorted(ck); names = [n for n, _ in out["model"].named_parameters()]
m_end = out["model"]; tr = evaluate_relu_mlp(model=m_end, x_eval=x, y_eval=y, device="cuda"); te = evaluate_relu_mlp(model=m_end, x_eval=xe, y_eval=ye, device="cuda")
x20, y20 = x[:20000], y[:20000]; mask_np = {nm: mask[nm] for nm in names}; per_epoch = []
for e in epochs:
    W1 = ck[e]["hidden_layers.0.weight"].numpy(); t = hierarchy_probe_table(W1, tree); C = W1 / np.linalg.norm(W1, axis=1, keepdims=True) @ tree.w_list.T
    st = out["optimizer_snapshots"][e][0]["state"]; lr_e = out["lr_history"][e] if e < EPOCHS else out["lr_history"][-1]
    if 0 in st and "exp_avg_sq" in st[0] and float(st[0]["step"]) > 0:
        v_hat = float(st[0]["exp_avg_sq"]) / (1 - BETAS[1] ** float(st[0]["step"])); rms_g = float(np.sqrt(v_hat)); lr_eff = float(lr_e) / (rms_g + 1e-8)
    else:
        rms_g, lr_eff = np.nan, np.nan
    row = {"epoch": e, "step": e * (N_TRAIN // BS), "loss": np.nan if e == 0 else float(np.mean(out["epoch_losses"][max(0, e - SNAP):e])), "lr": float(lr_e), "rms_grad_norm": rms_g, "lr_eff": lr_eff,
           **{f"on_target_L{l}": float((g.top_node == g.node).mean()) for l, g in t.groupby("level")}, **{f"cos_target_L{l}": float(g.cos_target.mean()) for l, g in t.groupby("level")},
           "nodes_covered": len(set(np.abs(C).argmax(1)[np.abs(C).max(1) > 0.95]))}
    if e % 100 == 0:
        ev = top_hessian_eigenvalue(m_end, {nm: ck[e][nm].numpy() for nm in names}, x20, y20, device="cuda", iters=30, mask=mask_np)
        row["lambda_max"] = ev["lambda_max"]; row["lr_eff_x_lambda"] = lr_eff * ev["lambda_max"]; row["eigvec_frac_W1"] = ev["layer_fraction"]["hidden_layers.0.weight"]; row["eigvec_frac_W2"] = ev["layer_fraction"]["hidden_layers.1.weight"]
    per_epoch.append(row)
pe = pd.DataFrame(per_epoch); pe.to_csv(f"{OUT}/per_epoch_{tag}.csv", index=False)
L = np.array(out["epoch_losses"]); first = lambda thr: int(np.nonzero(L <= thr)[0][0] + 1) if (L <= thr).any() else -1
rec = {"tag": tag, "n_train": N_TRAIN, "batch": BS, "epochs": EPOCHS, "steps_per_epoch": N_TRAIN // BS, "steps": EPOCHS * (N_TRAIN // BS), "lr": LR, "betas": str(BETAS), "logit_rms_init": lg["rms_logit"],
       "final_epoch_loss": float(L[-1]), "min_epoch_loss": float(L.min()), "train_log_loss_end": tr["log_loss"], "test_error_end": te["zero_one_loss"], "test_error_strict_end": te["zero_one_loss_strict"],
       "zero_logit_frac_end": te["zero_logit_fraction"], "epoch_to_0.3": first(0.3), "epoch_to_0.2": first(0.2), "epoch_to_0.1": first(0.1), "epoch_to_0.05": first(0.05), "epoch_to_0.03": first(0.03),
       "loss_std_1000_3000": float(np.nanstd(L[1000:3000])), "max_lr_eff_x_lambda": float(pe.lr_eff_x_lambda.max()), "max_lambda": float(pe.lambda_max.max()),
       **{f"on_target_L{l}_end": float(pe[f"on_target_L{l}"].iloc[-1]) for l in range(4)}, "nodes_covered_end": int(pe.nodes_covered.iloc[-1]), "wall_min": (time.time() - t0) / 60}
pd.DataFrame([rec]).to_csv(f"{OUT}/summary.csv", index=False)
json.dump({"n_train": N_TRAIN, "epochs": EPOCHS, "batch": BS, "steps_per_epoch": N_TRAIN // BS, "optimizer": "scalar_adam", "lr": LR, "betas": list(BETAS), "eta_min_ratio": ETA_MIN_RATIO, "snapshot_every_epochs": SNAP,
           "scale": "kaiming", "cues": "frozen", "design": {k: (list(v) if isinstance(v, tuple) else v) for k, v in DESIGN.items()}, "data": DATA}, open(f"{OUT}/manifest.json", "w"), indent=2)
print(f"{tag}: final epoch loss {rec['final_epoch_loss']:.4f}, train log loss {tr['log_loss']:.4f}, test error {te['zero_one_loss']:.4f} (strict {te['zero_one_loss_strict']:.4f}); "
      f"epochs to 0.3/0.2/0.1/0.05/0.03: {rec['epoch_to_0.3']}/{rec['epoch_to_0.2']}/{rec['epoch_to_0.1']}/{rec['epoch_to_0.05']}/{rec['epoch_to_0.03']}; nodes covered {rec['nodes_covered_end']}/31; "
      f"max lr_eff*lambda {rec['max_lr_eff_x_lambda']:.2f} (max lambda {rec['max_lambda']:.0f}); loss std 1000-3000 {rec['loss_std_1000_3000']:.4f}; on-target end {[round(rec[f'on_target_L{l}_end'], 2) for l in range(4)]}; {rec['wall_min']:.1f} min", flush=True)
print("ALL DONE", flush=True)
