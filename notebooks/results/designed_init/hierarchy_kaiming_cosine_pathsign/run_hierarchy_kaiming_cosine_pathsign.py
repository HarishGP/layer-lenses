"""80-30-40 balanced-output design with the path-sign-aware cue rule (units whose init path to the logit is negative get
their cue signs flipped) and a third layer repaired so every unit has a nonzero path. Otherwise as run_hierarchy_kaiming_cosine_balanced.py:
10M points, batch 10k, cues frozen, Kaiming rows, linear cues, one unit per probe reading every cue, exact-half third layer, seed 0,
1000 epochs x 1000 steps, lr 0.02 -> 0.001 cosine, momentum 0.9."""
import json, os, time, numpy as np, pandas as pd, torch
from layer_lenses.designed_init import build_hierarchy_model, hierarchy_probe_table, logit_scale, standard_scale_spec
from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data, build_data_chunked, save_tree
from layer_lenses.relu_training import ReLUTrainConfig, train_relu_mlp, evaluate_relu_mlp
OUT = "results/designed_init/hierarchy_kaiming_cosine_pathsign"; T0 = time.time(); os.makedirs(f"{OUT}/runs", exist_ok=True)
EPOCHS, LR, MOM, ETA_MIN_RATIO, SNAP = 1000, 0.02, 0.9, 0.05, 10
BASE = dict(seed=0, cue_mode="linear", subset_mode="all", units_per_level=(1, 1, 1, 1), width2=30, width3=40, layer3_exact=True, cue_offspan_ratio=0.2, output_signs="balanced", path_sign_cues=True, layer3_nonzero_paths=True)
RUNS = [("pathsign_80", 10_000_000, 10_000, 80)]
tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); save_tree(tree, f"{OUT}/tree.npz")
xe, ye = build_data(SweepRunSpec(tree_seed=5178, data_seed=999_777, init_seed=0, batch_size=1, n_train=200_000), tree)
rows = []
for name, n_train, bs, width1 in RUNS:
    t0 = time.time(); tag = f"linear_frozen_kaiming_cosine_{name}_seed0"
    x, y = build_data_chunked(SweepRunSpec(tree_seed=5178, data_seed=5178, init_seed=0, batch_size=1, n_train=n_train), tree)
    print(f"{tag}: data {x.shape} {x.dtype}, {(time.time() - t0) / 60:.1f} min", flush=True)
    hs = standard_scale_spec("kaiming", width1=width1, **BASE)
    model, meta = build_hierarchy_model(hs, tree); lg = logit_scale(model, x[:1_000_000], device="cuda")
    print(f"{tag}: flipped units {meta['flipped_units']}, zero-path units after repair {meta['zero_path_units']}, path signs {meta['unit_path_sign'].tolist()}", flush=True)
    mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}
    mask["hidden_layers.0.weight"][meta["cue_rows"]] = 0.0
    cfg = ReLUTrainConfig(epochs=EPOCHS, lr=LR, batch_size=bs, seed=1000, device="cuda", snapshot_epochs=tuple(range(0, EPOCHS + 1, SNAP)),
                          lr_scheduler_eta_min_ratio=ETA_MIN_RATIO, optimizer="sgd", momentum=MOM, parameter_update_mask=mask, show_progress=False)
    out = train_relu_mlp(model=model, x_train=x, y_train=y, config=cfg)
    ck = out["checkpoint_snapshots"]; epochs = sorted(ck)
    np.savez(f"{OUT}/runs/{tag}.npz", epochs=np.array(epochs), epoch_losses=np.array(out["epoch_losses"]), lr_history=np.array(out["lr_history"]),
             **{f"w__{k}": np.stack([ck[e][k].numpy() for e in epochs]) for k in ck[epochs[0]]})
    m_end = out["model"]; n_eval = min(2_000_000, n_train)
    tr = evaluate_relu_mlp(model=m_end, x_eval=x[:n_eval], y_eval=y[:n_eval], device="cuda"); te = evaluate_relu_mlp(model=m_end, x_eval=xe, y_eval=ye, device="cuda")
    assert (model.output_layer.weight.detach().cpu().numpy() < 0).sum() >= 0  # balanced init checked at build time
    per_epoch = []
    for e in epochs:
        t = hierarchy_probe_table(ck[e]["hidden_layers.0.weight"].numpy(), tree)
        W1 = ck[e]["hidden_layers.0.weight"].numpy(); C = W1 / np.linalg.norm(W1, axis=1, keepdims=True) @ tree.w_list.T
        per_epoch.append({"epoch": e, "loss": np.nan if e == 0 else out["epoch_losses"][e - 1], **{f"on_target_L{l}": float((g.top_node == g.node).mean()) for l, g in t.groupby("level")},
                          **{f"cos_target_L{l}": float(g.cos_target.mean()) for l, g in t.groupby("level")}, "cues_coarse_top": int(sum(np.abs(C[k]).argmax() < 15 for k in range(30, 62))),
                          "nodes_covered": len(set(np.abs(C).argmax(1)[np.abs(C).max(1) > 0.95]))})
    pe = pd.DataFrame(per_epoch); pe.to_csv(f"{OUT}/per_epoch_{tag}.csv", index=False)
    L = np.array(out["epoch_losses"]); first = lambda thr: int(np.nonzero(L <= thr)[0][0] + 1) if (L <= thr).any() else -1
    rec = {"tag": tag, "name": name, "width1": width1, "n_train": n_train, "batch": bs, "logit_rms_init": lg["rms_logit"], "epochs": EPOCHS, "steps": EPOCHS * 1000,
           "final_epoch_loss": float(L[-1]), "min_epoch_loss": float(L.min()), "train_log_loss_end": tr["log_loss"], "test_error_end": te["zero_one_loss"], "test_error_strict_end": te["zero_one_loss_strict"], "zero_logit_frac_end": te["zero_logit_fraction"],
           "epoch_to_0.3": first(0.3), "epoch_to_0.2": first(0.2), "epoch_to_0.1": first(0.1), "epoch_to_0.05": first(0.05),
           **{f"on_target_L{l}_ep10": float(pe.loc[pe.epoch == 10, f'on_target_L{l}'].iloc[0]) for l in range(4)}, **{f"on_target_L{l}_end": float(pe[f'on_target_L{l}'].iloc[-1]) for l in range(4)},
           "cues_coarse_top_end": int(pe.cues_coarse_top.iloc[-1]), "nodes_covered_end": int(pe.nodes_covered.iloc[-1]), "wall_min": (time.time() - t0) / 60}
    rows.append(rec); pd.DataFrame(rows).to_csv(f"{OUT}/summary.csv", index=False)
    print(f"{tag}: final epoch loss {rec['final_epoch_loss']:.4f}, train log loss {tr['log_loss']:.4f}, test error {te['zero_one_loss']:.4f} (strict {te['zero_one_loss_strict']:.4f}, zero-logit {te['zero_logit_fraction']:.3f}); "
          f"epochs to 0.3/0.2/0.1/0.05: {rec['epoch_to_0.3']}/{rec['epoch_to_0.2']}/{rec['epoch_to_0.1']}/{rec['epoch_to_0.05']}; nodes covered {rec['nodes_covered_end']}/31; "
          f"on-target end {[round(rec[f'on_target_L{l}_end'],2) for l in range(4)]}; {rec['wall_min']:.1f} min", flush=True)
    del x, y, out, ck; torch.cuda.empty_cache()
json.dump({"runs": [{"name": n, "n_train": nt, "batch": b, "width1": w} for n, nt, b, w in RUNS], "epochs": EPOCHS, "lr": LR, "eta_min_ratio": ETA_MIN_RATIO, "momentum": MOM,
           "scale": "kaiming", "cues": "frozen", "output_signs": "balanced", "path_sign_cues": True, "layer3_nonzero_paths": True, "design": {k: (list(v) if isinstance(v, tuple) else v) for k, v in BASE.items()}}, open(f"{OUT}/manifest.json", "w"), indent=2)
print("ALL DONE", flush=True)
