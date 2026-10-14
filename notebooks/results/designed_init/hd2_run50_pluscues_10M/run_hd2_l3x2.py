"""Notebook hierarchy_design_runs_2, run 2: run 41 (= run 1 there: 62-60-4, the four distinct third-layer neurons of run 38) with the
two redundant third-layer neurons removed and a plain, uniform-learning-rate, Kaiming-scaled init.

Third layer: two neurons, S (reads the 30 path-sign-+ units, pattern 1001 per quadruple) and Sc (reads the 30 path-sign-- units, 0110);
run 41's neurons 1 and 2 (the 5-copy classes, which flip their output sign and end as weaker copies of 0 and 3) are dropped.
Kaiming row norms sqrt(2) in every layer: first-layer rows sqrt(2) (orthogonal off-span probes, cues pole*(v+u_j)), second-layer rows
sqrt(2) (the quadruple design's rows already are), W3 rows sqrt(2) (30 entries of sqrt(2/30) = 0.258), output (+1, -1).
No gradient scales, nothing frozen: SGD momentum 0.9, lr 0.01 -> 0.0005 cosine, 1M points (seed 5178), batch 10k, 10 000 epochs,
snapshots every 10 epochs. The init passes the flip-equivariance test for all 15 tree flips.
  run 47: --out hd2_run2_l3x2_kaiming --tag hd2_run2_l3x2_kaiming_seed0                      (section 2 of hierarchy_design_runs_2.ipynb)
  run 48: --w3-entry 1.633 --out-weight 0.5: the readout rescaled so that the effective output weight per unit is +-0.816 as in run 41
          AND the factorised readout's shared-mode rate at init, lr x (30 e^2 + 2 w^2), matches run 41's full model (30 x 2.67 + 2 = 82):
          e = sqrt(2.67) = 1.633, w = 0.5 (run 47: e = 0.258, w = 1 gives 4; e = 0.816, w = 1 or e = 0.258, w = 3.16 would give 22).
  run 50: run 49 with --plus-cues-only: only the 16 cues on +u_j (rows (m, +)) are kept, 46-60-2; each unit's cue block keeps its
          coefficients on them (-side x c), so it is no longer linear in x (relu(v + u_m) alone). Tree flips still fix the init (they never
          negate a level-4 normal); the label-negation symmetry, which needed each cue's twin, is gone. Rows are rescaled to norm sqrt(2) as before.
  run 49: run 48 with --n-train 10000000 --epochs 1000 --snap 1: ten times the data, the same 1M steps and the same cosine schedule in
          step units (an epoch is then 1000 steps; snapshots every epoch).
"""
import argparse, json, os, time, numpy as np, pandas as pd, torch
from layer_lenses.designed_init import build_hierarchy_model, hierarchy_probe_table, logit_scale, standard_scale_spec, equivariant_quadruple_weights, flip_equivariance_report
from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data, build_data_chunked, save_tree
from layer_lenses.relu_mlp import ReLUMLP
from layer_lenses.relu_training import ReLUTrainConfig, evaluate_relu_mlp
from layer_lenses.relu_analysis import train_relu_mlp_with_optimizer_snapshots, designed_run_seed_result, save_designed_run
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--data-seed", type=int, default=5178); ap.add_argument("--orth-seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=10_000); ap.add_argument("--n-train", type=int, default=1_000_000); ap.add_argument("--snap", type=int, default=10)
ap.add_argument("--w3-entry", type=float, default=float(np.sqrt(2.0 / 30.0))); ap.add_argument("--out-weight", type=float, default=1.0); ap.add_argument("--plus-cues-only", action="store_true")
args = ap.parse_args(); OUT = f"results/designed_init/{args.out}"; os.makedirs(f"{OUT}/runs", exist_ok=True); T0 = time.time()
N_TRAIN, EPOCHS, BS, LR, MOM, ETA_MIN_RATIO, SNAP = args.n_train, args.epochs, 10_000, 0.01, 0.9, 0.05, args.snap
DESIGN = dict(seed=0, cue_mode="linear", subset_mode="all", units_per_level=(1, 1, 1, 1), width1=80, width2=30, width3=40, cue_offspan_ratio=0.2,
              output_signs="balanced", path_sign_cues=True, layer3_mode="balanced_paths", layer3_majority=15, cue_block_scale="inv_sqrt")
DATA = dict(kind="sweep_chunked", tree_seed=5178, data_seed=args.data_seed, n_train=N_TRAIN, chunk_size=1_000_000, eval_seed=999_777, n_eval=200_000, dim=100, depth=5)
tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); save_tree(tree, f"{OUT}/tree.npz"); U = tree.w_list
x, y = build_data_chunked(SweepRunSpec(tree_seed=5178, data_seed=args.data_seed, init_seed=0, batch_size=1, n_train=N_TRAIN), tree)
xe, ye = build_data(SweepRunSpec(tree_seed=5178, data_seed=999_777, init_seed=0, batch_size=1, n_train=200_000), tree)
tag = args.tag; t0 = time.time()
hs = standard_scale_spec("kaiming", **DESIGN); base, meta = build_hierarchy_model(hs, tree); v = U[15:31].sum(0)
W1 = base.hidden_layers[0].weight.detach().numpy().astype(np.float64).copy(); W2 = base.hidden_layers[1].weight.detach().numpy().astype(np.float64)
W3 = base.hidden_layers[2].weight.detach().numpy().astype(np.float64); Wout = base.output_layer.weight.detach().numpy().astype(np.float64)
for r in meta["cue_rows"]:                                                     # cues as in runs 26-41
    c = W1[r] / np.linalg.norm(W1[r]) @ U.T; m = int(np.abs(c).argmax()); pole = int(np.sign(c[m])); assert m >= 15 and abs(c[m]) > 0.9
    new = pole * (v + U[m]); W1[r] = new / np.linalg.norm(new) * np.linalg.norm(W1[r])
q = equivariant_quadruple_weights(W1, W2, W3, Wout, cue_rows=meta["cue_rows"], tree=tree, orth_seed=args.orth_seed)
W1 = q["W1"][:62]; W2 = q["W2"][:, :62]                                        # no spare rows
CUE_ROWS = list(meta["cue_rows"]); CUE_DESC = "32 cues +-u_j"
if args.plus_cues_only:                                                        # run 50: keep the probes and the 16 cues on +u_j only
    Cin = W1 @ U.T; keep = list(range(30)) + [r for r in meta["cue_rows"] if Cin[r, int(np.abs(Cin[r]).argmax())] > 0]
    assert len(keep) == 46 and sorted(int(np.abs(Cin[r]).argmax()) for r in keep[30:]) == list(range(15, 31))
    W1 = W1[keep]; W2 = W2[:, keep]; CUE_ROWS = list(range(30, 46)); CUE_DESC = "16 cues +u_j only"
s_units = np.sign(q["W3"].T @ q["Wout"].ravel())                               # +1 for the path-sign-+ units (types 0, 3), -1 for types 1, 2
assert s_units.tolist() == [1, -1, -1, 1] * 15
W3 = np.stack([(s_units > 0).astype(np.float64), (s_units < 0).astype(np.float64)]) * args.w3_entry             # rows S, Sc (run 47: entries sqrt(2/30), row norm sqrt(2))
Wout = np.array([[args.out_weight, -args.out_weight]])                                                           # run 47: (+1, -1), norm sqrt(2)
W2_PRE = np.linalg.norm(W2, axis=1); W1 = W1 / np.linalg.norm(W1, axis=1, keepdims=True) * np.sqrt(2.0); W2 = W2 / W2_PRE[:, None] * np.sqrt(2.0)
for name, W in [("W1", W1), ("W2", W2)]: assert np.allclose(np.linalg.norm(W, axis=1), np.sqrt(2.0)), name
W_EFF = args.out_weight * args.w3_entry; SHARED_RATE = 30 * args.w3_entry ** 2 + 2 * args.out_weight ** 2
reports = {j: flip_equivariance_report(W1, W2, W3, Wout, tree, j, cue_rows=CUE_ROWS, units_per_node=4) for j in range(15)}
worst = max(max(r.values()) for r in reports.values()); assert worst < 1e-5, reports
HIDDEN = [W1.shape[0], 60, 2]; model = ReLUMLP(input_dim=100, hidden_dims=HIDDEN, bias=False)
model.load_state_dict({"hidden_layers.0.weight": torch.as_tensor(W1, dtype=torch.float32), "hidden_layers.1.weight": torch.as_tensor(W2, dtype=torch.float32),
                       "hidden_layers.2.weight": torch.as_tensor(W3, dtype=torch.float32), "output_layer.weight": torch.as_tensor(Wout, dtype=torch.float32)})
lg = logit_scale(model, x, device="cuda")
print(f"{tag}: {HIDDEN[0]}-60-2, {CUE_DESC} (W1, W2 row norms sqrt 2; W3 entries {args.w3_entry:.3f}, output +-{args.out_weight}; effective output +-{W_EFF:.3f}, shared-mode rate factor {SHARED_RATE:.1f}); {N_TRAIN} points x {EPOCHS} epochs of {N_TRAIN // BS} steps, snapshots every {SNAP}; flip test worst deviation {worst:.1e}; "
      f"W2 row norms before rescale {np.linalg.norm(q['W2'][:, :62], axis=1).min():.3f}-{np.linalg.norm(q['W2'][:, :62], axis=1).max():.3f} (after the cue selection {W2_PRE.min():.3f}-{W2_PRE.max():.3f}); data seed {args.data_seed} {x.shape}; logit rms {lg['rms_logit']:.4f}", flush=True)
mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}                                                # nothing frozen, uniform lr
cfg = ReLUTrainConfig(epochs=EPOCHS, lr=LR, batch_size=BS, seed=1000, device="cuda", snapshot_epochs=tuple(range(0, EPOCHS + 1, SNAP)),
                      lr_scheduler_eta_min_ratio=ETA_MIN_RATIO, optimizer="sgd", momentum=MOM, parameter_update_mask=mask, show_progress=False)
out = train_relu_mlp_with_optimizer_snapshots(model=model, x_train=x, y_train=y, config=cfg)
print(f"trained in {(time.time() - t0) / 60:.1f} min; snapshots {len(out['checkpoint_snapshots'])}", flush=True)
sr = designed_run_seed_result(out=out, train_config=cfg, model_spec={"input_dim": 100, "hidden_dims": HIDDEN, "bias": False, "activation": "relu"},
                              data_spec=DATA, tree=tree, design={**{k: (list(vv) if isinstance(vv, tuple) else vv) for k, vv in DESIGN.items()}, "scale": "kaiming row norms sqrt(2) in every layer", "cues": "free; pole*(v + u_j) renormalised",
                                                                   "init": "equivariant quadruples, third layer S / Sc (2 neurons), uniform lr", "arch": f"{HIDDEN[0]}-60-2", "cues_desc": CUE_DESC, "w3_entry": args.w3_entry, "out_weight": args.out_weight, "w_eff": W_EFF, "shared_rate": SHARED_RATE, "n_train": N_TRAIN, "epochs": EPOCHS, "snap": SNAP, "orth_seed": args.orth_seed, "data_seed": args.data_seed, "cue_rows": CUE_ROWS, "probe_rows": q["probe_rows"], "spare_rows": [],
                                                                   "unit_path_sign": q["unit_path_sign"].tolist(), "unit_block_sign": q["unit_block_sign"].tolist(), "unit_node": q["unit_node"].tolist(), "flip_reports": {int(k): r for k, r in reports.items()}}, tag=tag)
save_designed_run(sr, f"{OUT}/replay")
ck = out["checkpoint_snapshots"]; epochs = sorted(ck); m_end = out["model"]
tr = evaluate_relu_mlp(model=m_end, x_eval=x, y_eval=y, device="cuda"); te = evaluate_relu_mlp(model=m_end, x_eval=xe, y_eval=ye, device="cuda")
per_epoch = []
for ep in epochs:
    W1e = ck[ep]["hidden_layers.0.weight"].numpy(); t = hierarchy_probe_table(W1e, tree); Ce = W1e / np.linalg.norm(W1e, axis=1, keepdims=True) @ U.T
    per_epoch.append({"epoch": ep, "step": ep * (N_TRAIN // BS), "loss": np.nan if ep == 0 else float(np.mean(out["epoch_losses"][max(0, ep - SNAP):ep])),
                      **{f"on_target_L{l}": float((g.top_node == g.node).mean()) for l, g in t.groupby("level")}, **{f"cos_target_L{l}": float(g.cos_target.mean()) for l, g in t.groupby("level")},
                      "nodes_covered": len(set(np.abs(Ce).argmax(1)[np.abs(Ce).max(1) > 0.95]))})
pe = pd.DataFrame(per_epoch); pe.to_csv(f"{OUT}/per_epoch_{tag}.csv", index=False)
L = np.array(out["epoch_losses"]); first = lambda thr: int(np.nonzero(L <= thr)[0][0] + 1) if (L <= thr).any() else -1
rec = {"tag": tag, "arch": f"{HIDDEN[0]}-60-2", "cues": CUE_DESC, "w3_entry": args.w3_entry, "out_weight": args.out_weight, "w_eff": W_EFF, "orth_seed": args.orth_seed, "data_seed": args.data_seed, "n_train": N_TRAIN, "batch": BS, "epochs": EPOCHS, "lr": LR, "logit_rms_init": lg["rms_logit"], "final_epoch_loss": float(L[-1]), "min_epoch_loss": float(L.min()),
       "train_log_loss_end": tr["log_loss"], "test_error_end": te["zero_one_loss"], "test_error_strict_end": te["zero_one_loss_strict"], "zero_logit_frac_end": te["zero_logit_fraction"],
       "epoch_to_0.3": first(0.3), "epoch_to_0.2": first(0.2), "epoch_to_0.1": first(0.1), "epoch_to_0.05": first(0.05), "epoch_to_0.03": first(0.03), "epoch_to_0.01": first(0.01),
       **{f"on_target_L{l}_end": float(pe[f"on_target_L{l}"].iloc[-1]) for l in range(4)}, "nodes_covered_end": int(pe.nodes_covered.iloc[-1]), "wall_min": (time.time() - t0) / 60}
pd.DataFrame([rec]).to_csv(f"{OUT}/summary.csv", index=False)
json.dump({"arch": f"{HIDDEN[0]}-60-2", "cues_desc": CUE_DESC, "init": "two third-layer neurons, uniform lr", "w3_entry": args.w3_entry, "out_weight": args.out_weight, "w_eff": W_EFF, "snapshot_every_epochs": SNAP, "orth_seed": args.orth_seed, "data_seed": args.data_seed, "cues": "free; pole*(v + u_j) renormalised", "n_train": N_TRAIN, "epochs": EPOCHS, "batch": BS, "lr": LR, "eta_min_ratio": ETA_MIN_RATIO, "momentum": MOM, "snapshot_every_epochs": SNAP,
           "design": {k: (list(vv) if isinstance(vv, tuple) else vv) for k, vv in DESIGN.items()}, "data": DATA}, open(f"{OUT}/manifest.json", "w"), indent=2)
print(f"{tag}: final epoch loss {rec['final_epoch_loss']:.4f}, test error {te['zero_one_loss']:.4f} (strict {te['zero_one_loss_strict']:.4f}); epochs to 0.3/0.2/0.1/0.05/0.03/0.01: {rec['epoch_to_0.3']}/{rec['epoch_to_0.2']}/{rec['epoch_to_0.1']}/{rec['epoch_to_0.05']}/{rec['epoch_to_0.03']}/{rec['epoch_to_0.01']}; "
      f"nodes covered {rec['nodes_covered_end']}/31; on-target end {[round(rec[f'on_target_L{l}_end'], 2) for l in range(4)]}; {rec['wall_min']:.1f} min", flush=True)
print("ALL DONE", flush=True)
