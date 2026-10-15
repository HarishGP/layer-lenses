"""Run 51 (notebook 2, section 6): run 49's init (62-60-2, readout e = 1.633 / w = 0.5, 10M points) with a PHASED protocol:
  epochs    0- 500: the 32 cue rows frozen (probes, W2, W3, output train);
  epochs  500-1000: the 30 probe rows frozen (cues, W2, W3, output train);
  epochs 1000-1200: nothing frozen.
Three segments of `train_relu_mlp_with_optimizer_snapshots` chained with `start_epoch` and the previous segment's optimizer state, so the
momentum buffers and the cosine schedule (lr 0.01 -> 0.0005 over the 1200 epochs) continue across the phase changes; batch 10k, 1000
steps per epoch, snapshots every epoch. Saved as one replayable run (segments merged).
  run 51: --out hd2_run51_phased_10M --tag hd2_run51_phased_10M_seed0
  run 56: --two-layer --out-scale 80 --phases 500 500: run 55 with the output row's gradient scaled by 80 (parameter_update_mask value),
          the shared-mode growth rate 30 e^2 + 2 w^2 of run 53's factorised readout at init, so the readout grows at run 53's initial pace.
  run 55: --two-layer --phases 500 500: run 53 WITHOUT the third hidden layer (62-60): each unit is wired straight to the logit with weight
          path sign x (w3_entry x out_weight) = +-0.816, the third layer's effective path weight to the logit at run 53's init, so the
          logit function at init is run 53's (its third-layer ReLUs gate < 3% of inputs). Only the readout's growth dynamics differ.
  run 54: --negate-w2 --phases 500 500: run 53 with W2 negated at init (every probe weight +1 instead of -1, every cue block B -> -B,
          i.e. the same four blocks per node attached to the opposite probe weight sign); the flip equivariance is unchanged.
  run 53: --cue-bias 1.0 --phases 500 500: the bias-1 control of run 52 (same two-phase 1000-epoch schedule).
  run 52: --cue-bias 0.5 --phases 500 500: cues = pole * (v + 0.5 u_j) renormalised (half the level-4 bias), TWO phases: cues frozen for
          500 epochs, then nothing frozen for 500 (cosine over the 1000 epochs, as run 49).
"""
import argparse, json, os, time, numpy as np, pandas as pd, torch
from layer_lenses.designed_init import build_hierarchy_model, hierarchy_probe_table, logit_scale, standard_scale_spec, equivariant_quadruple_weights, flip_equivariance_report
from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data, build_data_chunked, save_tree
from layer_lenses.relu_mlp import ReLUMLP
from layer_lenses.relu_training import ReLUTrainConfig, evaluate_relu_mlp, checkpoint_model_from_state
from layer_lenses.relu_analysis import train_relu_mlp_with_optimizer_snapshots, designed_run_seed_result, save_designed_run
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--data-seed", type=int, default=5178); ap.add_argument("--orth-seed", type=int, default=0)
ap.add_argument("--n-train", type=int, default=10_000_000); ap.add_argument("--phases", type=int, nargs="+", default=[500, 500, 200], help="3 values: epochs with cues frozen, with probes frozen, with nothing frozen; 2 values: cues frozen, nothing frozen"); ap.add_argument("--cue-bias", type=float, default=1.0); ap.add_argument("--negate-w2", action="store_true")
ap.add_argument("--w3-entry", type=float, default=1.633); ap.add_argument("--out-weight", type=float, default=0.5); ap.add_argument("--two-layer", action="store_true", help="run 55: no third layer, Wout = path sign x w3_entry x out_weight per unit"); ap.add_argument("--out-scale", type=float, default=1.0, help="run 56: gradient scale on the output row (mask value)")
args = ap.parse_args(); OUT = f"results/designed_init/{args.out}"; os.makedirs(f"{OUT}/runs", exist_ok=True); T0 = time.time()
N_TRAIN, BS, LR, MOM, ETA_MIN_RATIO, SNAP = args.n_train, 10_000, 0.01, 0.9, 0.05, 1
assert len(args.phases) in (2, 3); EPOCHS = sum(args.phases); _b = np.cumsum([0] + args.phases); BOUNDS = [(int(_b[i]), int(_b[i + 1])) for i in range(len(args.phases))]
PHASE_FROZEN = ["cues", "probes", "none"] if len(args.phases) == 3 else ["cues", "none"]
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
for r in meta["cue_rows"]:
    c = W1[r] / np.linalg.norm(W1[r]) @ U.T; m = int(np.abs(c).argmax()); pole = int(np.sign(c[m])); assert m >= 15 and abs(c[m]) > 0.9
    new = pole * (v + args.cue_bias * U[m]); W1[r] = new / np.linalg.norm(new) * np.linalg.norm(W1[r])
q = equivariant_quadruple_weights(W1, W2, W3, Wout, cue_rows=meta["cue_rows"], tree=tree, orth_seed=args.orth_seed)
W1 = q["W1"][:62]; W2 = q["W2"][:, :62]; CUE_ROWS = list(meta["cue_rows"]); PROBE_ROWS = list(range(30))
s_units = np.sign(q["W3"].T @ q["Wout"].ravel()); assert s_units.tolist() == [1, -1, -1, 1] * 15
W3 = np.stack([(s_units > 0).astype(np.float64), (s_units < 0).astype(np.float64)]) * args.w3_entry; Wout = np.array([[args.out_weight, -args.out_weight]])
W1 = W1 / np.linalg.norm(W1, axis=1, keepdims=True) * np.sqrt(2.0); W2 = W2 / np.linalg.norm(W2, axis=1, keepdims=True) * np.sqrt(2.0)
if args.negate_w2: W2 = -W2                                                    # run 54
if args.two_layer:                                                             # run 55: the output row is the third layer's linear map at init
    Wout = (s_units * args.w3_entry * args.out_weight)[None, :]; W3 = None
_W3, _Wo = (Wout, np.ones((1, 1))) if args.two_layer else (W3, Wout)           # two-layer: the output row plays W3's part in the flip test
reports = {j: flip_equivariance_report(W1, W2, _W3, _Wo, tree, j, cue_rows=CUE_ROWS, units_per_node=4) for j in range(15)}
worst = max(max(r.values()) for r in reports.values()); assert worst < 1e-5, reports
HIDDEN = [62, 60] if args.two_layer else [62, 60, 2]; ARCH = "-".join(map(str, HIDDEN)); model = ReLUMLP(input_dim=100, hidden_dims=HIDDEN, bias=False)
_sd = {"hidden_layers.0.weight": torch.as_tensor(W1, dtype=torch.float32), "hidden_layers.1.weight": torch.as_tensor(W2, dtype=torch.float32), "output_layer.weight": torch.as_tensor(Wout, dtype=torch.float32)}
if not args.two_layer: _sd["hidden_layers.2.weight"] = torch.as_tensor(W3, dtype=torch.float32)
model.load_state_dict(_sd)
lg = logit_scale(model, x, device="cuda")
print(f"{tag}: {ARCH} (run 49's init, cue bias {args.cue_bias}{', W2 negated' if args.negate_w2 else ''}{', no third layer, Wout = path sign x ' + str(args.w3_entry * args.out_weight) if args.two_layer else ''}{', output gradient x ' + str(args.out_scale) if args.out_scale != 1.0 else ''}); phases (frozen rows, epochs): {list(zip(PHASE_FROZEN, BOUNDS))}; {N_TRAIN} points, {N_TRAIN // BS} steps per epoch, cosine lr over {EPOCHS} epochs; flip test {worst:.1e}; logit rms {lg['rms_logit']:.4f}", flush=True)
def phase_mask(k):
    mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}
    if PHASE_FROZEN[k] == "cues": mask["hidden_layers.0.weight"][CUE_ROWS] = 0.0
    if PHASE_FROZEN[k] == "probes": mask["hidden_layers.0.weight"][PROBE_ROWS] = 0.0
    if args.out_scale != 1.0: mask["output_layer.weight"][:] = args.out_scale                  # run 56: output gradient x out_scale
    return mask
merged = None; prev = None; cfg = None
for k, (a, b) in enumerate(BOUNDS):
    cfg = ReLUTrainConfig(epochs=EPOCHS, lr=LR, batch_size=BS, seed=1000, device="cuda", snapshot_epochs=tuple(range(a, b + 1, SNAP)), lr_scheduler_eta_min_ratio=ETA_MIN_RATIO,
                          optimizer="sgd", momentum=MOM, parameter_update_mask=phase_mask(k), show_progress=False)
    if prev is not None: model = checkpoint_model_from_state(prev["model"], prev["checkpoint_snapshots"][a])
    out = train_relu_mlp_with_optimizer_snapshots(model=model, x_train=x, y_train=y, config=cfg, start_epoch=a, stop_epoch=b,      # the cosine schedule spans all EPOCHS
                                                  initial_optimizer_states=prev["optimizer_snapshots"][a] if prev is not None else None)
    print(f"phase {k}: epochs {a}-{b} in {(time.time() - t0) / 60:.1f} min; loss at end {np.array(out['epoch_losses'])[b - 1]:.4f}", flush=True)
    if merged is None: merged = out
    else:
        for key in ("checkpoint_snapshots", "optimizer_snapshots", "permutation_generator_snapshots"):
            if isinstance(out.get(key), dict): merged[key].update(out[key])
        for key in ("epoch_losses", "lr_history"):
            L_old, L_new = np.array(merged[key], dtype=float), np.array(out[key], dtype=float); L_old[a:b] = L_new[a:b]; merged[key] = L_old.tolist()
        merged["lr_history_by_group"] = out.get("lr_history_by_group", merged.get("lr_history_by_group")); merged["model"] = out["model"]
    prev = out
out = merged
sr = designed_run_seed_result(out=out, train_config=cfg, model_spec={"input_dim": 100, "hidden_dims": HIDDEN, "bias": False, "activation": "relu"},
                              data_spec=DATA, tree=tree, design={**{k: (list(vv) if isinstance(vv, tuple) else vv) for k, vv in DESIGN.items()}, "scale": "kaiming row norms sqrt(2), readout e / w", "cues": f"pole*(v + {args.cue_bias} u_j) renormalised",
                                                                   "init": "run 49's", "arch": ARCH, "two_layer": args.two_layer, "out_scale": args.out_scale, "w3_entry": args.w3_entry, "out_weight": args.out_weight, "n_train": N_TRAIN, "epochs": EPOCHS, "snap": SNAP, "phases": dict(zip(PHASE_FROZEN, BOUNDS)), "cue_bias": args.cue_bias, "negate_w2": args.negate_w2,
                                                                   "orth_seed": args.orth_seed, "data_seed": args.data_seed, "cue_rows": CUE_ROWS, "probe_rows": PROBE_ROWS, "spare_rows": [], "unit_path_sign": q["unit_path_sign"].tolist(), "unit_block_sign": q["unit_block_sign"].tolist(), "unit_node": q["unit_node"].tolist()}, tag=tag)
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
rec = {"tag": tag, "arch": ARCH, "two_layer": args.two_layer, "out_scale": args.out_scale, "phases": str(BOUNDS), "n_train": N_TRAIN, "batch": BS, "epochs": EPOCHS, "lr": LR, "logit_rms_init": lg["rms_logit"], "final_epoch_loss": float(L[-1]), "min_epoch_loss": float(np.nanmin(L)),
       "train_log_loss_end": tr["log_loss"], "test_error_end": te["zero_one_loss"], "test_error_strict_end": te["zero_one_loss_strict"], "zero_logit_frac_end": te["zero_logit_fraction"],
       "epoch_to_0.3": first(0.3), "epoch_to_0.2": first(0.2), "epoch_to_0.1": first(0.1), "epoch_to_0.05": first(0.05), "epoch_to_0.03": first(0.03), "epoch_to_0.01": first(0.01),
       **{f"loss_end_phase{k + 1}": float(L[b - 1]) for k, (a, b) in enumerate(BOUNDS)}, "cue_bias": args.cue_bias, "negate_w2": args.negate_w2,
       **{f"on_target_L{l}_end": float(pe[f"on_target_L{l}"].iloc[-1]) for l in range(4)}, "nodes_covered_end": int(pe.nodes_covered.iloc[-1]), "wall_min": (time.time() - t0) / 60}
pd.DataFrame([rec]).to_csv(f"{OUT}/summary.csv", index=False)
json.dump({"arch": ARCH, "two_layer": args.two_layer, "out_scale": args.out_scale, "init": "run 49's", "cue_bias": args.cue_bias, "phases": dict(zip(PHASE_FROZEN, BOUNDS)), "n_train": N_TRAIN, "epochs": EPOCHS, "batch": BS, "lr": LR, "eta_min_ratio": ETA_MIN_RATIO, "momentum": MOM, "snapshot_every_epochs": SNAP,
           "design": {k: (list(vv) if isinstance(vv, tuple) else vv) for k, vv in DESIGN.items()}, "data": DATA}, open(f"{OUT}/manifest.json", "w"), indent=2)
print(f"{tag}: final epoch loss {rec['final_epoch_loss']:.4f} (end of phases: {[round(rec[f'loss_end_phase{k + 1}'], 4) for k in range(len(BOUNDS))]}), test error {te['zero_one_loss']:.4f}; epochs to 0.3/0.2/0.1/0.05/0.03/0.01: {rec['epoch_to_0.3']}/{rec['epoch_to_0.2']}/{rec['epoch_to_0.1']}/{rec['epoch_to_0.05']}/{rec['epoch_to_0.03']}/{rec['epoch_to_0.01']}; "
      f"nodes covered {rec['nodes_covered_end']}/31; on-target end {[round(rec[f'on_target_L{l}_end'], 2) for l in range(4)]}; {rec['wall_min']:.1f} min", flush=True)
print("ALL DONE", flush=True)
