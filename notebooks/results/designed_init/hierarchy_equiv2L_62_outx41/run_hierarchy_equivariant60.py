"""Run 37: the 60-unit EQUIVARIANT init (80-60-40), run 26's protocol. Built from run 26's design (`equivariant_quadruple_weights`):
  * the 30 probe rows and the 18 spare rows are mutually orthogonal, purely off-span vectors (norm sqrt(2), from --orth-seed);
  * cues as in run 26: pole * (v + u_j) renormalised, free;
  * every node has four units (4j + t, t = 0..3) of types (block sign, path sign) = (+,+), (+,-), (-,-), (-,+): the design's cue block
    B_j or its negation, with the design's probe weight -1 on the +pole probe (types 0, 2) or the -pole probe (types 1, 3);
  * W3 uses one common column (run 26's unit-0 column) for every path-sign-+ unit and the complementary one (unit 1's) for every
    path-sign-- unit, so every third-layer row has 30 ones; output alternates +-1 as before.
The init is fixed, up to a rotation of the off-span complement, by every flip of the tree's symmetry group (`flip_equivariance_report`
is asserted here for all 15 flips), so the sample / batch order are the only breakers of congruence between same-level nodes.
Protocol: 1M points, batch 10k, 10 000 epochs, lr 0.01 -> 0.0005 cosine, momentum 0.9, nothing frozen, snapshots every 10 epochs.
  run 37: --out hierarchy_equivariant_60 --tag linear_free_kaiming_cosine_equiv60_seed0
  run 38: --out hierarchy_equivariant_60_nospare --tag linear_free_kaiming_cosine_equiv60_nospare_seed0 --no-spare   (62-60-40: the 18 spare rows dropped after the build)
  run 39: run 38 with --cue-bias 0.1: cues = pole * (v + 0.1 u_j) renormalised (the design's u_j content of the cues ten times weaker)
  run 41: run 38 with --reduced-l3: the third layer is its FOUR distinct neurons (62-60-4): rows (S,+) x15, (S,-) x5, (Sc,+) x5, (Sc,-) x15,
          where S reads the path-sign-+ units; the multiplicity m is folded into the output weight (m x 0.2236) and the gradient of that
          weight is scaled by m, that of the W3 row by 1/m (parameter_update_mask), which reproduces the 40-neuron dynamics exactly.
  runs 42-46: run 38 with the THIRD LAYER REMOVED (--two-layer, 62-60): the output weights start at w_eff = Wout . W3 = +-0.816 x the
          units' path signs (the third layer's linear map at init). 42: plain, SGD momentum as run 38; 43: --out-scale 41 (the output
          layer's gradient scaled by 41, the shared-mode rate of the factorised readout at init); 44: --freeze-out; 45 / 46: ScalarAdam
          (--optimizer scalar_adam --const-lr) at lr 0.01 / 0.03 (--lr).
  run 40: run 38 with --full-batch-steps 100000: FULL-BATCH gradient descent on the 1M points (one step per "epoch", constant lr 0.01,
          momentum 0.9, snapshots every 100 steps) as the population-gradient-flow proxy; no batch noise, the fixed sample is the only noise.
"""
import argparse, json, os, time, numpy as np, pandas as pd, torch
from layer_lenses.designed_init import build_hierarchy_model, hierarchy_probe_table, logit_scale, standard_scale_spec, equivariant_quadruple_weights, flip_equivariance_report
from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data, build_data_chunked, save_tree
from layer_lenses.relu_mlp import ReLUMLP
from layer_lenses.relu_training import ReLUTrainConfig, evaluate_relu_mlp
from layer_lenses.relu_analysis import train_relu_mlp_with_optimizer_snapshots, designed_run_seed_result, save_designed_run
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--data-seed", type=int, default=5178); ap.add_argument("--orth-seed", type=int, default=0); ap.add_argument("--no-spare", action="store_true"); ap.add_argument("--cue-bias", type=float, default=1.0)
ap.add_argument("--full-batch-steps", type=int, default=0); ap.add_argument("--reduced-l3", action="store_true"); ap.add_argument("--two-layer", action="store_true"); ap.add_argument("--out-scale", type=float, default=1.0); ap.add_argument("--freeze-out", action="store_true")
ap.add_argument("--optimizer", default="sgd"); ap.add_argument("--const-lr", action="store_true"); ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--epochs", type=int, default=10_000); ap.add_argument("--n-train", type=int, default=1_000_000)
args = ap.parse_args(); OUT = f"results/designed_init/{args.out}"; os.makedirs(f"{OUT}/runs", exist_ok=True); T0 = time.time()
N_TRAIN, EPOCHS, BS, LR, MOM, ETA_MIN_RATIO, SNAP = args.n_train, args.epochs, 10_000, args.lr, 0.9, (1.0 if args.const_lr else 0.05), 10
if args.full_batch_steps:                                                       # run 40: full-batch GD, one step per epoch, constant lr
    EPOCHS, BS, ETA_MIN_RATIO, SNAP = args.full_batch_steps, N_TRAIN, 1.0, 100
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
for r in meta["cue_rows"]:                                                     # cues as in run 26
    c = W1[r] / np.linalg.norm(W1[r]) @ U.T; m = int(np.abs(c).argmax()); pole = int(np.sign(c[m])); assert m >= 15 and abs(c[m]) > 0.9
    new = pole * (v + args.cue_bias * U[m]); W1[r] = new / np.linalg.norm(new) * np.linalg.norm(W1[r])
q = equivariant_quadruple_weights(W1, W2, W3, Wout, cue_rows=meta["cue_rows"], tree=tree, orth_seed=args.orth_seed)
if args.no_spare:                                                              # run 38: drop the 18 spare rows (W2 has no weight on them)
    keep = q["probe_rows"] + q["cue_rows"]; assert keep == list(range(62)) and not (q["W2"][:, 62:] != 0).any()
    q["W1"] = q["W1"][:62]; q["W2"] = q["W2"][:, :62]; q["spare_rows"] = []
grad_scale = None
if args.reduced_l3:                                                            # run 41: four distinct third-layer neurons with multiplicities
    inS = q["W3"][:, 0] != 0; wo = q["Wout"].ravel(); classes = [(inS & (wo > 0)), (inS & (wo < 0)), (~inS & (wo > 0)), (~inS & (wo < 0))]
    mult = np.array([int(c.sum()) for c in classes]); assert mult.tolist() == [15, 5, 5, 15]
    reps = [int(np.nonzero(c)[0][0]) for c in classes]
    q["W3"] = np.stack([q["W3"][r] for r in reps]); q["Wout"] = (mult * np.array([wo[r] for r in reps]))[None, :]
    grad_scale = {"hidden_layers.2.weight": np.repeat(1.0 / mult[:, None], 60, axis=1).astype(np.float32), "output_layer.weight": mult[None, :].astype(np.float32)}
    q["multiplicity"] = mult.tolist()
if args.two_layer:                                                             # runs 42-46: no third layer; output = the third layer's linear map at init
    assert not args.reduced_l3
    q["Wout"] = (q["Wout"] @ q["W3"]).reshape(1, 60); q["W3"] = None
WIDTH1 = q["W1"].shape[0]; WIDTH3 = None if args.two_layer else q["W3"].shape[0]; ARCH = f"{WIDTH1}-60" if args.two_layer else f"{WIDTH1}-60-{WIDTH3}"
HIDDEN = [WIDTH1, 60] if args.two_layer else [WIDTH1, 60, WIDTH3]
_W3, _Wo = (q["Wout"], np.ones((1, 1))) if args.two_layer else (q["W3"], q["Wout"])               # two-layer: the output row plays W3's part in the flip test
reports = {j: flip_equivariance_report(q["W1"], q["W2"], _W3, _Wo, tree, j, cue_rows=meta["cue_rows"], units_per_node=4) for j in range(15)}
worst = max(max(r.values()) for r in reports.values()); assert worst < 1e-5, reports
model = ReLUMLP(input_dim=100, hidden_dims=HIDDEN, bias=False)
_sd = {"hidden_layers.0.weight": torch.as_tensor(q["W1"], dtype=torch.float32), "hidden_layers.1.weight": torch.as_tensor(q["W2"], dtype=torch.float32), "output_layer.weight": torch.as_tensor(q["Wout"], dtype=torch.float32)}
if not args.two_layer: _sd["hidden_layers.2.weight"] = torch.as_tensor(q["W3"], dtype=torch.float32)
model.load_state_dict(_sd)
lg = logit_scale(model, x, device="cuda")
rows = q["probe_rows"] + q["spare_rows"]; R = q["W1"][rows]
print(f"{tag}: optimizer {args.optimizer} lr {LR}{' const' if args.const_lr else ' cosine'}; out-scale {args.out_scale} freeze-out {args.freeze_out}; cue bias {args.cue_bias}; batch {BS} x {EPOCHS} epochs, lr {LR} eta_min_ratio {ETA_MIN_RATIO}; {ARCH} equivariant init; flip test worst deviation {worst:.1e} over 15 flips; probe+spare rows off-span to {np.abs(R @ U.T).max():.1e}, "
      f"pairwise |cos| max {np.abs(R @ R.T / 2 - np.eye(len(rows))).max():.1e}; W3 ones per row {None if args.two_layer else np.unique((q['W3'] != 0).sum(1)).tolist()}; third layer {WIDTH3} rows{(' with multiplicities ' + str(q.get('multiplicity'))) if args.reduced_l3 else ''}; data seed {args.data_seed} {x.shape}; logit rms {lg['rms_logit']:.4f}", flush=True)
mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}                                                # nothing frozen
if grad_scale: mask.update(grad_scale)                                                                                                   # run 41: per-entry gradient scales
if args.out_scale != 1.0: mask["output_layer.weight"] = np.full(mask["output_layer.weight"].shape, args.out_scale, dtype=np.float32)      # run 43
if args.freeze_out: mask["output_layer.weight"] = np.zeros(mask["output_layer.weight"].shape, dtype=np.float32)                            # run 44
cfg = ReLUTrainConfig(epochs=EPOCHS, lr=LR, batch_size=BS, seed=1000, device="cuda", snapshot_epochs=tuple(range(0, EPOCHS + 1, SNAP)),
                      lr_scheduler_eta_min_ratio=ETA_MIN_RATIO, optimizer=args.optimizer, momentum=MOM, parameter_update_mask=mask, show_progress=False)
out = train_relu_mlp_with_optimizer_snapshots(model=model, x_train=x, y_train=y, config=cfg)
print(f"trained in {(time.time() - t0) / 60:.1f} min; snapshots {len(out['checkpoint_snapshots'])}", flush=True)
sr = designed_run_seed_result(out=out, train_config=cfg, model_spec={"input_dim": 100, "hidden_dims": HIDDEN, "bias": False, "activation": "relu"},
                              data_spec=DATA, tree=tree, design={**{k: (list(vv) if isinstance(vv, tuple) else vv) for k, vv in DESIGN.items()}, "scale": "kaiming", "cues": f"free; pole*(v + {args.cue_bias} u_j) renormalised",
                                                                   "init": "equivariant quadruples (run 37)", "arch": ARCH, "two_layer": args.two_layer, "out_scale": args.out_scale, "freeze_out": args.freeze_out, "optimizer": args.optimizer, "const_lr": args.const_lr, "reduced_l3": args.reduced_l3, "multiplicity": q.get("multiplicity"), "cue_bias": args.cue_bias, "full_batch_steps": args.full_batch_steps, "orth_seed": args.orth_seed, "data_seed": args.data_seed, "cue_rows": meta["cue_rows"], "probe_rows": q["probe_rows"], "spare_rows": q["spare_rows"],
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
rec = {"tag": tag, "arch": ARCH, "optimizer": args.optimizer, "const_lr": args.const_lr, "out_scale": args.out_scale, "freeze_out": args.freeze_out, "cue_bias": args.cue_bias, "full_batch_steps": args.full_batch_steps, "init": "equivariant quadruples", "orth_seed": args.orth_seed, "data_seed": args.data_seed, "n_train": N_TRAIN, "batch": BS, "epochs": EPOCHS, "lr": LR, "logit_rms_init": lg["rms_logit"], "final_epoch_loss": float(L[-1]), "min_epoch_loss": float(L.min()),
       "train_log_loss_end": tr["log_loss"], "test_error_end": te["zero_one_loss"], "test_error_strict_end": te["zero_one_loss_strict"], "zero_logit_frac_end": te["zero_logit_fraction"],
       "epoch_to_0.3": first(0.3), "epoch_to_0.2": first(0.2), "epoch_to_0.1": first(0.1), "epoch_to_0.05": first(0.05), "epoch_to_0.03": first(0.03),
       **{f"on_target_L{l}_end": float(pe[f"on_target_L{l}"].iloc[-1]) for l in range(4)}, "nodes_covered_end": int(pe.nodes_covered.iloc[-1]), "wall_min": (time.time() - t0) / 60}
pd.DataFrame([rec]).to_csv(f"{OUT}/summary.csv", index=False)
json.dump({"arch": ARCH, "optimizer": args.optimizer, "const_lr": args.const_lr, "out_scale": args.out_scale, "freeze_out": args.freeze_out, "cue_bias": args.cue_bias, "full_batch_steps": args.full_batch_steps, "init": "equivariant quadruples (run 37)", "orth_seed": args.orth_seed, "data_seed": args.data_seed, "cues": f"free; pole*(v + {args.cue_bias} u_j) renormalised", "n_train": N_TRAIN, "epochs": EPOCHS, "batch": BS, "lr": LR, "eta_min_ratio": ETA_MIN_RATIO, "momentum": MOM, "snapshot_every_epochs": SNAP,
           "design": {k: (list(vv) if isinstance(vv, tuple) else vv) for k, vv in DESIGN.items()}, "data": DATA}, open(f"{OUT}/manifest.json", "w"), indent=2)
print(f"{tag}: final epoch loss {rec['final_epoch_loss']:.4f}, test error {te['zero_one_loss']:.4f} (strict {te['zero_one_loss_strict']:.4f}); epochs to 0.3/0.2/0.1/0.05/0.03: {rec['epoch_to_0.3']}/{rec['epoch_to_0.2']}/{rec['epoch_to_0.1']}/{rec['epoch_to_0.05']}/{rec['epoch_to_0.03']}; "
      f"nodes covered {rec['nodes_covered_end']}/31; on-target end {[round(rec[f'on_target_L{l}_end'], 2) for l in range(4)]}; {rec['wall_min']:.1f} min", flush=True)
print("ALL DONE", flush=True)
