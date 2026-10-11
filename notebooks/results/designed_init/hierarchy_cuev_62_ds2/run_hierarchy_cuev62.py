"""Runs 31-36: run 26 (cues = pole * (v + u_j) renormalised, free; random probe directions; nothing frozen; SGD momentum 0.9, lr 0.01 ->
0.0005 cosine, 1M points, batch 10k, 10 000 epochs) WITHOUT the 18 spare rows (62-30-40). Layers 3-4 come from run 11's stored init so
the third-layer pattern is run 26's (a 62-row build draws a different one); with --w3-seed N the balanced-paths 0/1 pattern of W3 is
redrawn with that seed instead (same 15-of-20 structure per unit, same nonzero value, same output signs), so only W3's pattern differs.
  run 31: --out hierarchy_cuev_62 --tag linear_free_kaiming_cosine_cuev_62_seed0
  run 32: --out hierarchy_cuev_62_w3seed1 --tag linear_free_kaiming_cosine_cuev_62_w3seed1 --w3-seed 1
--data-seed N changes the training set only (tree and 200k test set unchanged): runs 33-34 = 31-32 with --data-seed 6178.
--probe-seed N replaces the 30 probe rows by fresh off-span random vectors (norm sqrt(2)) drawn from that seed: runs 35-36 = 31-32 with --probe-seed 7.
"""
import argparse, json, os, time, numpy as np, pandas as pd, torch
from layer_lenses.designed_init import build_hierarchy_model, hierarchy_probe_table, logit_scale, standard_scale_spec
from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data, build_data_chunked, save_tree
from layer_lenses.relu_training import ReLUTrainConfig, evaluate_relu_mlp
from layer_lenses.relu_analysis import train_relu_mlp_with_optimizer_snapshots, designed_run_seed_result, save_designed_run, load_designed_run
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--w3-seed", type=int, default=None); ap.add_argument("--data-seed", type=int, default=5178); ap.add_argument("--probe-seed", type=int, default=None)
args = ap.parse_args(); OUT = f"results/designed_init/{args.out}"; os.makedirs(f"{OUT}/runs", exist_ok=True); T0 = time.time()
N_TRAIN, EPOCHS, BS, LR, MOM, ETA_MIN_RATIO, SNAP = 1_000_000, 10_000, 10_000, 0.01, 0.9, 0.05, 10
DESIGN = dict(seed=0, cue_mode="linear", subset_mode="all", units_per_level=(1, 1, 1, 1), width1=62, width2=30, width3=40, cue_offspan_ratio=0.2,
              output_signs="balanced", path_sign_cues=True, layer3_mode="balanced_paths", layer3_majority=15, cue_block_scale="inv_sqrt")
DATA = dict(kind="sweep_chunked", tree_seed=5178, data_seed=args.data_seed, n_train=N_TRAIN, chunk_size=1_000_000, eval_seed=999_777, n_eval=200_000, dim=100, depth=5)
tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); save_tree(tree, f"{OUT}/tree.npz"); U = tree.w_list
x, y = build_data_chunked(SweepRunSpec(tree_seed=5178, data_seed=args.data_seed, init_seed=0, batch_size=1, n_train=N_TRAIN), tree)
xe, ye = build_data(SweepRunSpec(tree_seed=5178, data_seed=999_777, init_seed=0, batch_size=1, n_train=200_000), tree)
tag = args.tag; t0 = time.time()
hs = standard_scale_spec("kaiming", **DESIGN); model, meta = build_hierarchy_model(hs, tree); v = U[15:31].sum(0)
_s11 = load_designed_run("results/designed_init/hierarchy_replayable/replay")["out"]["checkpoint_snapshots"][0]
with torch.no_grad():
    W1 = model.hidden_layers[0].weight
    assert torch.equal(W1, _s11["hidden_layers.0.weight"][:62]) and torch.equal(model.hidden_layers[1].weight, _s11["hidden_layers.1.weight"][:, :62])
    W3_11 = _s11["hidden_layers.2.weight"].clone(); model.output_layer.weight.copy_(_s11["output_layer.weight"])
    if args.w3_seed is None:
        model.hidden_layers[2].weight.copy_(W3_11); w3_desc = "run 11 init"
    else:                                                                      # redraw the balanced-paths pattern (same construction as the design) with another seed
        val = float(W3_11[W3_11 != 0].unique().item()); rng = np.random.default_rng(args.w3_seed); W3 = np.zeros((40, 30)); pos_rows, neg_rows = np.arange(0, 40, 2), np.arange(1, 40, 2)
        for j in range(15):
            a, b = 2 * j, 2 * j + 1; pa = rng.permutation(pos_rows); W3[pa[:15], a] = 1.0; W3[pa[15:], b] = 1.0; nb = rng.permutation(neg_rows); W3[nb[:15], b] = 1.0; W3[nb[15:], a] = 1.0
        W3 *= val; model.hidden_layers[2].weight.copy_(torch.as_tensor(W3, dtype=torch.float32)); w3_desc = f"balanced-paths pattern redrawn with seed {args.w3_seed}"
        same = float((model.hidden_layers[2].weight != 0).float().eq((W3_11 != 0).float()).float().mean()); assert (model.hidden_layers[2].weight != 0).sum(1).eq(15).all()
        print(f"W3 redrawn: {same * 100:.0f}% of the 0/1 entries agree with run 11's pattern; unit paths {(model.hidden_layers[2].weight.T @ model.output_layer.weight.ravel()).round().tolist()[:6]} ...", flush=True)
    probe_desc = "design seed 0 (run 26's)"
    if args.probe_seed is not None:                                            # fresh off-span probe directions from another seed
        P = np.eye(100) - U.T @ U; prng = np.random.default_rng(args.probe_seed)
        for pidx in range(30):
            e = P @ prng.normal(size=100); W1[pidx] = torch.as_tensor(e / np.linalg.norm(e) * np.sqrt(2.0), dtype=torch.float32)
        probe_desc = f"off-span random vectors from seed {args.probe_seed}"
        assert np.abs((W1[:30].numpy() / np.sqrt(2.0)) @ U.T).max() < 1e-5
    for r in meta["cue_rows"]:                                                 # cues as in run 26
        w = W1[r].numpy(); c = w / np.linalg.norm(w) @ U.T; j = int(np.abs(c).argmax()); pole = int(np.sign(c[j])); assert j >= 15 and abs(c[j]) > 0.9
        new = pole * (v + U[j]); W1[r] = torch.as_tensor(new / np.linalg.norm(new) * np.linalg.norm(w), dtype=torch.float32)
lg = logit_scale(model, x, device="cuda"); print(f"{tag}: 62 rows, W3 = {w3_desc}; probes = {probe_desc}; data seed {args.data_seed} {x.shape}; flipped units {meta['flipped_units']}, logit rms {lg['rms_logit']:.3f}", flush=True)
mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}                                                # nothing frozen
cfg = ReLUTrainConfig(epochs=EPOCHS, lr=LR, batch_size=BS, seed=1000, device="cuda", snapshot_epochs=tuple(range(0, EPOCHS + 1, SNAP)),
                      lr_scheduler_eta_min_ratio=ETA_MIN_RATIO, optimizer="sgd", momentum=MOM, parameter_update_mask=mask, show_progress=False)
out = train_relu_mlp_with_optimizer_snapshots(model=model, x_train=x, y_train=y, config=cfg)
print(f"trained in {(time.time() - t0) / 60:.1f} min; snapshots {len(out['checkpoint_snapshots'])}", flush=True)
sr = designed_run_seed_result(out=out, train_config=cfg, model_spec={"input_dim": 100, "hidden_dims": [62, 30, 40], "bias": False, "activation": "relu"},
                              data_spec=DATA, tree=tree, design={**{k: (list(vv) if isinstance(vv, tuple) else vv) for k, vv in DESIGN.items()}, "scale": "kaiming", "cues": "free; pole*(v + u_j) renormalised",
                                                                   "spare_rows": "none", "w3": w3_desc, "probes": probe_desc, "data_seed": args.data_seed, "cue_rows": meta["cue_rows"], "flipped_units": meta["flipped_units"], "unit_path_sign": meta["unit_path_sign"].tolist()}, tag=tag)
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
rec = {"tag": tag, "width1": 62, "w3": w3_desc, "probes": probe_desc, "data_seed": args.data_seed, "n_train": N_TRAIN, "batch": BS, "epochs": EPOCHS, "lr": LR, "logit_rms_init": lg["rms_logit"], "final_epoch_loss": float(L[-1]), "min_epoch_loss": float(L.min()),
       "train_log_loss_end": tr["log_loss"], "test_error_end": te["zero_one_loss"], "test_error_strict_end": te["zero_one_loss_strict"], "zero_logit_frac_end": te["zero_logit_fraction"],
       "epoch_to_0.3": first(0.3), "epoch_to_0.2": first(0.2), "epoch_to_0.1": first(0.1), "epoch_to_0.05": first(0.05), "epoch_to_0.03": first(0.03),
       **{f"on_target_L{l}_end": float(pe[f"on_target_L{l}"].iloc[-1]) for l in range(4)}, "nodes_covered_end": int(pe.nodes_covered.iloc[-1]), "wall_min": (time.time() - t0) / 60}
pd.DataFrame([rec]).to_csv(f"{OUT}/summary.csv", index=False)
json.dump({"width1": 62, "w3": w3_desc, "probes": probe_desc, "data_seed": args.data_seed, "cues": "free; pole*(v + u_j) renormalised", "n_train": N_TRAIN, "epochs": EPOCHS, "batch": BS, "lr": LR, "eta_min_ratio": ETA_MIN_RATIO, "momentum": MOM, "snapshot_every_epochs": SNAP,
           "design": {k: (list(vv) if isinstance(vv, tuple) else vv) for k, vv in DESIGN.items()}, "data": DATA}, open(f"{OUT}/manifest.json", "w"), indent=2)
print(f"{tag}: final epoch loss {rec['final_epoch_loss']:.4f}, test error {te['zero_one_loss']:.4f} (strict {te['zero_one_loss_strict']:.4f}); epochs to 0.3/0.2/0.1/0.05/0.03: {rec['epoch_to_0.3']}/{rec['epoch_to_0.2']}/{rec['epoch_to_0.1']}/{rec['epoch_to_0.05']}/{rec['epoch_to_0.03']}; "
      f"nodes covered {rec['nodes_covered_end']}/31; on-target end {[round(rec[f'on_target_L{l}_end'], 2) for l in range(4)]}; {rec['wall_min']:.1f} min", flush=True)
print("ALL DONE", flush=True)
