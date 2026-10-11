"""Runs 27-30: run 26's protocol (cues = pole * (v + u_j) renormalised, v = sum of the level-4 normals; nothing frozen; SGD momentum 0.9,
lr 0.01 -> 0.0005 cosine, 1M points, batch 10k, 10 000 epochs of 100 steps) with a SYMMETRIC probe init: all 30 probe rows equal one
off-span vector e (probe 0's design init, norm sqrt(2), orthogonal to the 31 tree normals). With --width 80 the 18 spare rows are set
to off-span vectors orthogonal to each other and to e (Gram-Schmidt of their design inits against the tree span and e, norm sqrt(2));
with --width 62 there are no spare rows and W3 / the output layer are copied from run 11's stored init (a 62-row build draws a
different third-layer pattern). --data-seed changes the training data (the tree and the 200k test set are unchanged).
  run 27: --width 80 --data-seed 5178 | run 28: --width 62 --data-seed 5178 | run 29: --width 80 --data-seed 6178 | run 30: --width 62 --data-seed 6178
"""
import argparse, json, os, time, numpy as np, pandas as pd, torch
from layer_lenses.designed_init import build_hierarchy_model, hierarchy_probe_table, logit_scale, standard_scale_spec
from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data, build_data_chunked, save_tree
from layer_lenses.relu_training import ReLUTrainConfig, evaluate_relu_mlp
from layer_lenses.relu_analysis import train_relu_mlp_with_optimizer_snapshots, designed_run_seed_result, save_designed_run, load_designed_run
ap = argparse.ArgumentParser(); ap.add_argument("--width", type=int, choices=(80, 62), required=True); ap.add_argument("--data-seed", type=int, default=5178); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
args = ap.parse_args(); OUT = f"results/designed_init/{args.out}"; os.makedirs(f"{OUT}/runs", exist_ok=True); T0 = time.time()
N_TRAIN, EPOCHS, BS, LR, MOM, ETA_MIN_RATIO, SNAP = 1_000_000, 10_000, 10_000, 0.01, 0.9, 0.05, 10
DESIGN = dict(seed=0, cue_mode="linear", subset_mode="all", units_per_level=(1, 1, 1, 1), width1=args.width, width2=30, width3=40, cue_offspan_ratio=0.2,
              output_signs="balanced", path_sign_cues=True, layer3_mode="balanced_paths", layer3_majority=15, cue_block_scale="inv_sqrt")
DATA = dict(kind="sweep_chunked", tree_seed=5178, data_seed=args.data_seed, n_train=N_TRAIN, chunk_size=1_000_000, eval_seed=999_777, n_eval=200_000, dim=100, depth=5)
tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); save_tree(tree, f"{OUT}/tree.npz"); U = tree.w_list
x, y = build_data_chunked(SweepRunSpec(tree_seed=5178, data_seed=args.data_seed, init_seed=0, batch_size=1, n_train=N_TRAIN), tree)
xe, ye = build_data(SweepRunSpec(tree_seed=5178, data_seed=999_777, init_seed=0, batch_size=1, n_train=200_000), tree)
tag = args.tag; t0 = time.time()
hs = standard_scale_spec("kaiming", **DESIGN); model, meta = build_hierarchy_model(hs, tree)
v = U[15:31].sum(0); P = np.eye(100) - U.T @ U                                   # projector onto the 69-dim complement of the tree span
with torch.no_grad():
    W1 = model.hidden_layers[0].weight
    if args.width == 62:                                                       # layers 3-4 from run 11's init, as in runs 21-23
        _s11 = load_designed_run("results/designed_init/hierarchy_replayable/replay")["out"]["checkpoint_snapshots"][0]
        assert torch.equal(model.hidden_layers[1].weight, _s11["hidden_layers.1.weight"][:, :62])
        model.hidden_layers[2].weight.copy_(_s11["hidden_layers.2.weight"]); model.output_layer.weight.copy_(_s11["output_layer.weight"])
    for r in meta["cue_rows"]:                                                 # cues as in run 26
        w = W1[r].numpy(); c = w / np.linalg.norm(w) @ U.T; j = int(np.abs(c).argmax()); pole = int(np.sign(c[j])); assert j >= 15 and abs(c[j]) > 0.9
        new = pole * (v + U[j]); W1[r] = torch.as_tensor(new / np.linalg.norm(new) * np.linalg.norm(w), dtype=torch.float32)
    e = W1[0].numpy().copy(); assert np.abs(U @ e).max() < 1e-5, "probe 0's init is not off-span"
    e = P @ e; e = e / np.linalg.norm(e) * np.sqrt(2.0); W1[:30] = torch.as_tensor(e, dtype=torch.float32)   # all probes = e
    if args.width == 80:                                                       # spare rows: orthogonal off-span vectors, orthogonal to e
        basis = [e / np.linalg.norm(e)]
        for r in range(62, 80):
            w = P @ W1[r].numpy()
            for b in basis: w = w - (w @ b) * b
            w = w / np.linalg.norm(w); basis.append(w); W1[r] = torch.as_tensor(w * np.sqrt(2.0), dtype=torch.float32)
        B = np.stack(basis); assert np.abs(B @ B.T - np.eye(19)).max() < 1e-5 and np.abs(U @ B.T).max() < 1e-5
    C = (W1 / W1.norm(dim=1, keepdim=True)).numpy() @ U.T
print(f"{tag}: width {args.width}, data seed {args.data_seed}; probes identical (max |probe_i - probe_0| {float((W1[:30] - W1[0]).abs().max()):.1e}), |cos(probe, nodes)| max {np.abs(C[:30]).max():.1e}; "
      f"cues |cos own normal| {np.abs(C[meta['cue_rows']]).max(1).mean():.3f}; spare rows {args.width - 62}", flush=True)
lg = logit_scale(model, x, device="cuda"); print(f"data {x.shape}; flipped units {meta['flipped_units']}, logit rms {lg['rms_logit']:.3f}", flush=True)
mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}                                                # nothing frozen
cfg = ReLUTrainConfig(epochs=EPOCHS, lr=LR, batch_size=BS, seed=1000, device="cuda", snapshot_epochs=tuple(range(0, EPOCHS + 1, SNAP)),
                      lr_scheduler_eta_min_ratio=ETA_MIN_RATIO, optimizer="sgd", momentum=MOM, parameter_update_mask=mask, show_progress=False)
out = train_relu_mlp_with_optimizer_snapshots(model=model, x_train=x, y_train=y, config=cfg)
print(f"trained in {(time.time() - t0) / 60:.1f} min; snapshots {len(out['checkpoint_snapshots'])}", flush=True)
sr = designed_run_seed_result(out=out, train_config=cfg, model_spec={"input_dim": 100, "hidden_dims": [args.width, 30, 40], "bias": False, "activation": "relu"},
                              data_spec=DATA, tree=tree, design={**{k: (list(vv) if isinstance(vv, tuple) else vv) for k, vv in DESIGN.items()}, "scale": "kaiming", "cues": "free; pole*(v + u_j) renormalised",
                                                                   "probes": "all equal to one off-span vector e (probe 0's design init)", "spare_rows": "orthogonal off-span vectors, orthogonal to e" if args.width == 80 else "none",
                                                                   "layers_3_4_from": "run 11 init" if args.width == 62 else "own build (= run 11's)", "cue_rows": meta["cue_rows"], "flipped_units": meta["flipped_units"], "unit_path_sign": meta["unit_path_sign"].tolist()}, tag=tag)
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
rec = {"tag": tag, "width1": args.width, "data_seed": args.data_seed, "n_train": N_TRAIN, "batch": BS, "epochs": EPOCHS, "lr": LR, "logit_rms_init": lg["rms_logit"], "final_epoch_loss": float(L[-1]), "min_epoch_loss": float(L.min()),
       "train_log_loss_end": tr["log_loss"], "test_error_end": te["zero_one_loss"], "test_error_strict_end": te["zero_one_loss_strict"], "zero_logit_frac_end": te["zero_logit_fraction"],
       "epoch_to_0.3": first(0.3), "epoch_to_0.2": first(0.2), "epoch_to_0.1": first(0.1), "epoch_to_0.05": first(0.05), "epoch_to_0.03": first(0.03),
       **{f"on_target_L{l}_end": float(pe[f"on_target_L{l}"].iloc[-1]) for l in range(4)}, "nodes_covered_end": int(pe.nodes_covered.iloc[-1]), "wall_min": (time.time() - t0) / 60}
pd.DataFrame([rec]).to_csv(f"{OUT}/summary.csv", index=False)
json.dump({"width1": args.width, "data_seed": args.data_seed, "probes": "all equal to one off-span vector", "spare_rows": "orthogonal off-span" if args.width == 80 else "none", "cues": "free; pole*(v + u_j) renormalised",
           "n_train": N_TRAIN, "epochs": EPOCHS, "batch": BS, "lr": LR, "eta_min_ratio": ETA_MIN_RATIO, "momentum": MOM, "snapshot_every_epochs": SNAP, "design": {k: (list(vv) if isinstance(vv, tuple) else vv) for k, vv in DESIGN.items()}, "data": DATA}, open(f"{OUT}/manifest.json", "w"), indent=2)
print(f"{tag}: final epoch loss {rec['final_epoch_loss']:.4f}, test error {te['zero_one_loss']:.4f}; epochs to 0.3/0.2/0.1/0.05/0.03: {rec['epoch_to_0.3']}/{rec['epoch_to_0.2']}/{rec['epoch_to_0.1']}/{rec['epoch_to_0.05']}/{rec['epoch_to_0.03']}; "
      f"nodes covered {rec['nodes_covered_end']}/31; on-target end {[round(rec[f'on_target_L{l}_end'], 2) for l in range(4)]}; {rec['wall_min']:.1f} min", flush=True)
print("ALL DONE", flush=True)
