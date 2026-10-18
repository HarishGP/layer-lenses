"""Simulated population gradient flow for a designed two-layer init: full-batch-style SGD whose minibatch gradient is projected at every
step onto the symmetric subspace (average over the orbits of the tree-flip / label-negation group, signs included, forced-zero orbits
set to 0; see layer_lenses.symmetric_flow). The projection removes the sample's asymmetric part of the gradient, so the trajectory is an
estimate of the population flow (plain SGD discretisation, noise only within the 157-dim symmetric subspace).
  run 72: --init-run hd2_run70_symcue_partner --epochs 500 --freeze-cues 500 --schedule-epochs 1000 (phase 1 of run 70's protocol: plain SGD
          lr 0.1 cosine over 1000 epochs, batch 10k, 10M points, cues frozen, output rate x1), snapshots every epoch, replay format.
  run 73: --init-run hd2_run70_symcue_partner --epochs 1500 --freeze-cues 1000 (two phases, phase 1 doubled; cosine over the 1500 epochs).
  run 74: --init-run hd2_run60_sgd_lr1_1000 --epochs 1000 --freeze-cues 500 (run 60's init and protocol under the symmetrised flow).
  run 75: --init-run hd2_run60_sgd_lr1_1000 --partner-probe-weight 1 --epochs 1000 --freeze-cues 500 (run 60's init + partner links +1).
  runs 76-78: the same three inits in a single phase, nothing frozen (--freeze-cues 0), 1500 / 1000 / 1000 epochs.
"""
import argparse, json, os, pickle, time, numpy as np, pandas as pd, torch
from layer_lenses.designed_init import cue_row_identity, hierarchy_probe_table, NODE_LEVEL, HIERARCHY_TARGETS
from layer_lenses.init_sweep import SweepRunSpec, build_data, build_data_chunked, load_tree, save_tree
from layer_lenses.symmetric_flow import symmetric_orbits, symmetrise
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--init-run", required=True)
ap.add_argument("--epochs", type=int, default=500); ap.add_argument("--freeze-cues", type=int, default=0, help="epochs with the cue rows frozen (phase 1)"); ap.add_argument("--schedule-epochs", type=int, default=None, help="length of the cosine schedule (default: --epochs)"); ap.add_argument("--partner-probe-weight", type=float, default=None, help="if given, set every unit's weight on the pole partner of its own probe to this value"); ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--out-scale", type=float, default=1.0); ap.add_argument("--w2-scale", type=float, default=1.0)
ap.add_argument("--n-train", type=int, default=10_000_000); ap.add_argument("--batch", type=int, default=10_000); ap.add_argument("--data-seed", type=int, default=5178); ap.add_argument("--no-symmetrise", action="store_true")
args = ap.parse_args(); OUT = f"results/designed_init/{args.out}"; os.makedirs(f"{OUT}/runs", exist_ok=True); dev = "cuda"; t0 = time.time()
src = f"results/designed_init/{args.init_run}/replay"; tree = load_tree(f"results/designed_init/{args.init_run}/tree.npz"); save_tree(tree, f"{OUT}/tree.npz"); U = tree.w_list
z = np.load(f"{src}/weights.npz"); meta = pickle.load(open(f"{src}/meta.pkl", "rb")); d = meta["design"]
W1, W2, Wo = (z[k][0].astype(np.float64) for k in ["w__hidden_layers.0.weight", "w__hidden_layers.1.weight", "w__output_layer.weight"])
cid = cue_row_identity(W1, tree, d["cue_rows"]); units = list(zip(d["unit_node"], d["unit_block_sign"], d["unit_path_sign"])); CUE_ROWS = list(d["cue_rows"])
if args.partner_probe_weight is not None:
    for u_ in range(60):
        pr_ = [p_ for p_ in range(30) if W2[u_, p_] < 0][0]; W2[u_, pr_ + 1 if pr_ % 2 == 0 else pr_ - 1] = args.partner_probe_weight
    print(f"partner probe weight set to {args.partner_probe_weight}", flush=True)
O = W1[:30] - (W1[:30] @ U.T) @ U; O /= np.linalg.norm(O, axis=1, keepdims=True); B = np.vstack([U, O]); assert np.allclose(B @ B.T, np.eye(61), atol=1e-6)
orb = symmetric_orbits(cid, units); print(f"orbits: {orb['n_orbits']}, free {orb['n_free']}; init run {args.init_run}", flush=True)
Bt = torch.as_tensor(B, dtype=torch.float32, device=dev)
x, y = build_data_chunked(SweepRunSpec(tree_seed=5178, data_seed=args.data_seed, init_seed=0, batch_size=1, n_train=args.n_train), tree); xe, ye = build_data(SweepRunSpec(tree_seed=5178, data_seed=999_777, init_seed=0, batch_size=1, n_train=200_000), tree)
X = torch.from_numpy(x).float().to(dev); Y = torch.from_numpy(y.astype(np.float32)).to(dev); Xe = torch.from_numpy(xe).float().to(dev); Ye = torch.from_numpy(ye.astype(np.float32)).to(dev)
P = [torch.as_tensor(w, dtype=torch.float32, device=dev).requires_grad_(True) for w in (W1, W2, Wo)]
EPOCHS = args.schedule_epochs or args.epochs; TOTAL = args.epochs; assert EPOCHS >= TOTAL; steps_per_epoch = args.n_train // args.batch; total_steps = EPOCHS * steps_per_epoch; cue_mask = torch.ones(62, 1, device=dev); cue_mask[CUE_ROWS] = 0.0
def project(G1, G2, Go, freeze_cues):
    C = (G1 @ Bt.T).cpu().numpy().astype(np.float64); C = symmetrise(C, *orb["W1"], orb["forced_zero"], orb["n_orbits"]); G1s = torch.as_tensor(C, dtype=torch.float32, device=dev) @ Bt
    G2s = torch.as_tensor(symmetrise(G2.cpu().numpy().astype(np.float64), *orb["W2"], orb["forced_zero"], orb["n_orbits"]), dtype=torch.float32, device=dev)
    Gos = torch.as_tensor(symmetrise(Go.cpu().numpy().astype(np.float64).ravel(), *orb["Wout"], orb["forced_zero"], orb["n_orbits"]).reshape(1, -1), dtype=torch.float32, device=dev)
    if freeze_cues: G1s = G1s * cue_mask
    return G1s, G2s, Gos
snaps = {0: [p.detach().cpu().numpy().copy() for p in P]}; losses = []; lrs = []; gen = torch.Generator(device=dev); gen.manual_seed(1000); step = 0
for ep in range(TOTAL):
    freeze = ep < args.freeze_cues; perm = torch.randperm(args.n_train, generator=gen, device=dev); run_loss = 0.0
    for b in range(steps_per_epoch):
        lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * step / total_steps)))                                        # cosine to 5% over EPOCHS, as the runs
        idx = perm[b * args.batch:(b + 1) * args.batch]; xb, yb = X[idx], Y[idx]
        f = (torch.relu(torch.relu(xb @ P[0].T) @ P[1].T) @ P[2].T).squeeze(1); loss = torch.nn.functional.softplus(-yb * f).mean(); loss.backward()
        with torch.no_grad():
            G1, G2, Go = (p.grad for p in P)
            if not args.no_symmetrise: G1, G2, Go = project(G1, G2, Go, freeze)
            elif freeze: G1 = G1 * cue_mask
            P[0] -= lr * G1; P[1] -= lr * args.w2_scale * G2; P[2] -= lr * args.out_scale * Go
            for p in P: p.grad = None
        run_loss += float(loss); step += 1
    losses.append(run_loss / steps_per_epoch); lrs.append(lr); snaps[ep + 1] = [p.detach().cpu().numpy().copy() for p in P]
    if (ep + 1) % 25 == 0 or ep == 0: print(f"epoch {ep + 1}: loss {losses[-1]:.4f}, {(time.time() - t0) / 60:.1f} min", flush=True)
with torch.no_grad(): fe = (torch.relu(torch.relu(Xe @ P[0].T) @ P[1].T) @ P[2].T).squeeze(1); te = float(((fe >= 0) != (Ye > 0)).float().mean())
eps = np.array(sorted(snaps)); names = ["hidden_layers.0.weight", "hidden_layers.1.weight", "output_layer.weight"]
def save_replay(folder):
    """Write the trajectory in the replay format of save_designed_run (weights.npz, an empty optimizer_state.npz, meta.pkl), so load_designed_run,
    the notebooks' load_replay_run and the anatomy notebook read it like any other run. Plain SGD has no optimizer state."""
    from layer_lenses.relu_training import ReLUTrainConfig
    os.makedirs(folder, exist_ok=True); np.savez(f"{folder}/weights.npz", epochs=eps, **{f"w__{k}": np.stack([snaps[e][i] for e in eps]).astype(np.float32) for i, k in enumerate(names)}); np.savez(f"{folder}/optimizer_state.npz", epochs=eps)
    L = np.full(EPOCHS, np.nan); L[:TOTAL] = losses; LR = np.full(EPOCHS, np.nan); LR[:TOTAL] = lrs
    cfg = ReLUTrainConfig(epochs=EPOCHS, lr=args.lr, batch_size=args.batch, seed=1000, device="cuda", snapshot_epochs=tuple(int(e) for e in eps), lr_scheduler_eta_min_ratio=0.05, optimizer="sgd", momentum=0.0, show_progress=False)
    design = {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}, "init_run": args.init_run, "symmetrised_gradient": not args.no_symmetrise, "epochs": TOTAL, "freeze_cues": args.freeze_cues, "schedule_epochs": EPOCHS, "partner_probe_weight": args.partner_probe_weight, "out_scale": args.out_scale, "w2_scale": args.w2_scale}
    meta_out = {"tag": args.tag, "tree": tree, "data_spec": {"kind": "sweep_chunked", "tree_seed": 5178, "data_seed": args.data_seed, "n_train": args.n_train, "chunk_size": 1_000_000, "eval_seed": 999_777, "n_eval": 200_000, "dim": 100, "depth": 5},
                "model_spec": {"input_dim": 100, "hidden_dims": [62, 60], "bias": False, "activation": "relu"}, "train_config": cfg, "design": design,
                "out": {"epoch_losses": L.tolist(), "lr_history": LR.tolist(), "model_state_final": {k: torch.as_tensor(snaps[int(eps[-1])][i]) for i, k in enumerate(names)}},
                "parameter_names": names, "optimizer_state_keys": [], "optimizer_param_group": {"lr": args.lr, "momentum": 0.0, "dampening": 0, "weight_decay": 0.0, "nesterov": False, "maximize": False, "foreach": None, "differentiable": False, "fused": None}}
    pickle.dump(meta_out, open(f"{folder}/meta.pkl", "wb"))
save_replay(f"{OUT}/replay")
per = []
for e in eps:
    W1e = snaps[e][0]; t = hierarchy_probe_table(W1e, tree); Ce = W1e / np.linalg.norm(W1e, axis=1, keepdims=True) @ U.T
    per.append({"epoch": int(e), "step": int(e) * steps_per_epoch, "loss": np.nan if e == 0 else losses[e - 1], **{f"on_target_L{l}": float((g.top_node == g.node).mean()) for l, g in t.groupby("level")}, **{f"cos_target_L{l}": float(g.cos_target.mean()) for l, g in t.groupby("level")}, "nodes_covered": len(set(np.abs(Ce).argmax(1)[np.abs(Ce).max(1) > 0.95]))})
pe = pd.DataFrame(per); pe.to_csv(f"{OUT}/per_epoch_{args.tag}.csv", index=False)
rec = {"tag": args.tag, "init_run": args.init_run, "symmetrised": not args.no_symmetrise, "epochs": TOTAL, "freeze_cues": args.freeze_cues, "schedule_epochs": EPOCHS, "partner_probe_weight": args.partner_probe_weight, "lr": args.lr, "final_epoch_loss": losses[-1], "min_epoch_loss": float(np.min(losses)), "test_error_end": te, "nodes_covered_end": int(pe.nodes_covered.iloc[-1]), **{f"on_target_L{l}_end": float(pe[f"on_target_L{l}"].iloc[-1]) for l in range(4)}, "wall_min": (time.time() - t0) / 60}
pd.DataFrame([rec]).to_csv(f"{OUT}/summary.csv", index=False); json.dump({k: (v if not isinstance(v, np.generic) else float(v)) for k, v in rec.items()}, open(f"{OUT}/manifest.json", "w"), indent=2)
pickle.dump({"design": d, "cue_identity": {str(k): v for k, v in cid.items()}, "basis_off_span": O}, open(f"{OUT}/identities.pkl", "wb"))   # the off-span basis used for the projection
print(f"{args.tag}: final loss {losses[-1]:.4f}, test error {te:.4f}, nodes {rec['nodes_covered_end']}/31, on-target {[round(rec[f'on_target_L{l}_end'], 2) for l in range(4)]}; {rec['wall_min']:.1f} min", flush=True); print("ALL DONE", flush=True)
