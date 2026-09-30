"""Cleaner version of the cosine-annealed Kaiming hierarchy run: 10x data (10M points, chunked float32),
10x batch (10k), first layer of 62 rows (30 probes + 32 cues, no random rows), one second-layer unit per
probe reading every candidate cue, third layer with exactly 15 ones per row, cues frozen for the whole run.
Same schedule as run_hierarchy_kaiming_cosine.py: 1000 epochs x 1000 steps, lr 0.02 -> 0.001 cosine, mom 0.9."""
import json, os, time, numpy as np, pandas as pd, torch
from layer_lenses.designed_init import build_hierarchy_model, hierarchy_probe_table, logit_scale, standard_scale_spec
from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data, build_data_chunked, save_tree
from layer_lenses.relu_training import ReLUTrainConfig, train_relu_mlp, evaluate_relu_mlp
OUT = "results/designed_init/hierarchy_kaiming_cosine_big"; T0 = time.time(); os.makedirs(f"{OUT}/runs", exist_ok=True)
N_TRAIN, EPOCHS, BS, LR, MOM, ETA_MIN_RATIO, SNAP = 10_000_000, 1000, 10_000, 0.02, 0.9, 0.05, 10
DESIGN = dict(seed=0, cue_mode="linear", subset_mode="all", units_per_level=(1, 1, 1, 1), width1=62, width2=30, width3=40, layer3_exact=True, cue_offspan_ratio=0.2)
tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); save_tree(tree, f"{OUT}/tree.npz")
x, y = build_data_chunked(SweepRunSpec(tree_seed=5178, data_seed=5178, init_seed=0, batch_size=1, n_train=N_TRAIN), tree)
xe, ye = build_data(SweepRunSpec(tree_seed=5178, data_seed=999_777, init_seed=0, batch_size=1, n_train=200_000), tree)
print(f"data ready: {x.shape} {x.dtype}, label mean {y.mean():.4f}, {(time.time() - T0) / 60:.1f} min", flush=True)
rows = []
for cues in ("frozen",):
    t0 = time.time(); tag = f"linear_{cues}_kaiming_cosine_big_seed0"
    hs = standard_scale_spec("kaiming", **DESIGN)
    model, meta = build_hierarchy_model(hs, tree); lg = logit_scale(model, x[:1_000_000], device="cuda")
    W2 = model.hidden_layers[1].weight.detach().cpu().numpy()
    print(f"probe entry in its unit: root {abs(W2[0, 0]):.3f}, level-1 {abs(W2[2, 2]):.3f}, level-2 {abs(W2[6, 6]):.3f}, level-3 {abs(W2[14, 14]):.3f}", flush=True)
    mask = None
    if cues == "frozen":
        mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}
        mask["hidden_layers.0.weight"][meta["cue_rows"]] = 0.0
    cfg = ReLUTrainConfig(epochs=EPOCHS, lr=LR, batch_size=BS, seed=1000, device="cuda", snapshot_epochs=tuple(range(0, EPOCHS + 1, SNAP)),
                          lr_scheduler_eta_min_ratio=ETA_MIN_RATIO, optimizer="sgd", momentum=MOM, parameter_update_mask=mask, show_progress=False)
    out = train_relu_mlp(model=model, x_train=x, y_train=y, config=cfg)
    ck = out["checkpoint_snapshots"]; epochs = sorted(ck)
    np.savez(f"{OUT}/runs/{tag}.npz", epochs=np.array(epochs), epoch_losses=np.array(out["epoch_losses"]), lr_history=np.array(out["lr_history"]),
             **{f"w__{k}": np.stack([ck[e][k].numpy() for e in epochs]) for k in ck[epochs[0]]})
    m_end = out["model"]
    tr = evaluate_relu_mlp(model=m_end, x_eval=x[:2_000_000], y_eval=y[:2_000_000], device="cuda"); te = evaluate_relu_mlp(model=m_end, x_eval=xe, y_eval=ye, device="cuda")
    per_epoch = []
    for e in epochs:
        t = hierarchy_probe_table(ck[e]["hidden_layers.0.weight"].numpy(), tree)
        W1 = ck[e]["hidden_layers.0.weight"].numpy(); C = W1 / np.linalg.norm(W1, axis=1, keepdims=True) @ tree.w_list.T
        per_epoch.append({"epoch": e, "loss": np.nan if e == 0 else out["epoch_losses"][e - 1], **{f"on_target_L{l}": float((g.top_node == g.node).mean()) for l, g in t.groupby("level")},
                          **{f"cos_target_L{l}": float(g.cos_target.mean()) for l, g in t.groupby("level")}, "cues_coarse_top": int(sum(np.abs(C[k]).argmax() < 15 for k in range(30, 62)))})
    pe = pd.DataFrame(per_epoch); pe.to_csv(f"{OUT}/per_epoch_{tag}.csv", index=False)
    L = np.array(out["epoch_losses"]); first = lambda thr: int(np.nonzero(L <= thr)[0][0] + 1) if (L <= thr).any() else -1
    rec = {"tag": tag, "cues": cues, "n_train": N_TRAIN, "batch": BS, "logit_rms_init": lg["rms_logit"], "epochs": EPOCHS, "steps": EPOCHS * 1000, "final_epoch_loss": float(L[-1]), "min_epoch_loss": float(L.min()),
           "train_log_loss_end_2M": tr["log_loss"], "test_error_end": te["zero_one_loss"], "epoch_to_0.3": first(0.3), "epoch_to_0.2": first(0.2), "epoch_to_0.1": first(0.1), "epoch_to_0.05": first(0.05),
           **{f"on_target_L{l}_ep10": float(pe.loc[pe.epoch == 10, f'on_target_L{l}'].iloc[0]) for l in range(4)}, **{f"on_target_L{l}_end": float(pe[f'on_target_L{l}'].iloc[-1]) for l in range(4)},
           "cues_coarse_top_end": int(pe.cues_coarse_top.iloc[-1]), "wall_min": (time.time() - t0) / 60}
    rows.append(rec); pd.DataFrame(rows).to_csv(f"{OUT}/summary.csv", index=False)
    print(f"{tag}: final epoch loss {rec['final_epoch_loss']:.4f} (min {rec['min_epoch_loss']:.4f}), train log loss (2M) {tr['log_loss']:.4f}, test error {te['zero_one_loss']:.4f}; "
          f"epochs to 0.3/0.2/0.1/0.05: {rec['epoch_to_0.3']}/{rec['epoch_to_0.2']}/{rec['epoch_to_0.1']}/{rec['epoch_to_0.05']}; on-target ep10 {[round(rec[f'on_target_L{l}_ep10'],2) for l in range(4)]} end {[round(rec[f'on_target_L{l}_end'],2) for l in range(4)]}; {rec['wall_min']:.1f} min", flush=True)
json.dump({"n_train": N_TRAIN, "epochs": EPOCHS, "batch": BS, "lr": LR, "eta_min_ratio": ETA_MIN_RATIO, "momentum": MOM, "scale": "kaiming", "cues": "frozen",
           "design": {k: (list(v) if isinstance(v, tuple) else v) for k, v in DESIGN.items()}}, open(f"{OUT}/manifest.json", "w"), indent=2)
print("ALL DONE", flush=True)
