"""Run 15: branch of run 11 from its epoch-750 checkpoint (step 75k) with ALL layers trained by AdamW (weight decay 0) from a
fresh optimizer state, lr 1e-3 on the same cosine shape (resumed at epoch 750, T_max 10 000, eta_min ratio 0.05). Only the cue
rows stay frozen, as in run 11. Keeps run 11's data (1M points, batch 10k, epoch = 100 steps), 10 000 epochs in total, and
stores weights and the full Adam state every 10 epochs, so every interval after 750 replays exactly. Compare with run 14
(same optimizer switch with the first layer frozen)."""
import json, os, time, numpy as np, pandas as pd, torch
from layer_lenses.designed_init import hierarchy_probe_table
from layer_lenses.init_sweep import save_tree
from layer_lenses.relu_training import ReLUTrainConfig, evaluate_relu_mlp
from layer_lenses.relu_analysis import (load_designed_run, regenerate_seed_result_data, checkpoint_model_from_state, train_relu_mlp_with_optimizer_snapshots,
                                         designed_run_seed_result, save_designed_run)
SRC = "results/designed_init/hierarchy_replayable"; BRANCH_EPOCH, EPOCHS, SNAP = 750, 10_000, 10
OUT = "results/designed_init/hierarchy_branch750_adam_all"; os.makedirs(OUT, exist_ok=True)
base = load_designed_run(f"{SRC}/replay"); out0 = base["out"]; tree = base["tree"]; cfg0 = base["train_config"]; save_tree(tree, f"{OUT}/tree.npz")
data = regenerate_seed_result_data(base); x, y, xe, ye = data["x_train"], data["y_train"], data["x_eval"], data["y_eval"]
print(f"branching run 11 at epoch {BRANCH_EPOCH} (step {BRANCH_EPOCH * 100}); data {x.shape}", flush=True)
opt_kw = dict(optimizer="adamw", lr=1e-3, momentum=0.0, weight_decay=0.0); t0 = time.time()
tag = "linear_frozen_kaiming_cosine_branch750_adam_all_seed0"
model = checkpoint_model_from_state(out0["model"], out0["checkpoint_snapshots"][BRANCH_EPOCH])
mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}; mask["hidden_layers.0.weight"][base["design"]["cue_rows"]] = 0.0   # cues frozen, as in run 11
cfg = ReLUTrainConfig(epochs=EPOCHS, batch_size=cfg0.batch_size, seed=cfg0.seed, device="cuda", snapshot_epochs=tuple(range(BRANCH_EPOCH, EPOCHS + 1, SNAP)),
                      lr_scheduler_eta_min_ratio=cfg0.lr_scheduler_eta_min_ratio, parameter_update_mask=mask, show_progress=False, **opt_kw)
out = train_relu_mlp_with_optimizer_snapshots(model=model, x_train=x, y_train=y, config=cfg, start_epoch=BRANCH_EPOCH)
print(f"{tag}: trained epochs {BRANCH_EPOCH}..{EPOCHS} in {(time.time() - t0) / 60:.1f} min; snapshots {len(out['checkpoint_snapshots'])}", flush=True)
sr = designed_run_seed_result(out=out, train_config=cfg, model_spec=base["model_spec"], data_spec=base["data_spec"], tree=tree,
                              design={**base["design"], "branch_of": base["tag"], "branch_epoch": BRANCH_EPOCH, "first_layer_frozen": False, "optimizer": opt_kw, "optimizer_state_continued": False}, tag=tag)
save_designed_run(sr, f"{OUT}/replay")
ck = out["checkpoint_snapshots"]; epochs = sorted(ck); m_end = out["model"]
tr = evaluate_relu_mlp(model=m_end, x_eval=x, y_eval=y, device="cuda"); te = evaluate_relu_mlp(model=m_end, x_eval=xe, y_eval=ye, device="cuda")
L = np.array(out["epoch_losses"]); per_epoch = []
for e in epochs:
    W1 = ck[e]["hidden_layers.0.weight"].numpy(); t = hierarchy_probe_table(W1, tree); C = W1 / np.linalg.norm(W1, axis=1, keepdims=True) @ tree.w_list.T
    per_epoch.append({"epoch": e, "step": e * 100, "loss": float(np.nanmean(L[max(BRANCH_EPOCH, e - SNAP):e])) if e > BRANCH_EPOCH else np.nan,
                      **{f"on_target_L{l}": float((g.top_node == g.node).mean()) for l, g in t.groupby("level")}, **{f"cos_target_L{l}": float(g.cos_target.mean()) for l, g in t.groupby("level")},
                      "nodes_covered": len(set(np.abs(C).argmax(1)[np.abs(C).max(1) > 0.95]))})
pe = pd.DataFrame(per_epoch); pe.to_csv(f"{OUT}/per_epoch_{tag}.csv", index=False)
Lv = L[BRANCH_EPOCH:]; first = lambda thr: int(np.nonzero(Lv <= thr)[0][0] + BRANCH_EPOCH + 1) if (Lv <= thr).any() else -1
rec = {"tag": tag, "branch_of": base["tag"], "branch_epoch": BRANCH_EPOCH, "optimizer": opt_kw["optimizer"], "lr": opt_kw["lr"], "final_epoch_loss": float(L[-1]), "min_epoch_loss": float(np.nanmin(L)),
       "train_log_loss_end": tr["log_loss"], "test_error_end": te["zero_one_loss"], "test_error_strict_end": te["zero_one_loss_strict"], "zero_logit_frac_end": te["zero_logit_fraction"],
       "epoch_to_0.1": first(0.1), "epoch_to_0.05": first(0.05), "epoch_to_0.03": first(0.03), "loss_std_1000_3000": float(np.nanstd(L[1000:3000])),
       **{f"on_target_L{l}_end": float(pe[f"on_target_L{l}"].iloc[-1]) for l in range(4)}, "nodes_covered_end": int(pe.nodes_covered.iloc[-1]), "wall_min": (time.time() - t0) / 60}
pd.DataFrame([rec]).to_csv(f"{OUT}/summary.csv", index=False)
json.dump({"branch_of": base["tag"], "branch_epoch": BRANCH_EPOCH, "first_layer_frozen": False, "cues": "frozen", "optimizer": opt_kw, "optimizer_state_continued": False, "epochs": EPOCHS,
           "snapshot_every_epochs": SNAP, "eta_min_ratio": cfg0.lr_scheduler_eta_min_ratio, "batch": cfg0.batch_size}, open(f"{OUT}/manifest.json", "w"), indent=2)
print(f"{tag}: final loss {rec['final_epoch_loss']:.4f}, test error {te['zero_one_loss']:.4f} (strict {te['zero_one_loss_strict']:.4f}, zero-logit {te['zero_logit_fraction']:.3f}); epochs to 0.1/0.05/0.03: {rec['epoch_to_0.1']}/{rec['epoch_to_0.05']}/{rec['epoch_to_0.03']}; "
      f"loss std over epochs 1000-3000: {rec['loss_std_1000_3000']:.4f}; nodes covered {rec['nodes_covered_end']}/31; on-target end {[round(rec[f'on_target_L{l}_end'], 2) for l in range(4)]}; {rec['wall_min']:.1f} min", flush=True)
print("ALL DONE", flush=True)
