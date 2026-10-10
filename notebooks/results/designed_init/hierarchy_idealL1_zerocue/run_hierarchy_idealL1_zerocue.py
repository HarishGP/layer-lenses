"""Run 23: run 22 with the CUE BLOCK OF W2 ZEROED at init (the 32 cue rows start with zero fan-out, like run 11's random rows, but already
aligned with the level-4 normals) and each unit's probe weight set to +1 instead of the design's -1 (with the cues gone the -1 would leave every
unit dead: zero pre-activations, zero logits, zero gradients); everything else as run 22. Run 22: run 21 (62-30-40, no random rows, W3 / output from run 11's init) with the FIRST LAYER FROZEN and the 30 probe rows placed
exactly on their designed targets (row 2j = +u_j, row 2j+1 = -u_j, at the cue-row norm sqrt(2)), so layers 2-4 train on an ideal,
fixed first layer: the 15 internal-node detector pairs plus the 16 level-4 cue pairs. Run 21: run 11 with NO random first-layer rows (62-30-40: 30 probes + 32 cues; everything else identical). Run 11: run 10's design (80-30-40, balanced output, path-sign cues, inv_sqrt cue blocks, balanced +-10-quanta third-layer
paths, cues frozen) trained replayably: 1M training points (the first chunk of run 10's data), batch 10k so an epoch is 100
steps, 10 000 epochs (1M steps, as before), lr 0.01 -> 0.0005 cosine (half of run 10), momentum 0.9, snapshots of weights AND
momentum buffers every 10 epochs (1000 steps), so any interval between adjacent checkpoints can be replayed exactly with
`replay_interval_from_snapshot`. Saved with `save_designed_run` (weights.npz, momentum.npz, meta.pkl, each < 50 MB)."""
import json, os, time, numpy as np, pandas as pd, torch
from layer_lenses.designed_init import build_hierarchy_model, hierarchy_probe_table, logit_scale, standard_scale_spec
from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data, build_data_chunked, save_tree
from layer_lenses.relu_training import ReLUTrainConfig, evaluate_relu_mlp
from layer_lenses.relu_analysis import train_relu_mlp_with_optimizer_snapshots, designed_run_seed_result, save_designed_run, load_designed_run
OUT = "results/designed_init/hierarchy_idealL1_zerocue"; T0 = time.time(); os.makedirs(f"{OUT}/runs", exist_ok=True)
N_TRAIN, EPOCHS, BS, LR, MOM, ETA_MIN_RATIO, SNAP = 1_000_000, 10_000, 10_000, 0.01, 0.9, 0.05, 10
DESIGN = dict(seed=0, cue_mode="linear", subset_mode="all", units_per_level=(1, 1, 1, 1), width1=62, width2=30, width3=40, cue_offspan_ratio=0.2,
              output_signs="balanced", path_sign_cues=True, layer3_mode="balanced_paths", layer3_majority=15, cue_block_scale="inv_sqrt")
DATA = dict(kind="sweep_chunked", tree_seed=5178, data_seed=5178, n_train=N_TRAIN, chunk_size=1_000_000, eval_seed=999_777, n_eval=200_000, dim=100, depth=5)
tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); save_tree(tree, f"{OUT}/tree.npz")
x, y = build_data_chunked(SweepRunSpec(tree_seed=5178, data_seed=5178, init_seed=0, batch_size=1, n_train=N_TRAIN), tree)
xe, ye = build_data(SweepRunSpec(tree_seed=5178, data_seed=999_777, init_seed=0, batch_size=1, n_train=200_000), tree)
tag = "linear_frozen_kaiming_cosine_idealL1_zerocue_62_seed0"; t0 = time.time()
hs = standard_scale_spec("kaiming", **DESIGN); model, meta = build_hierarchy_model(hs, tree)
# the 62-row build draws its third-layer 0/1 pattern from a different RNG state than run 11's 80-row build; take layers 3-4 from run 11's stored init so the
# only difference from run 11 is the absence of the 18 random rows (rows 0-61 and W2[:, :62] already coincide exactly)
_s11 = load_designed_run("results/designed_init/hierarchy_replayable/replay")["out"]["checkpoint_snapshots"][0]
with torch.no_grad():
    assert torch.equal(model.hidden_layers[0].weight, _s11["hidden_layers.0.weight"][:62]) and torch.equal(model.hidden_layers[1].weight, _s11["hidden_layers.1.weight"][:, :62])
    model.hidden_layers[2].weight.copy_(_s11["hidden_layers.2.weight"]); model.output_layer.weight.copy_(_s11["output_layer.weight"])
    # probes onto their targets, at the cue-row norm; then the whole first layer is frozen (update mask below)
    from layer_lenses.designed_init import HIERARCHY_TARGETS
    cue_norm = float(model.hidden_layers[0].weight[meta["cue_rows"]].norm(dim=1).mean())
    for p, (node, pole) in enumerate(HIERARCHY_TARGETS):
        model.hidden_layers[0].weight[p] = torch.as_tensor(pole * cue_norm * tree.w_list[node], dtype=torch.float32)
    model.hidden_layers[1].weight[:, meta["cue_rows"]] = 0.0                     # cue block of W2 zeroed: cues start with zero fan-out
    # the design gives each unit its probe with weight -1 (the probe is a veto against the cues); with the cues gone every unit would be dead
    # (pre-activation <= 0 everywhere, zero logits, zero gradients), so the probe weight is set to +1: unit 2j starts as relu(u_j . x) itself
    for u in range(model.hidden_layers[1].weight.shape[0]):
        model.hidden_layers[1].weight[u, u] = model.hidden_layers[1].weight[u, u].abs()
    assert float(model.hidden_layers[1].weight[:, 30:].abs().max()) == 0.0 and float(model.hidden_layers[1].weight.diag().min()) > 0
print(f"probes set to +-u_j at row norm {cue_norm:.4f}; first layer frozen; W2 cue block zeroed (|W2| now {float(model.hidden_layers[1].weight.norm()):.2f})", flush=True)
lg = logit_scale(model, x, device="cuda")
print(f"{tag}: data {x.shape} {x.dtype}; flipped units {meta['flipped_units']}, zero-path units {meta['zero_path_units']}, logit rms {lg['rms_logit']:.3f}", flush=True)
mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}; mask["hidden_layers.0.weight"][:] = 0.0          # whole first layer frozen
cfg = ReLUTrainConfig(epochs=EPOCHS, lr=LR, batch_size=BS, seed=1000, device="cuda", snapshot_epochs=tuple(range(0, EPOCHS + 1, SNAP)),
                      lr_scheduler_eta_min_ratio=ETA_MIN_RATIO, optimizer="sgd", momentum=MOM, parameter_update_mask=mask, show_progress=False)
out = train_relu_mlp_with_optimizer_snapshots(model=model, x_train=x, y_train=y, config=cfg)
print(f"trained in {(time.time() - t0) / 60:.1f} min; snapshots {len(out['checkpoint_snapshots'])}", flush=True)
sr = designed_run_seed_result(out=out, train_config=cfg, model_spec={"input_dim": 100, "hidden_dims": [62, 30, 40], "bias": False, "activation": "relu"},
                              data_spec=DATA, tree=tree, design={**{k: (list(v) if isinstance(v, tuple) else v) for k, v in DESIGN.items()}, "scale": "kaiming", "cues": "frozen",
                                                                   "cue_rows": meta["cue_rows"], "flipped_units": meta["flipped_units"], "unit_path_sign": meta["unit_path_sign"].tolist(), "layers_3_4_from": "run 11 init (hierarchy_replayable)", "first_layer": "frozen; probes placed on +-u_j at the cue-row norm", "w2_cue_block": "zero at init", "w2_probe_weight": "+1", "w2_probe_weight": "+1 (design: -1)"}, tag=tag)
save_designed_run(sr, f"{OUT}/replay")
ck = out["checkpoint_snapshots"]; epochs = sorted(ck)
m_end = out["model"]; tr = evaluate_relu_mlp(model=m_end, x_eval=x, y_eval=y, device="cuda"); te = evaluate_relu_mlp(model=m_end, x_eval=xe, y_eval=ye, device="cuda")
per_epoch = []
for e in epochs:
    W1 = ck[e]["hidden_layers.0.weight"].numpy(); t = hierarchy_probe_table(W1, tree); C = W1 / np.linalg.norm(W1, axis=1, keepdims=True) @ tree.w_list.T
    per_epoch.append({"epoch": e, "step": e * (N_TRAIN // BS), "loss": np.nan if e == 0 else float(np.mean(out["epoch_losses"][max(0, e - SNAP):e])),
                      **{f"on_target_L{l}": float((g.top_node == g.node).mean()) for l, g in t.groupby("level")}, **{f"cos_target_L{l}": float(g.cos_target.mean()) for l, g in t.groupby("level")},
                      "nodes_covered": len(set(np.abs(C).argmax(1)[np.abs(C).max(1) > 0.95]))})
pe = pd.DataFrame(per_epoch); pe.to_csv(f"{OUT}/per_epoch_{tag}.csv", index=False)
L = np.array(out["epoch_losses"]); first = lambda thr: int(np.nonzero(L <= thr)[0][0] + 1) if (L <= thr).any() else -1
rec = {"tag": tag, "n_train": N_TRAIN, "batch": BS, "epochs": EPOCHS, "steps_per_epoch": N_TRAIN // BS, "steps": EPOCHS * (N_TRAIN // BS), "lr": LR, "logit_rms_init": lg["rms_logit"],
       "final_epoch_loss": float(L[-1]), "min_epoch_loss": float(L.min()), "train_log_loss_end": tr["log_loss"], "test_error_end": te["zero_one_loss"], "test_error_strict_end": te["zero_one_loss_strict"],
       "zero_logit_frac_end": te["zero_logit_fraction"], "epoch_to_0.3": first(0.3), "epoch_to_0.2": first(0.2), "epoch_to_0.1": first(0.1), "epoch_to_0.05": first(0.05),
       **{f"on_target_L{l}_end": float(pe[f"on_target_L{l}"].iloc[-1]) for l in range(4)}, "nodes_covered_end": int(pe.nodes_covered.iloc[-1]), "wall_min": (time.time() - t0) / 60}
pd.DataFrame([rec]).to_csv(f"{OUT}/summary.csv", index=False)
json.dump({"first_layer": "frozen, probes on targets", "w2_cue_block": "zero at init", "w2_probe_weight": "+1", "n_train": N_TRAIN, "epochs": EPOCHS, "batch": BS, "steps_per_epoch": N_TRAIN // BS, "lr": LR, "eta_min_ratio": ETA_MIN_RATIO, "momentum": MOM, "snapshot_every_epochs": SNAP,
           "scale": "kaiming", "cues": "frozen", "design": {k: (list(v) if isinstance(v, tuple) else v) for k, v in DESIGN.items()}, "data": DATA}, open(f"{OUT}/manifest.json", "w"), indent=2)
print(f"{tag}: final epoch loss {rec['final_epoch_loss']:.4f}, train log loss {tr['log_loss']:.4f}, test error {te['zero_one_loss']:.4f} (strict {te['zero_one_loss_strict']:.4f}); "
      f"epochs (of 100 steps) to 0.3/0.2/0.1/0.05: {rec['epoch_to_0.3']}/{rec['epoch_to_0.2']}/{rec['epoch_to_0.1']}/{rec['epoch_to_0.05']}; nodes covered {rec['nodes_covered_end']}/31; "
      f"on-target end {[round(rec[f'on_target_L{l}_end'], 2) for l in range(4)]}; {rec['wall_min']:.1f} min", flush=True)
print("ALL DONE", flush=True)
