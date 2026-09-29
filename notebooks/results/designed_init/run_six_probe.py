"""Six-probe design sweep: +-u_0, +-u_1, +-u_2 probes; relu vs linear cues; cues free vs frozen; short and long."""
import json, time, numpy as np, pandas as pd
from layer_lenses.designed_init import *
from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data
OUT = "results/designed_init"
tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); U = tree.w_list
x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=5178, init_seed=0, batch_size=1, n_train=1_000_000), tree)
EPS, LR = 1e-3, 3e-4
variants = [("six_relu", "relu", False, 1500), ("six_linear", "linear", False, 1500), ("six_linear_long", "linear", False, 6000),
            ("six_relu_frozen", "relu", True, 1500), ("six_linear_frozen", "linear", True, 1500), ("six_linear_frozen_long", "linear", True, 6000)]
rows = []
for tag, cue_mode, frozen, steps in variants:
    t0 = time.time()
    spec = six_probe_spec(cue_mode=cue_mode, eps=EPS, seed=0, layer3="identity", output_scale=20.0)
    m = build_multiprobe_model(spec, tree)
    init = probe_target_report(m, spec, SIX_PROBE_TARGETS, x, y, tree, device="cuda")
    init["tag"] = tag; init.to_csv(f"{OUT}/init_{tag}.csv", index=False)
    out = run_designed_flow(build_multiprobe_model(spec, tree), x, y, lr=LR, steps=steps, snapshot_every=50, device="cuda",
                            freeze_first_layer_rows=cue_rows(spec) if frozen else None)
    trajs = []
    for p, (node, pole) in enumerate(SIX_PROBE_TARGETS):
        t = neuron_trajectory_table(out, tree, neuron=p); t["probe"] = p; t["target_node"] = node; t["target_pole"] = pole
        t["cos_target"] = pole * t[f"cos_u{node}"]; trajs.append(t)
    T = pd.concat(trajs, ignore_index=True); T["loss"] = T.step.map(lambda s: np.nan if s == 0 else out["epoch_losses"][s - 1])
    T.to_csv(f"{OUT}/traj_{tag}.csv", index=False)
    Wend = out["checkpoint_snapshots"][steps]["hidden_layers.0.weight"].numpy(); nr = np.linalg.norm(Wend, axis=1) / EPS
    C = (Wend / np.linalg.norm(Wend, axis=1, keepdims=True)) @ U.T
    np.savez(f"{OUT}/first_layer_{tag}.npz", W_end=Wend, W0=out["checkpoint_snapshots"][0]["hidden_layers.0.weight"].numpy())
    for p, (node, pole) in enumerate(SIX_PROBE_TARGETS):
        rows.append({"tag": tag, "cue_mode": cue_mode, "cues_frozen": frozen, "steps": steps, "probe": p, "target": f"{'+' if pole>0 else '-'}u{node}",
                     "init_cos_target": float(init.cos_target[p]), "init_cos_u0": float(init.cos_u0[p]),
                     "final_cos_target": float(pole * C[p, node]), "final_cos_u0": float(C[p, 0]), "final_top_node": int(np.abs(C[p]).argmax()),
                     "final_level4_energy": float((C[p, 15:]**2).sum()), "final_norm_over_eps": float(nr[p]),
                     "cues_coarse_top": int(sum(np.abs(C[k]).argmax() < 15 for k in range(6, 38))), "cues_median_norm_over_eps": float(np.median(nr[6:38])),
                     "loss_end": out["epoch_losses"][-1], "wall_s": time.time() - t0})
    pd.DataFrame(rows).to_csv(f"{OUT}/summary_six_probe.csv", index=False)
    print(f"{tag:24s} done in {time.time()-t0:.0f}s: final cos_target {[round(float(pole*C[p,node]),2) for p,(node,pole) in enumerate(SIX_PROBE_TARGETS)]}", flush=True)
json.dump({"eps": EPS, "lr": LR, "layer3": "identity", "output_scale": 20.0, "root_size": 4, "child_size": 3}, open(f"{OUT}/manifest_six_probe.json", "w"), indent=2)
print("ALL DONE")
