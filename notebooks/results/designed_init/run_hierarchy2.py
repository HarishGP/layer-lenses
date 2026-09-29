"""(1) Frozen cues released at 10k steps, relu/linear x 12 seeds, width2=40.  (2) width2=120 (12/6/3/3 units per probe by level),
seed 0, relu/linear x {free, frozen, unfreeze@10k}."""
import json, time, numpy as np, pandas as pd
from layer_lenses.designed_init import HierarchyDesignSpec, build_hierarchy_model, hierarchy_probe_table
from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data, train_sweep_run, save_run, save_tree
T_START = time.time(); WALL_LIMIT_S = 110 * 60
MAX_STEPS, STOP_LOSS, BS, LR, MOM, UNFREEZE = 100_000, 0.3, 1000, 0.02, 0.9, 10_000
tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1))
x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=5178, init_seed=0, batch_size=1, n_train=1_000_000), tree)

def record(out, tag, cue_mode, cues, seed, width2, t0):
    W1s = out["weights"]["hidden_layers.0.weight"]; steps = out["snapshot_steps"]
    rec = {"tag": tag, "cue_mode": cue_mode, "cues": cues, "design_seed": seed, "width2": width2, "final_step": out["final_step"],
           "stopped_by": out["stopped_by"], "final_loss": float(out["snapshot_loss"][-1]), "wall_s": time.time() - t0}
    for s in (10_000, 30_000, None):
        i = len(steps) - 1 if s is None else int(np.argmin(np.abs(steps - s)))
        t = hierarchy_probe_table(W1s[i], tree); lab = "end" if s is None else f"{s//1000}k"
        for lvl, g in t.groupby("level"):
            rec[f"cos_target_L{lvl}_{lab}"] = float(g.cos_target.mean()); rec[f"on_target_L{lvl}_{lab}"] = float((g.top_node == g.node).mean())
        rec[f"loss_{lab}"] = float(out["snapshot_loss"][i])
        C = W1s[i] / np.linalg.norm(W1s[i], axis=1, keepdims=True) @ tree.w_list.T
        rec[f"cues_coarse_top_{lab}"] = int(sum(np.abs(C[k]).argmax() < 15 for k in range(30, 62)))
    return rec

def run_one(out_dir, tag, hs, cues, seed, width2, rows):
    t0 = time.time()
    model, meta = build_hierarchy_model(hs, tree)
    sp = SweepRunSpec(tree_seed=5178, data_seed=5178, init_seed=seed, batch_size=BS, hidden_dims=(80, width2, 40), lr=LR, momentum=MOM,
                      device="cuda", snapshot_every_steps=1000, max_steps=MAX_STEPS, stop_loss=STOP_LOSS, store_all_layers=True)
    out = train_sweep_run(sp, tree=tree, x_train=x, y_train=y, model=model,
                          freeze_first_layer_rows=meta["cue_rows"] if cues != "free" else None,
                          unfreeze_at_step=UNFREEZE if cues == "unfreeze10k" else None)
    save_run(out, f"{out_dir}/runs/{tag}.npz")
    json.dump({"tag": tag, "cue_mode": hs.cue_mode, "cues": cues, "design_seed": seed, "width2": width2,
               "design": {k: (list(v) if isinstance(v, tuple) else v) for k, v in hs.__dict__.items()}, "unit_probe": meta["unit_probe"].tolist()},
              open(f"{out_dir}/runs/{tag}.json", "w"))
    rec = record(out, tag, hs.cue_mode, cues, seed, width2, t0); rows.append(rec)
    pd.DataFrame(rows).to_csv(f"{out_dir}/summary.csv", index=False)
    print(f"{tag:28s} {out['stopped_by']:9s} loss {rec['final_loss']:.3f}  on-target by level: 10k {[round(rec[f'on_target_L{l}_10k'],2) for l in range(4)]}  end {[round(rec[f'on_target_L{l}_end'],2) for l in range(4)]}  cues coarse end {rec['cues_coarse_top_end']}  ({rec['wall_s']:.0f}s, elapsed {(time.time()-T_START)/60:.0f} min)", flush=True)

# (1) unfreeze at 10k, width 40, 12 seeds
OUT1 = "results/designed_init/hierarchy_unfreeze"; save_tree(tree, f"{OUT1}/tree.npz"); rows1 = []
for seed in range(12):
    for cue_mode in ("linear", "relu"):
        if time.time() - T_START > WALL_LIMIT_S: break
        hs = HierarchyDesignSpec(eps=0.01, seed=seed, cue_mode=cue_mode, subset_mode="random", cue_offspan_ratio=0.2)
        run_one(OUT1, f"{cue_mode}_unfreeze10k_seed{seed}", hs, "unfreeze10k", seed, 40, rows1)
# (2) width2 = 120, seed 0
OUT2 = "results/designed_init/hierarchy_w120"; save_tree(tree, f"{OUT2}/tree.npz"); rows2 = []
for cue_mode in ("linear", "relu"):
    for cues in ("free", "frozen", "unfreeze10k"):
        if time.time() - T_START > WALL_LIMIT_S: break
        hs = HierarchyDesignSpec(eps=0.01, seed=0, cue_mode=cue_mode, subset_mode="random", cue_offspan_ratio=0.2,
                                 width2=120, units_per_level=(12, 6, 3, 3), layer3_scale=1 / np.sqrt(60))
        run_one(OUT2, f"{cue_mode}_{cues}_w120_seed0", hs, cues, 0, 120, rows2)
json.dump({"unfreeze_at_step": UNFREEZE, "max_steps": MAX_STEPS, "stop_loss": STOP_LOSS, "batch": BS, "lr": LR, "momentum": MOM,
           "w120_units_per_level": [12, 6, 3, 3], "w120_layer3_scale": float(1 / np.sqrt(60))}, open(f"{OUT2}/manifest.json", "w"), indent=2)
print("ALL DONE", flush=True)
