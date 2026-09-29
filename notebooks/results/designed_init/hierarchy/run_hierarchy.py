"""Hierarchy design sweep: 2 probes per internal node, widths (80,40,40); {relu, linear} cues x {cues free, frozen} x seeds.
Small-batch SGD with momentum to loss 0.3 or a step cap; wall-clock guard for the whole sweep."""
import json, time, numpy as np, pandas as pd
from layer_lenses.designed_init import HierarchyDesignSpec, build_hierarchy_model, hierarchy_probe_table, NODE_LEVEL
from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data, train_sweep_run, save_run, save_tree
OUT = "results/designed_init/hierarchy"; T_START = time.time(); WALL_LIMIT_S = 110 * 60
SEEDS = list(range(12)); CONDITIONS = [(cm, fz) for cm in ("linear", "relu") for fz in (False, True)]
MAX_STEPS, STOP_LOSS, BS, LR, MOM = 100_000, 0.3, 1000, 0.02, 0.9
tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); save_tree(tree, f"{OUT}/tree.npz")
x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=5178, init_seed=0, batch_size=1, n_train=1_000_000), tree)
rows = []
for seed in SEEDS:
    for cue_mode, frozen in CONDITIONS:
        if time.time() - T_START > WALL_LIMIT_S:
            print("wall-clock limit reached; stopping before", seed, cue_mode, frozen, flush=True); break
        tag = f"{cue_mode}_{'frozen' if frozen else 'free'}_seed{seed}"
        t0 = time.time()
        hs = HierarchyDesignSpec(eps=0.01, seed=seed, cue_mode=cue_mode, subset_mode="random", cue_offspan_ratio=0.2)
        model, meta = build_hierarchy_model(hs, tree)
        sp = SweepRunSpec(tree_seed=5178, data_seed=5178, init_seed=seed, batch_size=BS, hidden_dims=(80, 40, 40), lr=LR, momentum=MOM,
                          device="cuda", snapshot_every_steps=1000, max_steps=MAX_STEPS, stop_loss=STOP_LOSS, store_all_layers=True)
        out = train_sweep_run(sp, tree=tree, x_train=x, y_train=y, model=model, freeze_first_layer_rows=meta["cue_rows"] if frozen else None)
        save_run(out, f"{OUT}/runs/{tag}.npz")
        json.dump({"tag": tag, "cue_mode": cue_mode, "cues_frozen": frozen, "design_seed": seed, "design": {k: (list(v) if isinstance(v, tuple) else v) for k, v in hs.__dict__.items()},
                   "unit_probe": meta["unit_probe"].tolist()}, open(f"{OUT}/runs/{tag}.json", "w"))
        W1s = out["weights"]["hidden_layers.0.weight"]; steps = out["snapshot_steps"]
        rec = {"tag": tag, "cue_mode": cue_mode, "cues_frozen": frozen, "design_seed": seed, "final_step": out["final_step"], "stopped_by": out["stopped_by"],
               "final_loss": float(out["snapshot_loss"][-1]), "wall_s": time.time() - t0}
        for s in (10_000, 30_000, None):
            i = len(steps) - 1 if s is None else int(np.argmin(np.abs(steps - s)))
            t = hierarchy_probe_table(W1s[i], tree); lab = "end" if s is None else f"{s//1000}k"
            for lvl, g in t.groupby("level"):
                rec[f"cos_target_L{lvl}_{lab}"] = float(g.cos_target.mean()); rec[f"on_target_L{lvl}_{lab}"] = float((g.top_node == g.node).mean())
            rec[f"loss_{lab}"] = float(out["snapshot_loss"][i])
        C = W1s[-1] / np.linalg.norm(W1s[-1], axis=1, keepdims=True) @ tree.w_list.T
        rec["cues_coarse_top_end"] = int(sum(np.abs(C[k]).argmax() < 15 for k in range(30, 62)))
        rec["random_coarse_aligned_end"] = int(sum(np.abs(C[k]).argmax() < 15 and np.abs(C[k]).max() > 0.8 for k in range(62, 80)))
        rows.append(rec); pd.DataFrame(rows).to_csv(f"{OUT}/summary.csv", index=False)
        print(f"{tag:24s} {out['stopped_by']:9s} step {out['final_step']:6d} loss {rec['final_loss']:.3f}  on-target by level at end "
              f"{[round(rec[f'on_target_L{l}_end'],2) for l in range(4)]}  ({rec['wall_s']:.0f}s, elapsed {(time.time()-T_START)/60:.0f} min)", flush=True)
    else:
        continue
    break
json.dump({"seeds": SEEDS, "conditions": CONDITIONS, "max_steps": MAX_STEPS, "stop_loss": STOP_LOSS, "batch": BS, "lr": LR, "momentum": MOM,
           "eps": 0.01, "cue_offspan_ratio": 0.2, "subset_mode": "random", "widths": [80, 40, 40]}, open(f"{OUT}/manifest.json", "w"), indent=2)
print("ALL DONE", flush=True)
