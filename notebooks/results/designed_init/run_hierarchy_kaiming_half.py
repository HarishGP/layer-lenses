"""Hierarchy design at Kaiming row norms scaled down by 2 (row norm 0.707 in every layer), linear cues, seeds 0-3,
cues free vs frozen-then-released at 10k; SGD batch 1000, lr 0.02, momentum 0.9, constant LR; stop at trailing loss 0.05
or 600k steps; whole sweep capped at 110 minutes."""
import json, time, numpy as np, pandas as pd
from layer_lenses.designed_init import HierarchyDesignSpec, build_hierarchy_model, hierarchy_probe_table, logit_scale, standard_row_norms
from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data, train_sweep_run, save_run, save_tree
OUT = "results/designed_init/hierarchy_kaiming_half"; T0 = time.time(); WALL = 110 * 60
MAX_STEPS, STOP_LOSS, BS, LR, MOM, UNFREEZE, SCALE = 600_000, 0.05, 1000, 0.02, 0.9, 10_000, 0.5
tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); save_tree(tree, f"{OUT}/tree.npz")
x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=5178, init_seed=0, batch_size=1, n_train=1_000_000), tree)
rows = []
for seed in range(4):
    for cues in ("free", "unfreeze10k"):
        if time.time() - T0 > WALL:
            print("wall-clock limit; skipping", seed, cues, flush=True); continue
        t0 = time.time(); tag = f"linear_{cues}_kaiming_half_seed{seed}"
        base = HierarchyDesignSpec(seed=seed, cue_mode="linear", subset_mode="random", cue_offspan_ratio=0.2)
        hs = HierarchyDesignSpec(**{**base.__dict__, "row_norms": tuple(SCALE * r for r in standard_row_norms("kaiming", base))})
        model, meta = build_hierarchy_model(hs, tree); lg = logit_scale(model, x, device="cuda")
        remaining = max(60, WALL - (time.time() - T0))
        sp = SweepRunSpec(tree_seed=5178, data_seed=5178, init_seed=seed, batch_size=BS, hidden_dims=(80, 40, 40), lr=LR, momentum=MOM, device="cuda",
                          snapshot_every_steps=2000, max_steps=MAX_STEPS, stop_loss=STOP_LOSS, store_all_layers=True)
        out = train_sweep_run(sp, tree=tree, x_train=x, y_train=y, model=model,
                              freeze_first_layer_rows=meta["cue_rows"] if cues != "free" else None,
                              unfreeze_at_step=UNFREEZE if cues == "unfreeze10k" else None)
        save_run(out, f"{OUT}/runs/{tag}.npz")
        json.dump({"tag": tag, "cues": cues, "design_seed": seed, "row_norms": list(hs.row_norms), "logit_at_init": lg,
                   "design": {k: (list(v) if isinstance(v, tuple) else v) for k, v in hs.__dict__.items()}}, open(f"{OUT}/runs/{tag}.json", "w"))
        steps, L, W1s = out["snapshot_steps"], out["snapshot_loss"], out["weights"]["hidden_layers.0.weight"]
        rec = {"tag": tag, "cues": cues, "design_seed": seed, "row_norm": float(hs.row_norms[0]), "logit_rms_init": lg["rms_logit"],
               "final_step": out["final_step"], "stopped_by": out["stopped_by"], "final_loss": float(L[-1]), "min_loss": float(np.nanmin(L)), "wall_s": time.time() - t0}
        for thr in (0.3, 0.2, 0.1, 0.05):
            hit = np.nonzero(L <= thr)[0]; rec[f"step_to_{thr}"] = int(steps[hit[0]]) if len(hit) else -1
        for s in (10_000, 30_000, 100_000, None):
            i = len(steps) - 1 if s is None else int(np.argmin(np.abs(steps - s))); lab = "end" if s is None else f"{s//1000}k"
            t = hierarchy_probe_table(W1s[i], tree)
            for lvl, g in t.groupby("level"):
                rec[f"on_target_L{lvl}_{lab}"] = float((g.top_node == g.node).mean()); rec[f"cos_target_L{lvl}_{lab}"] = float(g.cos_target.mean())
            C = W1s[i] / np.linalg.norm(W1s[i], axis=1, keepdims=True) @ tree.w_list.T
            rec[f"cues_coarse_top_{lab}"] = int(sum(np.abs(C[k]).argmax() < 15 for k in range(30, 62)))
            rec[f"loss_{lab}"] = float(L[i])
        rows.append(rec); pd.DataFrame(rows).to_csv(f"{OUT}/summary.csv", index=False)
        print(f"{tag:36s} {out['stopped_by']:9s} step {out['final_step']:7d} loss {rec['final_loss']:.3f} (min {rec['min_loss']:.3f})  steps to 0.3/0.2/0.1/0.05: "
              f"{rec['step_to_0.3']}/{rec['step_to_0.2']}/{rec['step_to_0.1']}/{rec['step_to_0.05']}  on-target end {[round(rec[f'on_target_L{l}_end'],2) for l in range(4)]}  "
              f"({rec['wall_s']/60:.1f} min, elapsed {(time.time()-T0)/60:.0f} min)", flush=True)
json.dump({"scale": SCALE, "row_norm": float(SCALE * np.sqrt(2)), "max_steps": MAX_STEPS, "stop_loss": STOP_LOSS, "batch": BS, "lr": LR, "momentum": MOM,
           "unfreeze_at_step": UNFREEZE, "seeds": [0, 1, 2, 3], "cues": ["free", "unfreeze10k"], "cue_mode": "linear"}, open(f"{OUT}/manifest.json", "w"), indent=2)
print("ALL DONE", flush=True)
