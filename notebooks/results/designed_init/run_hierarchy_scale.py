"""Hierarchy design rescaled to standard-init row norms ('default' = the model's own init, 'kaiming'); linear cues, seed 0,
cues free / frozen / released at 10k; same SGD settings as the other hierarchy sweeps."""
import json, time, numpy as np, pandas as pd, torch
from layer_lenses.designed_init import standard_scale_spec, build_hierarchy_model, hierarchy_probe_table, logit_scale
from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data, train_sweep_run, save_run, save_tree
OUT = "results/designed_init/hierarchy_scale"; T0 = time.time()
MAX_STEPS, STOP_LOSS, BS, LR, MOM, UNFREEZE = 100_000, 0.3, 1000, 0.02, 0.9, 10_000
tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); save_tree(tree, f"{OUT}/tree.npz")
x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=5178, init_seed=0, batch_size=1, n_train=1_000_000), tree)
rows = []
for scheme in ("default", "kaiming"):
    for cues in ("free", "frozen", "unfreeze10k"):
        t0 = time.time(); tag = f"linear_{cues}_{scheme}_seed0"
        hs = standard_scale_spec(scheme, seed=0, cue_mode="linear", subset_mode="random", cue_offspan_ratio=0.2)
        model, meta = build_hierarchy_model(hs, tree)
        lg = logit_scale(model, x, device="cuda")
        sp = SweepRunSpec(tree_seed=5178, data_seed=5178, init_seed=0, batch_size=BS, hidden_dims=(80, 40, 40), lr=LR, momentum=MOM,
                          device="cuda", snapshot_every_steps=1000, max_steps=MAX_STEPS, stop_loss=STOP_LOSS, store_all_layers=True)
        out = train_sweep_run(sp, tree=tree, x_train=x, y_train=y, model=model,
                              freeze_first_layer_rows=meta["cue_rows"] if cues != "free" else None,
                              unfreeze_at_step=UNFREEZE if cues == "unfreeze10k" else None)
        save_run(out, f"{OUT}/runs/{tag}.npz")
        json.dump({"tag": tag, "scheme": scheme, "cues": cues, "row_norms": list(hs.row_norms), "logit_at_init": lg,
                   "design": {k: (list(v) if isinstance(v, tuple) else v) for k, v in hs.__dict__.items()}}, open(f"{OUT}/runs/{tag}.json", "w"))
        W1s = out["weights"]["hidden_layers.0.weight"]; steps = out["snapshot_steps"]
        rec = {"tag": tag, "scheme": scheme, "cues": cues, "row_norm": float(hs.row_norms[0]), "logit_rms_init": lg["rms_logit"],
               "final_step": out["final_step"], "stopped_by": out["stopped_by"], "final_loss": float(out["snapshot_loss"][-1]), "wall_s": time.time() - t0}
        for s in (10_000, 30_000, None):
            i = len(steps) - 1 if s is None else int(np.argmin(np.abs(steps - s)))
            t = hierarchy_probe_table(W1s[i], tree); lab = "end" if s is None else f"{s//1000}k"
            for lvl, g in t.groupby("level"):
                rec[f"on_target_L{lvl}_{lab}"] = float((g.top_node == g.node).mean()); rec[f"cos_target_L{lvl}_{lab}"] = float(g.cos_target.mean())
            rec[f"loss_{lab}"] = float(out["snapshot_loss"][i])
            C = W1s[i] / np.linalg.norm(W1s[i], axis=1, keepdims=True) @ tree.w_list.T
            rec[f"cues_coarse_top_{lab}"] = int(sum(np.abs(C[k]).argmax() < 15 for k in range(30, 62)))
        rows.append(rec); pd.DataFrame(rows).to_csv(f"{OUT}/summary.csv", index=False)
        print(f"{tag:30s} {out['stopped_by']:9s} step {out['final_step']:6d} loss {rec['final_loss']:.3f}  logit@init {lg['rms_logit']:.3f}  on-target end {[round(rec[f'on_target_L{l}_end'],2) for l in range(4)]}  cues coarse {rec['cues_coarse_top_end']}  ({rec['wall_s']:.0f}s)", flush=True)
print("ALL DONE", flush=True)
