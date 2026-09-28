"""Flow sweep for the designed init: 16 probe-sign patterns + no-inert-group variants + a long base run."""
import json, time, numpy as np, pandas as pd, torch
from layer_lenses.designed_init import *
from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data
OUT = "results/designed_init"
tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1))
x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=5178, init_seed=0, batch_size=1, n_train=1_000_000), tree)
EPS, LR = 1e-3, 3e-4
variants = [("signs_" + "".join("p" if s > 0 else "m" if s < 0 else "z" for s in signs), signs, 1500) for signs in sign_flip_variants()]
variants += [("signs_zzmm", (0, 0, -1, -1), 1500), ("signs_mmzz", (-1, -1, 0, 0), 1500), ("base_long", (1, 1, -1, -1), 12000)]
rows = []
for tag, signs, steps in variants:
    t0 = time.time()
    m = build_designed_model(DesignedInitSpec(eps=EPS, seed=0, group_sign=tuple(signs)), tree)
    g0 = probe_gradient_report(m, x, y, tree, device="cuda", per_group=False)["full"]
    out = run_designed_flow(m, x, y, lr=LR, steps=steps, snapshot_every=50, device="cuda")
    traj = neuron_trajectory_table(out, tree)
    traj["loss"] = [np.nan] + [out["epoch_losses"][s - 1] for s in traj.step[1:]]
    traj.to_csv(f"{OUT}/traj_{tag}.csv", index=False)
    W1 = {s: out["checkpoint_snapshots"][s]["hidden_layers.0.weight"].numpy() for s in (0, steps)}
    np.savez(f"{OUT}/first_layer_{tag}.npz", W0=W1[0], W_end=W1[steps], losses=np.array(out["epoch_losses"]))
    c = traj.iloc[-1]; lvl4 = float(sum(c[f"cos_u{j}"] ** 2 for j in range(15, 31)))
    target = 0.9 * c.cos_u0
    reach = traj.step[(np.sign(c.cos_u0) * traj.cos_u0 >= abs(target))].min() if abs(c.cos_u0) > 0.05 else -1
    rows.append({"tag": tag, "signs": str(tuple(signs)), "steps": steps, "init_cos_u0": g0["cos_u0"], "init_grad_norm": g0["grad_norm"],
                 "final_cos_u0": float(c.cos_u0), "final_norm_over_eps": float(c.norm / EPS), "final_level4_energy": lvl4,
                 "final_off_span": float(c.off_span_fraction), "step_to_90pct": int(reach), "loss_start": out["epoch_losses"][0],
                 "loss_end": out["epoch_losses"][-1], "wall_s": time.time() - t0})
    pd.DataFrame(rows).to_csv(f"{OUT}/summary.csv", index=False)
    print(f"{tag:12s} signs {str(signs):16s} init cos {g0['cos_u0']:+.3f} -> final cos {c.cos_u0:+.3f}  (norm x{c.norm/EPS:.1f}, level4 {lvl4:.2f}, loss {out['epoch_losses'][-1]:.4f}, {time.time()-t0:.0f}s)", flush=True)
json.dump({"eps": EPS, "lr": LR, "tree_seed": 5178, "data_seed": 5178, "n_train": 1_000_000, "momentum": 0.0}, open(f"{OUT}/manifest.json", "w"), indent=2)
print("ALL DONE")
