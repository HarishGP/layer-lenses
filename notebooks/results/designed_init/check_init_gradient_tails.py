"""Tail probabilities of the init-gradient components: P(|push along u_j| > t x leaf push) per level, for the same cases as check_init_gradient_variance.py.
The leaf push of a row is the rms of its 16 level-4 components (the coherent v signal); thresholds t = 0.25 .. 2."""
import numpy as np, torch, pickle, pandas as pd
from math import erfc, sqrt
class norm:
    @staticmethod
    def sf(t): return 0.5 * erfc(t / sqrt(2))
from layer_lenses.designed_init import NODE_LEVEL
from layer_lenses.init_sweep import SweepRunSpec, build_data, load_tree
dev = "cuda"; RUN = "results/designed_init/hd2_run56_phased2_2L_outx80_10M"; tree = load_tree(f"{RUN}/tree.npz"); U = tree.w_list
x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=31337, init_seed=0, batch_size=1, n_train=1_000_000), tree)
X = torch.from_numpy(x).float().to(dev); Y = torch.from_numpy(y.astype(np.float32)).to(dev)
z = np.load(f"{RUN}/replay/weights.npz"); W1_56, Wo_56 = (z[k][0].astype(np.float32) for k in ["w__hidden_layers.0.weight", "w__output_layer.weight"])
def grad_W1(W1, W2, Wo, chunk=100_000):
    W1t = torch.as_tensor(W1, device=dev).requires_grad_(True); W2t = torch.as_tensor(W2, device=dev); Wot = torch.as_tensor(Wo, device=dev)
    for i in range(0, len(X), chunk):
        xb, yb = X[i:i + chunk], Y[i:i + chunk]; f = (torch.relu(torch.relu(xb @ W1t.T) @ W2t.T) @ Wot.T).squeeze(1)
        (torch.nn.functional.softplus(-yb * f).sum() / len(X)).backward()
    return W1t.grad.cpu().numpy().astype(np.float64)
rng = np.random.default_rng(0); cases = {"(a) He 62-60": [], "(b) He 256-256": [], "(c) run 56 init, W2 random (probes)": []}
for seed in range(10):
    W1 = (rng.standard_normal((62, 100)) * np.sqrt(2 / 100)).astype(np.float32); W2 = (rng.standard_normal((60, 62)) * np.sqrt(2 / 62)).astype(np.float32); Wo = (rng.standard_normal((1, 60)) * np.sqrt(2 / 60)).astype(np.float32)
    cases["(a) He 62-60"].append(grad_W1(W1, W2, Wo) @ U.T)
for seed in range(3):
    W1 = (rng.standard_normal((256, 100)) * np.sqrt(2 / 100)).astype(np.float32); W2 = (rng.standard_normal((256, 256)) * np.sqrt(2 / 256)).astype(np.float32); Wo = (rng.standard_normal((1, 256)) * np.sqrt(2 / 256)).astype(np.float32)
    cases["(b) He 256-256"].append(grad_W1(W1, W2, Wo) @ U.T)
for seed in range(10):
    W2 = rng.standard_normal((60, 62)); W2 = (W2 / np.linalg.norm(W2, axis=1, keepdims=True) * np.sqrt(2)).astype(np.float32)
    cases["(c) run 56 init, W2 random (probes)"].append((grad_W1(W1_56, W2, Wo_56) @ U.T)[:30])
np.savez("results/designed_init/init_gradient_components.npz", **{k.split(")")[0].strip("("): np.concatenate(v) for k, v in cases.items()})
pd.set_option("display.width", 250); THR = [0.25, 0.5, 1.0, 1.5, 2.0]
for name, Cs in cases.items():
    C = np.concatenate(Cs); leaf = np.sqrt(np.mean(C[:, 15:] ** 2, axis=1))                         # per-row leaf push (rms of the 16 level-4 components)
    rows = []
    for l in range(4):
        R = np.abs(C[:, NODE_LEVEL == l]) / leaf[:, None]                                            # |push along u_j| / leaf push, for every (row, node at level l)
        sig = np.sqrt(np.mean((C[:, NODE_LEVEL == l] / leaf[:, None]) ** 2))
        rows.append({"level": l, "n nodes": int((NODE_LEVEL == l).sum()), "sigma (rel. to leaf push)": round(float(sig), 3), **{f"P(> {t} x leaf)": float((R > t).mean()) for t in THR}, **{f"Gaussian P(> {t})": float(2 * norm.sf(t / sig)) for t in (1.0, 2.0)},
                     "P(some row > 1 x leaf) per node, per 62 rows": float(1 - (1 - (R > 1.0).mean()) ** 62)})
    df = pd.DataFrame(rows); print(f"\n{name}: {len(C)} rows over seeds"); print(df.to_string(index=False))
    r = df.set_index("level"); print("ratio root / level 3 of P(> t x leaf):", {t: round(float(r.loc[0, f'P(> {t} x leaf)'] / r.loc[3, f'P(> {t} x leaf)']), 1) if r.loc[3, f'P(> {t} x leaf)'] > 0 else "inf" for t in THR}, "| sigma ratio root / level 3:", round(float(r.loc[0, "sigma (rel. to leaf push)"] / r.loc[3, "sigma (rel. to leaf push)"]), 2))
