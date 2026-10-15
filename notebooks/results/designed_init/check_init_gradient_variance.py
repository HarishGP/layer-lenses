"""Population gradient on the first layer at init, projected on the 31 ODT normals: variance of the components by node level.
Cases: (a) He-init 62-60 two-layer net; (b) He-init 256-256; (c) run 56's init with only W2 randomised (Gaussian rows renormalised to
sqrt 2, run 56's W1 and path-sign readout); (d) run 56's actual init. Several seeds each; gradient of the mean logistic loss over 1M points."""
import numpy as np, torch, pickle, pandas as pd
from layer_lenses.designed_init import NODE_LEVEL, node_ancestors, cue_row_identity
from layer_lenses.init_sweep import SweepRunSpec, build_data, load_tree
torch.manual_seed(0); dev = "cuda"
RUN = "results/designed_init/hd2_run56_phased2_2L_outx80_10M"; tree = load_tree(f"{RUN}/tree.npz"); U = tree.w_list; Ut = torch.as_tensor(U, dtype=torch.float32)
x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=31337, init_seed=0, batch_size=1, n_train=1_000_000), tree)
X = torch.from_numpy(x).float().to(dev); Y = torch.from_numpy(y.astype(np.float32)).to(dev)
z = np.load(f"{RUN}/replay/weights.npz"); W1_56, W2_56, Wo_56 = (z[k][0].astype(np.float32) for k in ["w__hidden_layers.0.weight", "w__hidden_layers.1.weight", "w__output_layer.weight"])
meta = pickle.load(open(f"{RUN}/replay/meta.pkl", "rb")); cue_rows = meta["design"]["cue_rows"]; cid = cue_row_identity(W1_56.astype(np.float64), tree, cue_rows)     # {(node, pole): row}
def grad_W1(W1, W2, Wo, chunk=100_000):
    """Population gradient of the mean logistic loss wrt W1 (two hidden layers, ReLU, no bias)."""
    W1t = torch.as_tensor(W1, device=dev).requires_grad_(True); W2t = torch.as_tensor(W2, device=dev); Wot = torch.as_tensor(Wo, device=dev)
    for i in range(0, len(X), chunk):
        xb, yb = X[i:i + chunk], Y[i:i + chunk]; f = (torch.relu(torch.relu(xb @ W1t.T) @ W2t.T) @ Wot.T).squeeze(1)
        (torch.nn.functional.softplus(-yb * f).sum() / len(X)).backward()
    return W1t.grad.cpu().numpy().astype(np.float64)
def side_of(j, m):
    c = m
    while (c - 1) // 2 != j: c = (c - 1) // 2
    return +1 if c == 2 * j + 2 else -1
def summarise(name, G, rows=None):
    """rms of the gradient components by node level (over rows x nodes of that level), ratio ladder, and where each row's largest component sits."""
    rows = np.arange(len(G)) if rows is None else rows; C = G[rows] @ U.T; gn = np.linalg.norm(G[rows], axis=1)
    rms = {l: np.sqrt(np.mean(C[:, NODE_LEVEL == l] ** 2)) for l in range(5)}; mean_abs_rowmean = {l: np.abs(C[:, NODE_LEVEL == l].mean(0)).mean() for l in range(5)}
    inspan = np.sqrt(np.mean(np.sum(C ** 2, 1))); offspan = np.sqrt(np.mean(gn ** 2 - np.sum(C ** 2, 1)))
    Ci = C[:, :15]; argmax_level = NODE_LEVEL[np.abs(Ci).argmax(1)]; frac_max_at = {l: float((argmax_level == l).mean()) for l in range(4)}
    return {"case": name, **{f"rms L{l}": rms[l] for l in range(5)}, "var ratio L0:L1:L2:L3 (L3 = 1)": " : ".join(f"{rms[l] ** 2 / rms[3] ** 2:.2f}" for l in range(4)), "mean over rows |C| by level (coherent part)": " / ".join(f"{mean_abs_rowmean[l]:.1e}" for l in range(5)),
            "in-span rms / off-span rms": f"{inspan:.2e} / {offspan:.2e}", "frac of rows whose largest internal component is at level 0/1/2/3": " / ".join(f"{frac_max_at[l]:.2f}" for l in range(4)), "max over rows |C_root| / row grad norm": float(np.max(np.abs(C[:, 0]) / gn))}
recs = []; lens_corr = []
rng = np.random.default_rng(0)
for seed in range(10):
    W1 = (rng.standard_normal((62, 100)) * np.sqrt(2 / 100)).astype(np.float32); W2 = (rng.standard_normal((60, 62)) * np.sqrt(2 / 62)).astype(np.float32); Wo = (rng.standard_normal((1, 60)) * np.sqrt(2 / 60)).astype(np.float32)
    recs.append(summarise(f"(a) He 62-60 seed {seed}", grad_W1(W1, W2, Wo)))
for seed in range(3):
    W1 = (rng.standard_normal((256, 100)) * np.sqrt(2 / 100)).astype(np.float32); W2 = (rng.standard_normal((256, 256)) * np.sqrt(2 / 256)).astype(np.float32); Wo = (rng.standard_normal((1, 256)) * np.sqrt(2 / 256)).astype(np.float32)
    recs.append(summarise(f"(b) He 256-256 seed {seed}", grad_W1(W1, W2, Wo)))
for seed in range(10):
    W2 = rng.standard_normal((60, 62)); W2 = (W2 / np.linalg.norm(W2, axis=1, keepdims=True) * np.sqrt(2)).astype(np.float32)
    G = grad_W1(W1_56, W2, Wo_56); recs.append(summarise(f"(c) run 56 init, W2 random seed {seed} [probes only]", G, rows=np.arange(30)))
    # lens formula for the push on probe r along internal node a: sum_u W2[u,r] Wout[u] sum_{m under a} side_a(m) c_{u,m},  c_{u,m} = sum over m's cue rows of W2[u, row] x pole
    c = np.zeros((60, 31))
    for (m, pole), r in cid.items(): c[:, m] += W2[:, r] * pole
    pred = np.zeros((30, 15)); C = G[:30] @ U.T
    for a in range(15):
        leaves = [m for m in range(15, 31) if a in node_ancestors(m)]; contrast = sum(side_of(a, m) * c[:, m] for m in leaves)       # (60,)
        pred[:, a] = (W2[:, :30] * Wo_56.ravel()[:, None] * contrast[:, None]).sum(0)
    lens_corr.append(np.corrcoef(pred.ravel(), C[:, :15].ravel())[0, 1])
recs.append(summarise("(d) run 56 init as designed [probes only]", grad_W1(W1_56, W2_56, Wo_56), rows=np.arange(30)))
df = pd.DataFrame(recs); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(df.drop(columns=[f"rms L{l}" for l in range(5)]).to_string(index=False))
print("\nrms of the components by level (L0..L4), first rows of each case:"); print(df[["case"] + [f"rms L{l}" for l in range(5)]].iloc[[0, 10, 13, 23]].to_string(index=False))
print(f"\n(c) lens formula vs measured components on internal nodes (probes x 15 nodes): correlation per seed {np.round(lens_corr, 3).tolist()}")
agg = {}
for key, sel in [("(a)", df.case.str.startswith("(a)")), ("(b)", df.case.str.startswith("(b)")), ("(c)", df.case.str.startswith("(c)"))]:
    r = df[sel]; v = np.array([[float(s) for s in row.split(" : ")] for row in r["var ratio L0:L1:L2:L3 (L3 = 1)"]]); agg[key] = v.mean(0)
print("\nmean variance ladder L0:L1:L2:L3 (L3 = 1); the leaf-count prediction is 8 : 4 : 2 : 1 ->", {k: np.round(v, 2).tolist() for k, v in agg.items()})
