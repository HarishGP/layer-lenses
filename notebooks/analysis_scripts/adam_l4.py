import numpy as np, torch, math, time
from layer_lenses.relu_analysis import load_seed_result_pickle, regenerate_seed_result_data
from layer_lenses.relu_training import checkpoint_model_from_state, evaluate_relu_mlp
from layer_lenses.odt import _traverse_tree
torch.set_grad_enabled(False)
R = {}
R["FB-SGD"]  = load_seed_result_pickle("results/cache/seed_result_replayable_5_40_40_40_1Mtrain_1Mbatch_sgd_momentum_9e-1.pkl")
R["FB-Adam"] = load_seed_result_pickle("results/cache/seed_result_replayable_5_40_40_40_1Mtrain_1Mbatch_adam.pkl")
R["MB-SGD"]  = load_seed_result_pickle("results/cache/seed_result_replayable_5_40_40_40_1Mtrain_10kbatch_sgd_momentum_9e-1.pkl")
R["MB-Adam-200ep"] = load_seed_result_pickle("results/cache/seed_result_replayable_5178_5_40_40_40_1Mtrain_10kbatch_adam_randomhyper_noshuffle.pkl")
t0 = time.time()
try:
    R["MB-Adam-10k"] = load_seed_result_pickle("results/cache/seed_result_5178_5_40_40_40_1Mtrain_10kbatch_adam_randomhyper_Sep21.pkl")
    print(f"loaded legacy 10k-epoch Adam pickle in {time.time()-t0:.0f}s")
except Exception as ex:
    print("legacy Adam pickle not loaded:", ex)
tree = R["FB-SGD"]["tree"]; U = torch.from_numpy(tree.w_list).double()
for k, sr in R.items():
    assert np.allclose(sr["tree"].w_list, tree.w_list), k
    cfg = sr["train_config"] if "train_config" in sr else None
    if cfg: print(f"{k:14s} optimizer={cfg.optimizer:5s} epochs={cfg.epochs:6d} batch={cfg.batch_size:8d} lr={cfg.lr} mom={cfg.momentum} wd={cfg.weight_decay}")
    else:   print(f"{k:14s} legacy run: epochs={max(sr['out']['checkpoint_snapshots'])}, no stored config (Adam, batch 10k, lr 2e-3, wd 0.01 per scratch.ipynb)")
level = np.array([0] + [1]*2 + [2]*4 + [3]*8 + [4]*16)

def coverage(W, thr):
    C = (W / W.norm(dim=1, keepdim=True)) @ U.T; bst, arg = C.abs().max(dim=1)
    nodes = [int(a) for a, b in zip(arg, bst) if b > thr]
    per_level_nodes = [len(set(j for j in nodes if level[j] == L)) for L in range(5)]
    per_level_neurons = [sum(1 for j in nodes if level[j] == L) for L in range(5)]
    return per_level_nodes, per_level_neurons

data = regenerate_seed_result_data(R["FB-SGD"])
xe, ye = data["x_eval"], data["y_eval"]
# on-path split margins per level for every eval point
margins = xe.astype(np.float64) @ tree.w_list.T + tree.b_list
leaf = _traverse_tree(margins, depth=5)
path = np.zeros((len(xe), 5), dtype=np.int64); p = leaf.copy()
for L in range(4, -1, -1):
    p = (p - 1) // 2; path[:, L] = p
assert (path[:, 0] == 0).all()
pm = np.abs(np.take_along_axis(margins, path, axis=1))     # (N,5) |margin| to the on-path split at each level
closest = pm.argmin(axis=1)
print("\nfraction of eval points whose nearest on-path split is at level 0..4:", np.round(np.bincount(closest, minlength=5)/len(xe), 3))

def final_epoch(sr): return max(sr["out"]["checkpoint_snapshots"])
def preds(sr, e):
    m = checkpoint_model_from_state(sr["out"]["model"], sr["out"]["checkpoint_snapshots"][e]).to("cuda").eval()
    out = []
    for i in range(0, len(xe), 200_000):
        out.append(m(torch.from_numpy(xe[i:i+200_000]).float().cuda()).cpu())
    logits = torch.cat(out).numpy()
    return np.where(logits >= 0, 1, -1)

print("\n=== end of run: metrics, first-layer coverage per level [L0,L1,L2,L3,L4] of [1,2,4,8,16] ===")
print(f"{'run':14s} {'epoch':>6s} {'train loss':>10s} {'test err':>9s}   nodes@0.8        neurons@0.8       nodes@0.6")
for k, sr in R.items():
    e = final_epoch(sr); W = sr["out"]["checkpoint_snapshots"][e]["hidden_layers.0.weight"].double()
    m = checkpoint_model_from_state(sr["out"]["model"], sr["out"]["checkpoint_snapshots"][e])
    tr = evaluate_relu_mlp(model=m, x_eval=data["x_train"], y_eval=data["y_train"], device="cuda")["log_loss"]
    te = evaluate_relu_mlp(model=m, x_eval=xe, y_eval=ye, device="cuda")["zero_one_loss"]
    n8, u8 = coverage(W, 0.8); n6, _ = coverage(W, 0.6)
    print(f"{k:14s} {e:6d} {tr:10.4f} {te:9.4f}   {str(n8):16s} {str(u8):17s} {n6}")

print("\n=== where is the remaining test error?  zero-one error of each run, binned by the level of the nearest on-path split ===")
print(f"{'run':14s}  overall   L0      L1      L2      L3      L4     | share of all errors in L4 bin | error among points with |margin_L4|<0.02 (n={(pm[:,4]<0.02).sum()})")
for k, sr in R.items():
    yhat = preds(sr, final_epoch(sr)); err = (yhat != ye)
    per = [err[closest == L].mean() for L in range(5)]
    share4 = err[closest == 4].sum() / err.sum()
    near4 = err[pm[:, 4] < 0.02].mean()
    print(f"{k:14s}  {err.mean():.4f}  " + "  ".join(f"{v:.4f}" for v in per) + f"   |          {share4:.2f}            |   {near4:.4f}")

print("\n=== FB-Adam over time: loss and level-4 capture ===")
sr = R["FB-Adam"]; ck = sr["out"]["checkpoint_snapshots"]; losses = sr["out"]["epoch_losses"]
eps = sorted(ck.keys()); pick = [e for e in eps if e in (0, 50, 100, 199) or e % (max(eps)//10) == 0 or e == max(eps)]
for e in pick:
    W = ck[e]["hidden_layers.0.weight"].double(); n8, u8 = coverage(W, 0.8)
    print(f"  epoch {e:6d}  train loss {losses[min(e, len(losses)-1)]:.4f}   nodes captured per level {n8}   neurons {u8}")
