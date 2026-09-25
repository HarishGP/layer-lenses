import numpy as np, torch
from layer_lenses.relu_analysis import load_seed_result_pickle
torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
P = {"FB-SGD":"seed_result_replayable_5_40_40_40_1Mtrain_1Mbatch_sgd_momentum_9e-1.pkl",
     "FB-Adam":"seed_result_replayable_5_40_40_40_1Mtrain_1Mbatch_adam.pkl",
     "MB-SGD":"seed_result_replayable_5_40_40_40_1Mtrain_10kbatch_sgd_momentum_9e-1.pkl",
     "MB-Adam":"seed_result_replayable_5178_5_40_40_40_1Mtrain_10kbatch_adam_randomhyper_noshuffle.pkl"}
R = {k: load_seed_result_pickle("results/cache/"+v) for k, v in P.items()}
U = torch.from_numpy(R["FB-SGD"]["tree"].w_list).double()
level = np.array([0] + [1]*2 + [2]*4 + [3]*8 + [4]*16)
print("Row space of the first layer: fraction of each ODT normal captured, ||P_rowspace(W1) u_j||^2, averaged per level.")
print("A random 40-dim subspace of R^100 captures 0.40 of any direction. Also: max |cos| of any single neuron with the level's nodes, and the")
print("effective singular-value weight of the level-4 directions inside W1 (||W1 u_j|| relative to ||W1 u_j|| for level 0).\n")
print(f"{'run':8s} {'epoch':>6s}  captured fraction per level L0..L4        max single-neuron |cos| per level       ||W1 u|| per level (relative to L0)   rank(W1)")
for k, sr in R.items():
    ck = sr["out"]["checkpoint_snapshots"]
    for e in ([0, max(ck)] if k in ("FB-SGD","FB-Adam") else [max(ck)]):
        W = ck[e]["hidden_layers.0.weight"].double()
        Q, _ = torch.linalg.qr(W.T)                      # orthonormal basis of the row space (100 x 40)
        cap = ((U @ Q)**2).sum(dim=1)                    # (31,)
        C = (W / W.norm(dim=1, keepdim=True)) @ U.T
        mx = C.abs().max(dim=0).values
        resp = (W @ U.T).norm(dim=0)                     # ||W1 u_j||
        capL = [float(cap[level==L].mean()) for L in range(5)]
        mxL  = [float(mx[level==L].mean()) for L in range(5)]
        rspL = [float(resp[level==L].mean()/resp[0]) for L in range(5)]
        print(f"{k:8s} {e:6d}  {np.round(capL,2)}   {np.round(mxL,2)}   {np.round(rspL,2)}   {int(torch.linalg.matrix_rank(W))}")
print("\nLevel-4 detail for the two full-batch runs at the end: per-node captured fraction (16 nodes)")
for k in ("FB-SGD", "FB-Adam"):
    ck = R[k]["out"]["checkpoint_snapshots"]; W = ck[max(ck)]["hidden_layers.0.weight"].double()
    Q, _ = torch.linalg.qr(W.T); cap = ((U @ Q)**2).sum(dim=1)
    print(f"  {k:8s}", np.round(cap[15:].numpy(), 2))
# Is the whole 31-dim ODT span inside the row space? energy of W1 outside the ODT span
print("\nfraction of ||W1||_F^2 lying outside the 31-dim ODT span (a random matrix would have 0.69):")
for k, sr in R.items():
    ck = sr["out"]["checkpoint_snapshots"]; W = ck[max(ck)]["hidden_layers.0.weight"].double()
    inside = ((W @ U.T)**2).sum(); print(f"  {k:8s} {float(1 - inside / (W**2).sum()):.3f}")
