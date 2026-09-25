import numpy as np, torch
from layer_lenses.relu_analysis import load_seed_result_pickle, regenerate_seed_result_data, replay_epoch_from_snapshot
P = {"FB-SGD":"seed_result_replayable_5_40_40_40_1Mtrain_1Mbatch_sgd_momentum_9e-1.pkl",
     "FB-Adam":"seed_result_replayable_5_40_40_40_1Mtrain_1Mbatch_adam.pkl"}
R = {k: load_seed_result_pickle("results/cache/"+v) for k, v in P.items()}
U = torch.from_numpy(R["FB-SGD"]["tree"].w_list).double()
level = np.array([0] + [1]*2 + [2]*4 + [3]*8 + [4]*16)
data = regenerate_seed_result_data(R["FB-SGD"])
def per_level_energy(M):
    en = (M @ U.T).pow(2).sum(dim=0); return np.array([float(en[level==L].sum()) for L in range(5)])
print("One full-batch step replayed at late checkpoints. For layer 0: share of energy in level 4 for the gradient vs the actual update,")
print("and whether the update grows (+) or shrinks (-) the layer's existing level-4 components (sign of <W u_j, dW u_j> summed over L4).\n")
print(f"{'run':8s} {'epoch':>6s}  {'|grad|':>8s}  grad share per level L0..L4        update share per level L0..L4      L4 amplification (update/grad share)   L4 growth sign   |grad L4|/|grad L0| per node")
for k, sr in R.items():
    for e in [20000, 60000, 99800]:
        rp = replay_epoch_from_snapshot(sr, e, data=data, record_optimizer_state=False, verify_against_next_snapshot=False)
        s = rp["steps"][0]
        G = s["grad"]["hidden_layers.0.weight"].double(); D = s["update"]["hidden_layers.0.weight"].double()
        W = s["params_before"]["hidden_layers.0.weight"].double()
        g, d = per_level_energy(G), per_level_energy(D)
        gs, ds = g / g.sum(), d / d.sum()
        growth = float(((W @ U[15:].T) * (D @ U[15:].T)).sum())
        ratio_node = np.sqrt((g[4]/16) / (g[0]/1))
        print(f"{k:8s} {e:6d}  {float(G.norm()):8.2e}  {np.round(gs,3)}   {np.round(ds,3)}      {ds[4]/gs[4]:6.2f}                       {'+' if growth>0 else '-'}{abs(growth):.2e}        {ratio_node:.3f}")
