import numpy as np, torch, math
from layer_lenses.relu_analysis import load_seed_result_pickle
P = {"FB-SGD":"seed_result_replayable_5_40_40_40_1Mtrain_1Mbatch_sgd_momentum_9e-1.pkl",
     "FB-Adam":"seed_result_replayable_5_40_40_40_1Mtrain_1Mbatch_adam.pkl",
     "MB-SGD":"seed_result_replayable_5_40_40_40_1Mtrain_10kbatch_sgd_momentum_9e-1.pkl",
     "MB-Adam":"seed_result_replayable_5178_5_40_40_40_1Mtrain_10kbatch_adam_randomhyper_noshuffle.pkl"}
R = {k: load_seed_result_pickle("results/cache/"+v) for k, v in P.items()}
U = torch.from_numpy(R["FB-SGD"]["tree"].w_list).double()
level = np.array([0] + [1]*2 + [2]*4 + [3]*8 + [4]*16)
print("Energy of the first layer along each ODT level, sum_j in level ||W1 u_j||^2, over training.")
print("(At init the layer is isotropic, so energy per level is proportional to the number of nodes: 1:2:4:8:16.)\n")
for k, sr in R.items():
    ck = sr["out"]["checkpoint_snapshots"]; losses = sr["out"]["epoch_losses"]; eps = sorted(ck)
    E = max(eps); pick = sorted(set([0, 20, 50, 100, 150, 199] + [e for e in eps if e % (E//10) == 0]))
    print(f"=== {k} ===")
    print(f"{'epoch':>7s} {'loss':>7s}   energy per level L0..L4          L4 share   L4/L0 per-node ratio")
    for e in pick:
        W = ck[e]["hidden_layers.0.weight"].double(); en = (W @ U.T).pow(2).sum(dim=0)
        enL = np.array([float(en[level==L].sum()) for L in range(5)])
        print(f"{e:7d} {losses[min(e,len(losses)-1)]:7.4f}   {np.round(enL,3)}   {enL[4]/enL.sum():.3f}      {(enL[4]/16)/(enL[0]/1):.4f}")
