import numpy as np, torch, math
from layer_lenses.relu_analysis import load_seed_result_pickle
FB = load_seed_result_pickle("results/cache/seed_result_replayable_5_40_40_40_1Mtrain_1Mbatch_sgd_momentum_9e-1.pkl")
MB = load_seed_result_pickle("results/cache/seed_result_replayable_5_40_40_40_1Mtrain_10kbatch_sgd_momentum_9e-1.pkl")
U = torch.from_numpy(FB["tree"].w_list).double()
def flow_time(sr):
    cfg = sr["train_config"]; spe = math.ceil(1e6 / cfg.batch_size)
    return np.concatenate([[0.0], np.cumsum(sr["out"]["lr_history"]) * spe / (1 - cfg.momentum)])
TF, TM = flow_time(FB), flow_time(MB)
ckF, ckM = FB["out"]["checkpoint_snapshots"], MB["out"]["checkpoint_snapshots"]
fb_epochs = np.array(sorted(ckF.keys())); mb_epochs = np.array(sorted(ckM.keys()))
W = lambda ck, e: ck[e]["hidden_layers.0.weight"].double()
A = lambda ck, e: ck[e]["output_layer.weight"][0].double()
level = lambda j: 0 if j == 0 else 1 if j < 3 else 2 if j < 7 else 3 if j < 15 else 4

print("=== when does the first neuron diverge?  (mini-batch epoch vs flow-time-matched full-batch checkpoint) ===")
print(" e_M    T      e_F     rowwise cos: median   min   (#rows<0.99)   worst neuron   cos37")
for eM in [40, 60, 80, 100, 120, 140, 160, 180, 199, 300, 400, 500]:
    eF = int(fb_epochs[np.argmin(np.abs(TF[fb_epochs] - TM[eM]))])
    WF, WM = W(ckF, eF), W(ckM, eM)
    rc = (WF * WM).sum(dim=1) / (WF.norm(dim=1) * WM.norm(dim=1))
    print(f"{eM:4d}  {TM[eM]:6.0f}  {eF:6d}      {float(rc.median()):+.4f}   {float(rc.min()):+.4f}     {int((rc<0.99).sum()):2d}          {int(rc.argmin()):2d}        {float(rc[37]):+.4f}")

print("\n=== diverged neurons at the end of the full-batch run (T=10500) vs mini-batch epoch 500 ===")
eF, eM = 100000, 500
WF, WM = W(ckF, eF), W(ckM, eM); aF, aM = A(ckF, eF), A(ckM, eM)
rc = (WF * WM).sum(dim=1) / (WF.norm(dim=1) * WM.norm(dim=1))
def best(Wm, k):
    c = U @ (Wm[k] / Wm[k].norm()); j = int(c.abs().argmax()); return j, float(c[j])
print("neuron  cos(F,M)   full-batch: node cos |w| a         mini-batch: node cos |w| a")
for k in torch.argsort(rc)[:8]:
    k = int(k); jF, cF = best(WF, k); jM, cM = best(WM, k)
    print(f"  {k:2d}    {float(rc[k]):+.3f}     {jF:2d}(L{level(jF)}) {cF:+.2f} {float(WF[k].norm()):5.2f} {float(aF[k]):+.2f}       {jM:2d}(L{level(jM)}) {cM:+.2f} {float(WM[k].norm()):5.2f} {float(aM[k]):+.2f}")

print("\n=== node coverage by level at the end (|cos|>0.8 threshold and a looser 0.6), first layer only ===")
for label, Wm in [("full-batch 100k", WF), ("mini-batch 500", WM), ("mini-batch 10k (final)", W(ckM, 10000))]:
    C = (Wm / Wm.norm(dim=1, keepdim=True)) @ U.T; bst, arg = C.abs().max(dim=1)
    for thr in (0.8, 0.6):
        cov = {L: 0 for L in range(5)}
        for j in set(int(a) for a, b in zip(arg, bst) if b > thr): cov[level(j)] += 1
        print(f"  {label:24s} thr {thr}: nodes captured per level {[cov[L] for L in range(5)]} of [1, 2, 4, 8, 16];  neurons aligned {int((bst>thr).sum())}/40")
