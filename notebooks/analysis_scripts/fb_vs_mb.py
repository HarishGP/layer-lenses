import numpy as np, torch, math
from layer_lenses.relu_analysis import load_seed_result_pickle, regenerate_seed_result_data
from layer_lenses.relu_training import checkpoint_model_from_state, evaluate_relu_mlp
FB = load_seed_result_pickle("results/cache/seed_result_replayable_5_40_40_40_1Mtrain_1Mbatch_sgd_momentum_9e-1.pkl")
MB = load_seed_result_pickle("results/cache/seed_result_replayable_5_40_40_40_1Mtrain_10kbatch_sgd_momentum_9e-1.pkl")
assert FB["data_spec"] == MB["data_spec"], "different data"
tree = FB["tree"]; U = torch.from_numpy(tree.w_list).double(); u0 = U[0]
cfgF, cfgM = FB["train_config"], MB["train_config"]
print("full-batch :", f"epochs={cfgF.epochs} batch={cfgF.batch_size} lr={cfgF.lr} mom={cfgF.momentum} wd={cfgF.weight_decay} steps={cfgF.epochs*math.ceil(1e6/cfgF.batch_size):,}")
print("mini-batch :", f"epochs={cfgM.epochs} batch={cfgM.batch_size} lr={cfgM.lr} mom={cfgM.momentum} wd={cfgM.weight_decay} steps={cfgM.epochs*math.ceil(1e6/cfgM.batch_size):,}")

def flow_time(sr):
    cfg = sr["train_config"]; spe = math.ceil(1e6 / cfg.batch_size)
    lr = np.array(sr["out"]["lr_history"]); T = np.concatenate([[0.0], np.cumsum(lr) * spe / (1 - cfg.momentum)])
    return T   # T[e] = flow time at the START of epoch e (when checkpoint e is taken)
TF, TM = flow_time(FB), flow_time(MB)
print(f"total flow time: full-batch {TF[-1]:,.0f}   mini-batch {TM[-1]:,.0f}   (ratio {TM[-1]/TF[-1]:.1f})")

ckF, ckM = FB["out"]["checkpoint_snapshots"], MB["out"]["checkpoint_snapshots"]
W0F, W0M = ckF[0]["hidden_layers.0.weight"].double(), ckM[0]["hidden_layers.0.weight"].double()
print("same init:", torch.equal(W0F, W0M))
def layer_stats(ck, e):
    W = ck[e]["hidden_layers.0.weight"].double()
    C = (W / W.norm(dim=1, keepdim=True)) @ U.T          # (40,31)
    best, arg = C.abs().max(dim=1)
    captured = sorted(set(int(a) for a, b in zip(arg, best) if b > 0.8))
    return W, float(W[37] @ u0 / W[37].norm()), int((best > 0.8).sum()), captured
mb_epochs = np.array(sorted(ckM.keys()))
print("\n=== matched flow time: full-batch epoch e_F  vs  nearest mini-batch snapshot ===")
print("  T        e_F   lossF   |dW|F  cos37F  nAlnF | e_M   lossM   |dW|M  cos37M  nAlnM | rowwise cos(W_F,W_M) median/min   captured nodes F | M")
lossF, lossM = FB["out"]["epoch_losses"], MB["out"]["epoch_losses"]
for eF in [0, 50, 100, 199, 400, 1000, 2000, 4000, 10000, 20000, 40000, 60000, 80000, 100000]:
    if eF not in ckF: continue
    t = TF[eF]; eM = int(mb_epochs[np.argmin(np.abs(TM[mb_epochs] - t))])
    WF, c37F, nF, capF = layer_stats(ckF, eF); WM, c37M, nM, capM = layer_stats(ckM, eM)
    rc = (WF * WM).sum(dim=1) / (WF.norm(dim=1) * WM.norm(dim=1))
    lF = lossF[min(eF, len(lossF)-1)]; lM = lossM[min(eM, len(lossM)-1)]
    print(f"{t:8.0f} {eF:6d}  {lF:.4f}  {float((WF-W0F).norm()):6.3f}  {c37F:+.3f}   {nF:2d}   | {eM:4d}  {lM:.4f}  {float((WM-W0M).norm()):6.3f}  {c37M:+.3f}   {nM:2d}   |   {float(rc.median()):+.3f} / {float(rc.min()):+.3f}          {capF} | {capM}")

print("\n=== end-of-run metrics (train loss / test error on the 1M held-out split) ===")
data = regenerate_seed_result_data(FB)
def metrics(sr, e):
    m = checkpoint_model_from_state(sr["out"]["model"], sr["out"]["checkpoint_snapshots"][e])
    tr = evaluate_relu_mlp(model=m, x_eval=data["x_train"], y_eval=data["y_train"], device="cuda")
    te = evaluate_relu_mlp(model=m, x_eval=data["x_eval"],  y_eval=data["y_eval"],  device="cuda")
    return tr["log_loss"], te["zero_one_loss"]
eM_match = int(mb_epochs[np.argmin(np.abs(TM[mb_epochs] - TF[-1]))])
for label, sr, e in [("full-batch final", FB, cfgF.epochs), (f"mini-batch @ same flow time (epoch {eM_match})", MB, eM_match), ("mini-batch final", MB, cfgM.epochs)]:
    tl, te = metrics(sr, e)
    print(f"  {label:45s} train log loss {tl:.4f}   test error {te:.4f}")
print("\nmini-batch: epochs needed to reach full-batch's final train loss:",
      next((e for e, l in enumerate(lossM) if l <= lossF[-1]), None), f"(full-batch final epoch loss {lossF[-1]:.4f})")
