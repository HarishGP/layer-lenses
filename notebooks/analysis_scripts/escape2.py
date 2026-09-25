import numpy as np, torch
from layer_lenses.relu_analysis import load_seed_result_pickle, regenerate_seed_result_data, log_loss_gradients
from layer_lenses.relu_training import checkpoint_model_from_state
sr = load_seed_result_pickle("results/cache/seed_result_replayable_5_40_40_40_1Mtrain_10kbatch_sgd_momentum_9e-1.pkl")
ck, tree, out = sr["out"]["checkpoint_snapshots"], sr["tree"], sr["out"]
U = torch.from_numpy(tree.w_list).double(); u0 = U[0]
data = regenerate_seed_result_data(sr)
W = lambda e: ck[e]["hidden_layers.0.weight"].double()

def popgrad(e, n=None):
    m = checkpoint_model_from_state(out["model"], ck[e])
    x, y = data["x_train"], data["y_train"]
    if n is not None: x, y = x[:n], y[:n]
    return log_loss_gradients(model=m, x=x, y=y, device="cuda")["hidden_layers.0.weight"].double()

g0 = popgrad(0)                   # population gradient at init (1M points)
gb = popgrad(0, n=10_000)         # first mini-batch gradient at init
print("|pop grad| per neuron at init: min/median/max", [round(float(v),6) for v in (g0.norm(dim=1).min(), g0.norm(dim=1).median(), g0.norm(dim=1).max())])
print("\nDoes the early escape direction equal the population gradient direction at init?")
print("epoch   neuron 37: cos(d_hat, -g_pop(0))   cos(d_hat, -g_batch0(0))   |  all 40 neurons: median cos(d_hat_k, -g_pop_k(0))   min")
for e in [5, 10, 20, 30, 40, 50, 60]:
    D = W(e) - W(0)
    c37 = float(D[37] @ (-g0[37]) / (D[37].norm() * g0[37].norm()))
    b37 = float(D[37] @ (-gb[37]) / (D[37].norm() * gb[37].norm()))
    cs = (D * (-g0)).sum(dim=1) / (D.norm(dim=1) * g0.norm(dim=1))
    print(f"{e:4d}        {c37:+.4f}                    {b37:+.4f}              |        {float(cs.median()):+.4f}                        {float(cs.min()):+.4f}")
print("\ncos(-g_pop(0), -g_batch0(0)) for neuron 37:", float(g0[37] @ gb[37] / (g0[37].norm()*gb[37].norm())))

# where does the population gradient at init point, in the ODT basis?  (neuron 37 and the layer average)
gh = -g0[37] / g0[37].norm()
oc = U @ gh; top = torch.topk(oc.abs(), 5)
print("\n-g_pop(0) for neuron 37, top ODT components:", " ".join(f"({int(i)},{float(oc[i]):+.2f})" for i in top.indices), f" | total in ODT span: {float(oc.norm()):.2f}")
# level structure: node ids 0 | 1-2 | 3-6 | 7-14 | 15-30
levels = [range(0,1), range(1,3), range(3,7), range(7,15), range(15,31)]
print("energy of -g_pop(0)[37] by ODT level:", [round(float((oc[list(r)]**2).sum()),3) for r in levels])
G = -g0 / g0.norm(dim=1, keepdim=True)
E = torch.stack([((U[list(r)] @ G.T)**2).sum(dim=0) for r in levels], dim=1)   # (40,5)
print("mean over 40 neurons of energy by level:", [round(float(v),3) for v in E.mean(dim=0)], " (a random direction in R^100 would put ~0.01 per node: ", [round(len(r)/100,2) for r in levels], ")")

# and later: does the population gradient turn toward u0 as the neuron grows?
print("\nneuron 37: cos(-g_pop(e), -u0) and top ODT component of -g_pop(e), for e in the escape/arrival window")
for e in [0, 40, 60, 70, 80, 100]:
    g = -popgrad(e)[37]; gh = g / g.norm(); oc = U @ gh; top = torch.topk(oc.abs(), 2)
    print(f"  epoch {e:3d}: cos(-g, -u0) {float(gh @ -u0):+.3f}   top {' '.join(f'({int(i)},{float(oc[i]):+.2f})' for i in top.indices)}   |g| {float(g.norm()):.5f}")
