import numpy as np, torch
from layer_lenses.relu_analysis import load_seed_result_pickle
sr = load_seed_result_pickle("results/cache/seed_result_replayable_5_40_40_40_1Mtrain_10kbatch_sgd_momentum_9e-1.pkl")
ck, tree = sr["out"]["checkpoint_snapshots"], sr["tree"]
U = torch.from_numpy(tree.w_list).double(); u0 = U[0]
W = lambda e: ck[e]["hidden_layers.0.weight"].double()
A = lambda e: ck[e]["output_layer.weight"][0].double()
N = 37
w0 = W(0)[N]
print("Escape of neuron 37 from the initial plateau: deviation d(e) = w(e) - w(0)")
print("epoch   |d|        log-slope/epoch   cos(d_hat(e), d_hat(e-5))   cos(d_hat(e), -u0)   top ODT of d_hat   |a(e)-a(0)|")
prev = None; prev_norm = None
for e in range(5, 66, 5):
    d = W(e)[N] - w0; dn = d.norm(); dh = d / dn
    slope = float(torch.log(dn / prev_norm) / 5) if prev_norm is not None else float("nan")
    cprev = float(dh @ prev) if prev is not None else float("nan")
    oc = U @ dh; top = torch.topk(oc.abs(), 2)
    tops = " ".join(f"({int(i)},{float(oc[i]):+.2f})" for i in top.indices)
    print(f"{e:4d}   {float(dn):.5f}   {slope:+.3f}            {cprev:+.4f}                  {float(dh @ -u0):+.3f}          {tops:22s}  {float((A(e)[N]-A(0)[N]).abs()):.5f}")
    prev, prev_norm = dh, dn

# Is the escape a single collective unstable mode across the whole first layer? Compare all-neuron deviation.
print("\nAll 40 first-layer neurons: |W(e)-W(0)|_F and cos of successive deviation directions")
prev = None; prev_norm = None
for e in range(5, 66, 5):
    D = (W(e) - W(0)).reshape(-1); dn = D.norm(); dh = D / dn
    slope = float(torch.log(dn / prev_norm) / 5) if prev_norm is not None else float("nan")
    cprev = float(dh @ prev) if prev is not None else float("nan")
    per_neuron = (W(e) - W(0)).norm(dim=1)
    top = torch.topk(per_neuron, 3)
    print(f"{e:4d}   {float(dn):.5f}   slope {slope:+.3f}   cos_prev {cprev:+.4f}   most-moved neurons {[int(i) for i in top.indices]} norms {[round(float(v),4) for v in top.values]}")
