import numpy as np, torch
from layer_lenses.relu_analysis import load_seed_result_pickle
sr = load_seed_result_pickle("results/cache/seed_result_replayable_5_40_40_40_1Mtrain_10kbatch_sgd_momentum_9e-1.pkl")
ck, tree = sr["out"]["checkpoint_snapshots"], sr["tree"]; N = 37
Uodt = torch.from_numpy(tree.w_list).double()                # (31,100) orthonormal
u0 = Uodt[0]
def W(e): return ck[e]["hidden_layers.0.weight"].double()
w_init = W(0)[N]
print("epoch  cos(w,u0)  |dw|    radial  tang->-u0   other  | other: top ODT (id,cos)          cos(other, other_prev)  cos(other, -w_init_perp)")
prev_other = None
for e in range(55, 131, 5):
    w, w1 = W(e)[N], W(e+1)[N]; dw = w1 - w
    what = w / w.norm()
    t = -u0 - (-u0 @ what) * what; t = t / t.norm()
    rad = float(dw @ what); tan = float(dw @ t)
    other = dw - rad*what - tan*t
    oc = Uodt @ (other / other.norm())
    top = torch.topk(oc.abs(), 3)
    top_s = " ".join(f"({int(i)},{float(oc[i]):+.2f})" for i in top.indices)
    # component of the initial weight orthogonal to the current (w,u0) plane
    wi = w_init - (w_init@what)*what; wi = wi - (wi@t)*t
    co_init = float(other @ (-wi) / (other.norm()*wi.norm()))
    co_prev = float(other @ prev_other / (other.norm()*prev_other.norm())) if prev_other is not None else float('nan')
    print(f"{e:4d}   {float(what@u0):+.3f}   {float(dw.norm()):.4f}  {rad:+.4f}  {tan:+.4f}   {float(other.norm()):.4f}  | {top_s:34s}  {co_prev:+.2f}                {co_init:+.2f}")
    prev_other = other

# Is "other" shedding the component of w that lies outside span(u0)? Track |w_perp_to_u0| and the part of w outside span(u0, w_init)
print("\nepoch   |w|     |w - (w.u0)u0|   cos(w_perp_u0, w_init_perp_u0)   |w along ODT children u1,u2|")
wi_p = w_init - (w_init@u0)*u0
for e in range(40, 141, 10):
    w = W(e)[N]; wp = w - (w@u0)*u0
    print(f"{e:4d}   {float(w.norm()):.3f}    {float(wp.norm()):.3f}            {float(wp@wi_p/(wp.norm()*wi_p.norm())):+.3f}                        {float((Uodt[1:3] @ w).norm()):.3f}")
