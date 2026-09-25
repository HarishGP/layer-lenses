import numpy as np, torch
from layer_lenses.relu_analysis import (load_seed_result_pickle, regenerate_seed_result_data,
    replay_epoch_from_snapshot, epoch_replay_parameter_series)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=160)
sr = load_seed_result_pickle("results/cache/seed_result_replayable_5_40_40_40_1Mtrain_10kbatch_sgd_momentum_9e-1.pkl")
out, ck, tree = sr["out"], sr["out"]["checkpoint_snapshots"], sr["tree"]
N, E = 37, 140
u = torch.from_numpy(tree.w_list[0]).double(); u = u / u.norm()

print("=== A. neuron 37 trajectory vs u_0 (signed cos, norm, output weight) ===")
for e in list(range(0, 60, 5)) + list(range(60, 201, 10)):
    w = ck[e]["hidden_layers.0.weight"][N].double(); a = float(ck[e]["output_layer.weight"][0, N])
    print(f"epoch {e:4d}  cos(w,u0) {float(w@u/w.norm()):+.3f}  |w| {float(w.norm()):.3f}  a {a:+.3f}  a*|w| {a*float(w.norm()):+.3f}")

print("\n=== B. replay epoch", E, "===")
data = regenerate_seed_result_data(sr)
rp = replay_epoch_from_snapshot(sr, E, data=data, record_full_data_gradient=True, record_optimizer_state=False)
G  = epoch_replay_parameter_series(rp, "full_grad", "hidden_layers.0.weight")[:, N].double()   # (100,100) pop grads
Gb = epoch_replay_parameter_series(rp, "grad", "hidden_layers.0.weight")[:, N].double()        # minibatch grads
W  = epoch_replay_parameter_series(rp, "params_before", "hidden_layers.0.weight")[:, N].double()
U  = epoch_replay_parameter_series(rp, "update", "hidden_layers.0.weight")[:, N].double()
T = G.shape[0]
w0 = W[0]; what = w0 / w0.norm()
print("cos(w_before, u0) =", float(what@u), " |w| before/after:", float(w0.norm()), float((W[-1]+U[-1]).norm()))

def decompose(X, axis):
    along = X @ axis                       # (T,)
    perp  = X - along[:, None] * axis[None]
    return along, perp

for name, X in [("POP grad", G), ("MINIBATCH grad", Gb)]:
    al, pe = decompose(X, u)
    print(f"\n--- {name}: split along u0 vs perpendicular ---")
    print(f"per-step |along|  mean {al.abs().mean():.4g}   per-step |perp| mean {pe.norm(dim=1).mean():.4g}   (ratio perp/along {pe.norm(dim=1).mean()/al.abs().mean():.2f})")
    print(f"along: sum {al.sum():+.4g}   sum|.| {al.abs().sum():.4g}   coherence |sum|/sum|.| = {abs(al.sum())/al.abs().sum():.3f}   frac same sign as sum = {(torch.sign(al)==torch.sign(al.sum())).float().mean():.2f}")
    ps = pe.sum(dim=0)
    print(f"perp : |sum| {ps.norm():.4g}   sum|.| {pe.norm(dim=1).sum():.4g}   random-walk expectation sqrt(T)*rms = {np.sqrt(T)*float(pe.norm(dim=1).pow(2).mean().sqrt()):.4g}   coherence {ps.norm()/pe.norm(dim=1).sum():.3f}")
    tot = X.sum(dim=0)
    print(f"cos(sum, u0) = {float(tot@u/tot.norm()):+.3f}    cos(sum, w_hat) = {float(tot@what/tot.norm()):+.3f}")
    # lag autocorrelation of the perpendicular part
    pn = pe / pe.norm(dim=1, keepdim=True)
    ac = [float((pn[:-k] * pn[k:]).sum(dim=1).mean()) for k in range(1, 13)]
    print("perp lag-k cosine autocorr k=1..12:", " ".join(f"{a:+.2f}" for a in ac))
    # how 1-D is the perpendicular oscillation
    Uc, S, Vh = torch.linalg.svd(pe - pe.mean(dim=0), full_matrices=False)
    ev = S**2 / (S**2).sum()
    print(f"perp PCA: top-1 {ev[0]:.2f}, top-3 {ev[:3].sum():.2f}, top-10 {ev[:10].sum():.2f} of variance")

# --- C. is the perpendicular pop gradient a restoring force on this neuron's own position?
al_g, pe_g = decompose(G, u)
_, pe_w = decompose(W - W.mean(dim=0), u)       # perpendicular displacement of the iterate from its epoch-mean
cs = (pe_g * pe_w).sum(dim=1) / (pe_g.norm(dim=1) * pe_w.norm(dim=1))
print("\n=== C. restoring-force test: cos(perp pop-grad_t, perp displacement_t of w_37) ===")
print("per-step:", " ".join(f"{float(c):+.2f}" for c in cs[:30]), "...")
print(f"mean {cs.mean():+.3f}   (near +1 => gradient points back toward the epoch-mean position: local restoring force on THIS neuron)")
# effective curvature along the top perpendicular PC from finite differences of pop grad vs position
v = Vh[0]
dg = (G[1:] - G[:-1]) @ v; dw = (W[1:] - W[:-1]) @ v
lam = float((dg*dw).sum() / (dw*dw).sum())
lr, beta = rp["lr"], rp["momentum"]
print(f"\nfinite-diff curvature along top perp PC: lambda ~ {lam:.3g};  lr*lambda = {lr*lam:.3g}")
print(f"heavy-ball (beta={beta}) oscillates when lr*lambda > (1-sqrt(beta))^2 = {(1-beta**0.5)**2:.4f}; unstable when > 2(1+beta) = {2*(1+beta):.2f}")
q = (W - W.mean(dim=0)) @ v
print("iterate coordinate along top perp PC, steps 0..29:", " ".join(f"{float(x):+.4f}" for x in q[:30]))
# along-u0 curvature for comparison (radial/scale direction)
dg_a = (G[1:] - G[:-1]) @ u; dw_a = (W[1:] - W[:-1]) @ u
print(f"finite-diff curvature along u0 (radial): lambda ~ {float((dg_a*dw_a).sum()/(dw_a*dw_a).sum()):.3g}")
