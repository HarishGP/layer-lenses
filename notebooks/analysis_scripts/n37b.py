import numpy as np, torch
from layer_lenses.relu_analysis import (load_seed_result_pickle, regenerate_seed_result_data,
    replay_epoch_from_snapshot, epoch_replay_parameter_series)
sr = load_seed_result_pickle("results/cache/seed_result_replayable_5_40_40_40_1Mtrain_10kbatch_sgd_momentum_9e-1.pkl")
tree = sr["tree"]; N = 37
u = torch.from_numpy(tree.w_list[0]).double(); u = u / u.norm()
data = regenerate_seed_result_data(sr)

def proj_out(X, axis):
    return X - (X @ axis)[:, None] * axis[None]

for E in [75, 140]:
    rp = replay_epoch_from_snapshot(sr, E, data=data, record_full_data_gradient=True, record_optimizer_state=False)
    G  = epoch_replay_parameter_series(rp, "full_grad", "hidden_layers.0.weight")[:, N].double()
    Gb = epoch_replay_parameter_series(rp, "grad", "hidden_layers.0.weight")[:, N].double()
    W  = epoch_replay_parameter_series(rp, "params_before", "hidden_layers.0.weight")[:, N].double()
    U  = epoch_replay_parameter_series(rp, "update", "hidden_layers.0.weight")[:, N].double()
    T = G.shape[0]; w0 = W[0]; what = w0 / w0.norm()
    # tangential direction toward -u0 (the rotation the neuron is performing), orthogonal to w_hat
    t = -u - (-u @ what) * what; t = t / t.norm()
    print(f"\n================ epoch {E}: cos(w,u0)={float(what@u):+.3f}, |w|={float(w0.norm()):.3f} ================")
    tot = G.sum(dim=0); totU = U.sum(dim=0)
    print(f"sum pop-grad:  |sum| {tot.norm():.4g}   radial part (along w_hat) {float(tot@what):+.4g}   tangential-toward-(-u0) part {float(tot@t):+.4g}   other {float(proj_out(proj_out(tot[None],what),t).norm()):.4g}")
    print(f"sum update  :  |sum| {totU.norm():.4g}   radial {float(totU@what):+.4g}   tangential-toward-(-u0) {float(totU@t):+.4g}   other {float(proj_out(proj_out(totU[None],what),t).norm()):.4g}")
    # per-step: radial, tangential, rest
    rad = G @ what; tan = G @ t; rest = proj_out(proj_out(G, what), t)
    print(f"per-step pop grad rms: radial {rad.pow(2).mean().sqrt():.4g}  tangential {tan.pow(2).mean().sqrt():.4g}  rest {rest.norm(dim=1).pow(2).mean().sqrt():.4g}")
    for name, s in [("radial", rad), ("tangential", tan)]:
        print(f"  {name:10s}: mean {s.mean():+.4g}  std {s.std():.4g}  coherence |sum|/sum|.| {abs(s.sum())/s.abs().sum():.3f}  frac sign==sign(sum) {(torch.sign(s)==torch.sign(s.sum())).float().mean():.2f}")
    rs = rest.sum(dim=0)
    print(f"  rest      : |sum| {rs.norm():.4g}  sum|.| {rest.norm(dim=1).sum():.4g}  coherence {rs.norm()/rest.norm(dim=1).sum():.3f}   sqrt(T)*rms {np.sqrt(T)*float(rest.norm(dim=1).pow(2).mean().sqrt()):.4g}")
    # restoring-force test on the 'rest' subspace and oscillation period of the iterate there
    dW = proj_out(proj_out(W - W.mean(dim=0), what), t)
    cs = (rest * dW).sum(dim=1) / (rest.norm(dim=1) * dW.norm(dim=1))
    print(f"restoring-force cos(rest pop-grad, rest displacement): mean {cs.mean():+.3f}  min {cs.min():+.3f}")
    Uc, S, Vh = torch.linalg.svd(dW, full_matrices=False); v = Vh[0]
    q = dW @ v; qn = q - q.mean()
    ac = np.array([float((qn[:-k]*qn[k:]).mean()/(qn*qn).mean()) for k in range(1, 60)])
    zero_cross = next((k+1 for k in range(len(ac)-1) if ac[k] > 0 and ac[k+1] <= 0), None)
    first_min = int(np.argmin(ac)) + 1
    print(f"iterate wobble along its top PC: first zero-crossing of autocorr at lag {zero_cross}, first minimum at lag {first_min} => period ~ {2*first_min} steps; amplitude {q.abs().max():.4g} vs |w| {float(w0.norm()):.3f}")
    # minibatch noise is white?
    nb = proj_out(Gb - G, what)
    nbn = nb / nb.norm(dim=1, keepdim=True)
    print(f"minibatch-minus-pop grad: rms {nb.norm(dim=1).pow(2).mean().sqrt():.4g}, lag-1 cos autocorr {float((nbn[:-1]*nbn[1:]).sum(dim=1).mean()):+.3f}, lag-2 {float((nbn[:-2]*nbn[2:]).sum(dim=1).mean()):+.3f}")
