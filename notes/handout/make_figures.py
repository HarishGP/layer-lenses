"""Figures and numbers for the handout (notes/handout/handout.tex), generated from the saved replay trajectories of runs 60, 70, 73, 97, 74
and 114. Writes notes/handout/figs/*.pdf and notes/handout/facts.json. Run from the repository root with the project venv."""
import json, pickle, numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from layer_lenses.init_sweep import SweepRunSpec, build_data, load_tree
from layer_lenses.designed_init import HIERARCHY_TARGETS, NODE_LEVEL, node_ancestors
from layer_lenses.path_analysis import leaf_cells
from layer_lenses.symmetric_flow import symmetric_orbits
from layer_lenses.designed_init import cue_row_identity

R = "results/designed_init"; OUT = "notes/handout/figs"; dev = "cuda" if torch.cuda.is_available() else "cpu"
LEVEL_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; CUE_COL = "#6250d6"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e6e5e1"
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f4f4f2", "#eb6834"]); SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f4f2", "#2a78d6"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 120, "savefig.bbox": "tight"})
tree = load_tree(f"{R}/hd2_run60_sgd_lr1_1000/tree.npz"); U = tree.w_list
xe, ye = build_data(SweepRunSpec(tree_seed=5178, data_seed=999_777, init_seed=0, batch_size=1, n_train=200_000), tree); ye = ye.astype(np.float64)
Xe = torch.tensor(xe, dtype=torch.float32, device=dev); Ye = torch.tensor(ye, dtype=torch.float32, device=dev)
RUNS = {60: "hd2_run60_sgd_lr1_1000", 70: "hd2_run70_symcue_partner", 73: "hd2_run73_sym_pgf_two_phase", 97: "hd2_run97_sym_pgf_run73_cont", 74: "hd2_run74_sym_pgf_run60", 114: "hd2_run114_a13_1.0_restart_run74"}
LEVEL_ROWS = {l: [p for p in range(30) if NODE_LEVEL[HIERARCHY_TARGETS[p][0]] == l] for l in range(4)}

def load(n):
    f = f"{R}/{RUNS[n]}"; z = np.load(f"{f}/replay/weights.npz"); meta = pickle.load(open(f"{f}/replay/meta.pkl", "rb")); d = meta["design"]
    W1, W2, Wo = (z[k].astype(np.float64) for k in ["w__hidden_layers.0.weight", "w__hidden_layers.1.weight", "w__output_layer.weight"]); ep = np.array(z["epochs"])
    L = np.array(meta["out"]["epoch_losses"], float); L = L[~np.isnan(L)]; cr = list(d["cue_rows"]); cid = cue_row_identity(W1[0], tree, cr); inv = {v: k for k, v in cid.items()}
    return {"n": n, "W1": W1, "W2": W2, "Wo": Wo, "ep": ep, "loss": L, "d": d, "cue_rows": cr, "cid": cid, "inv": inv, "unode": np.array(d["unit_node"]), "usign": np.array(d["unit_path_sign"]), "ublock": np.array(d["unit_block_sign"])}

def cosines(run):
    """|cos| of every row with every normal, per snapshot (T x 62 x 31)."""
    W = run["W1"]; return np.abs(W / np.linalg.norm(W, axis=2, keepdims=True) @ U.T)

def test_error(run, every=10):
    idx = np.arange(0, len(run["ep"]), every); out = []
    for i in idx:
        W1, W2, Wo = (torch.tensor(run[k][i], dtype=torch.float32, device=dev) for k in ("W1", "W2", "Wo"))
        with torch.no_grad(): f = (torch.relu(torch.relu(Xe @ W1.T) @ W2.T) @ Wo.T).squeeze(1); out.append(float(((f >= 0) != (Ye > 0)).float().mean()))
    return run["ep"][idx], np.array(out)

def captures(run, C=None):
    """First epoch at which every probe of a level is within |cos| 0.95 of its own node; same for the cues and their leaves."""
    C = cosines(run) if C is None else C; cap = {}
    for l in range(4):
        ok = np.all([(C[:, p, :].argmax(1) == HIERARCHY_TARGETS[p][0]) & (C[:, p, HIERARCHY_TARGETS[p][0]] > 0.95) for p in LEVEL_ROWS[l]], axis=0); cap[l] = int(run["ep"][np.argmax(ok)]) if ok.any() else None
    okc = np.all([(C[:, r, :].argmax(1) == run["inv"][r][0]) & (C[:, r, run["inv"][r][0]] > 0.95) for r in run["cue_rows"]], axis=0); cap["cues"] = int(run["ep"][np.argmax(okc)]) if okc.any() else None
    return cap

facts = {}
runs = {n: load(n) for n in RUNS}
# ---------------------------------------------------------------- figure 1: the tree
fig, ax = plt.subplots(figsize=(7.2, 3.1)); ax.axis("off")
pos = {}
for j in range(31):
    l = NODE_LEVEL[j] if j < 15 else 4; k = j - (2 ** l - 1); pos[j] = ((k + 0.5) / 2 ** l, -l)
for j in range(15):
    for c in (2 * j + 1, 2 * j + 2): ax.plot([pos[j][0], pos[c][0]], [pos[j][1], pos[c][1]], color=INK2, lw=0.8, zorder=1)
for j in range(31):
    l = NODE_LEVEL[j] if j < 15 else 4; col = LEVEL_COL[l] if l < 4 else CUE_COL
    ax.scatter(*pos[j], s=170 if l < 4 else 120, color=col, zorder=2, edgecolor="white", linewidth=0.8); ax.text(pos[j][0], pos[j][1], str(j), ha="center", va="center", fontsize=6.5 if l == 4 else 7.5, color="white", zorder=3)
for l, name in enumerate(["level 0 (root)", "level 1", "level 2", "level 3", "level 4: label nodes"]): ax.text(-0.02, -l, name, ha="right", va="center", fontsize=8, color=INK2)
ax.text(pos[1][0] - 0.02, -0.5, r"$u_0\!\cdot\! x<0$", ha="right", va="center", fontsize=8, color=INK2); ax.text(pos[2][0] + 0.02, -0.5, r"$u_0\!\cdot\! x\geq 0$", ha="left", va="center", fontsize=8, color=INK2)
ax.set_xlim(-0.25, 1.02); ax.set_ylim(-4.4, 0.4); fig.savefig(f"{OUT}/tree.pdf"); plt.close(fig)
# ---------------------------------------------------------------- figure 2: the init of run 60 (W1 cosines, W2, readout) and of run 70's W2
def w1_cos_panel(ax, W1, title):
    C = W1 / np.linalg.norm(W1, axis=1, keepdims=True) @ U.T; im = ax.imshow(C.T, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1), aspect="auto", interpolation="nearest")
    ax.axvline(29.5, color=INK, lw=0.8); ax.set_xticks([14.5, 45.5]); ax.set_xticklabels(["probes (30)", "cues (32)"]); ax.set_yticks([0, 15, 30]); ax.set_yticklabels([r"$u_0$", r"$u_{15}$", r"$u_{30}$"]); ax.set_ylabel("tree normal"); ax.set_title(title, fontsize=9); ax.grid(False); return im
def w2_panel(ax, W2, title, vmax=None):
    vmax = vmax or np.abs(W2).max(); im = ax.imshow(W2, cmap=DIV, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto", interpolation="nearest"); ax.axvline(29.5, color=INK, lw=0.8)
    for b in (3.5, 11.5, 27.5): ax.axhline(b, color=INK, lw=0.5)
    ax.set_xticks([14.5, 45.5]); ax.set_xticklabels(["probe rows", "cue rows"]); ax.set_yticks([1.5, 7.5, 19.5, 43.5]); ax.set_yticklabels(["L0", "L1", "L2", "L3"]); ax.set_ylabel("unit (by node level)"); ax.set_title(title, fontsize=9); ax.grid(False); return im
r60, r70 = runs[60], runs[70]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1, 1]}, constrained_layout=True)
im = w1_cos_panel(axes[0], r60["W1"][0], "run 60 init: cos(row, normal)"); plt.colorbar(im, ax=axes[0], fraction=0.05, pad=0.03)
im = w2_panel(axes[1], r60["W2"][0], "run 60 init: $W_2$"); plt.colorbar(im, ax=axes[1], fraction=0.05, pad=0.03)
im = w2_panel(axes[2], r70["W2"][0], "run 70 init: $W_2$", vmax=1.0); plt.colorbar(im, ax=axes[2], fraction=0.05, pad=0.03)
fig.savefig(f"{OUT}/init_run60_70.pdf"); plt.close(fig)
# ---------------------------------------------------------------- figure 3/4: trajectories of runs 60 and 70; figure 6: 74, 73+97, 114
def traj_panels(axes, run, title, ep_offset=0, phase_lines=(), C=None, ep_err=None, err=None, legend=True):
    C = cosines(run) if C is None else C; ep = run["ep"] + ep_offset
    for l in range(4):
        m = np.mean([C[:, p, HIERARCHY_TARGETS[p][0]] for p in LEVEL_ROWS[l]], axis=0); axes[0].plot(ep, m, color=LEVEL_COL[l], lw=1.6, label=f"probes of level {l}")
    mc = np.mean([C[:, r, run["inv"][r][0]] for r in run["cue_rows"]], axis=0); axes[0].plot(ep, mc, color=CUE_COL, lw=1.6, ls="--", label="cues (with own leaf)")
    axes[0].set_ylim(0, 1.02); axes[0].set_ylabel("mean |cos| with own normal"); axes[0].set_title(title, fontsize=9)
    nrm = np.linalg.norm(run["W1"], axis=2)
    for l in range(4): axes[1].plot(ep, nrm[:, LEVEL_ROWS[l]].mean(1), color=LEVEL_COL[l], lw=1.6)
    axes[1].plot(ep, nrm[:, run["cue_rows"]].mean(1), color=CUE_COL, lw=1.6, ls="--"); axes[1].set_ylabel("mean row norm")
    axes[2].plot(ep[1:], run["loss"][:len(ep) - 1], color=INK, lw=1.4, label="train loss (epoch mean)")
    if err is not None: axes[2].plot(ep_err + ep_offset, err, color="#e34948", lw=1.4, label="test error")
    axes[2].set_yscale("log"); axes[2].set_ylim(2e-4, 1); axes[2].set_ylabel("loss / error (log)"); axes[2].set_xlabel("epoch")
    for a in axes:
        for e in phase_lines: a.axvline(e, color=INK2, lw=0.7, ls=":")
    if legend: axes[0].legend(fontsize=7, loc="center right", frameon=True, framealpha=0.92, edgecolor="white"); axes[2].legend(fontsize=7, loc="lower left")
for n, phases in [(60, (500,)), (70, (500,))]:
    run = runs[n]; C = cosines(run); ep_err, err = test_error(run); facts[f"run{n}"] = {"captures": captures(run, C), "final_loss": float(run["loss"][-1]), "test_error_end": float(err[-1]), "epochs": int(run["ep"][-1])}
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True); traj_panels(axes, run, f"run {n}: plain SGD from the designed init (cues frozen for 500 epochs, then free)", phase_lines=phases, C=C, ep_err=ep_err, err=err); fig.savefig(f"{OUT}/traj_run{n}.pdf"); plt.close(fig)
    facts[f"run{n}"]["err_curve"] = {"epochs": ep_err.tolist(), "error": err.tolist()}
# run 74 and 114, run 73 + 97
run = runs[74]; C = cosines(run); ep_err, err = test_error(run); facts["run74"] = {"captures": captures(run, C), "final_loss": float(run["loss"][-1]), "test_error_end": float(err[-1])}
fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True); traj_panels(axes, run, "run 74: symmetrised flow from run 60's init (cues frozen 500, then free)", phase_lines=(500,), C=C, ep_err=ep_err, err=err); fig.savefig(f"{OUT}/traj_run74.pdf"); plt.close(fig)
run = runs[114]; C = cosines(run); ep_err, err = test_error(run); facts["run114"] = {"captures": captures(run, C), "final_loss": float(run["loss"][-1]), "test_error_end": float(err[-1])}
r73, r97 = runs[73], runs[97]; C73 = cosines(r73); C97 = cosines(r97); e73, err73 = test_error(r73); e97, err97 = test_error(r97)
facts["run73"] = {"captures": captures(r73, C73), "final_loss": float(r73["loss"][-1]), "test_error_end": float(err73[-1])}; facts["run97"] = {"captures": {k: (v + 1500 if v is not None else None) for k, v in captures(r97, C97).items()}, "final_loss": float(r97["loss"][-1]), "test_error_end": float(err97[-1])}
fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True)
traj_panels(axes, r73, "runs 73 + 97: symmetrised flow from run 70's init (cues frozen 1000, free 500; then 2000 more epochs with a fresh cosine)", phase_lines=(1000, 1500), C=C73, ep_err=e73, err=err73)
traj_panels(axes, r97, None, ep_offset=1500, C=C97, ep_err=e97, err=err97, legend=False); axes[0].set_title("runs 73 + 97: symmetrised flow from run 70's init (cues released at 1000, run 97 from 1500)", fontsize=9)
fig.savefig(f"{OUT}/traj_run73_97.pdf"); plt.close(fig)
# overlay SGD vs symmetrised: level cosines of 60 vs 74 and 70 vs 73
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True, constrained_layout=True)
for ax, (a, b, ttl) in zip(axes, [(60, 74, "run 60 (SGD, solid) / run 74 (symmetrised, dashed)"), (70, 73, "run 70 (SGD, solid) / run 73 (symmetrised, dashed)")]):
    for n, ls in ((a, "-"), (b, "--")):
        run = runs[n]; C = cosines(run)
        for l in range(4): ax.plot(run["ep"], np.mean([C[:, p, HIERARCHY_TARGETS[p][0]] for p in LEVEL_ROWS[l]], axis=0), color=LEVEL_COL[l], lw=1.4, ls=ls, label=f"level {l}" if ls == "-" else None)
    ax.set_xlim(0, 1000); ax.set_ylim(0, 1.02); ax.set_title(ttl, fontsize=8.5); ax.set_xlabel("epoch")
axes[0].set_ylabel("mean |cos| of the probes with their node"); axes[0].legend(fontsize=7, loc="lower right"); fig.savefig(f"{OUT}/overlay_sgd_vs_sym.pdf"); plt.close(fig)
# ---------------------------------------------------------------- figure 5: the learned representation (run 60 at the end)
W1e, W2e, Woe = r60["W1"][-1], r60["W2"][-1], r60["Wo"][-1]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.15, 0.5]}, constrained_layout=True)
im = w1_cos_panel(axes[0], W1e, "run 60, epoch 1000: cos(row, normal)"); plt.colorbar(im, ax=axes[0], fraction=0.05, pad=0.03)
im = w2_panel(axes[1], W2e, "run 60, epoch 1000: $W_2$"); plt.colorbar(im, ax=axes[1], fraction=0.05, pad=0.03)
axes[2].barh(np.arange(60), Woe.ravel(), color=[LEVEL_COL[NODE_LEVEL[j]] for j in r60["unode"]], height=0.8); axes[2].invert_yaxis(); axes[2].set_title("readout $W_{out}$", fontsize=9); axes[2].set_yticks([1.5, 7.5, 19.5, 43.5]); axes[2].set_yticklabels(["L0", "L1", "L2", "L3"]); axes[2].grid(False)
fig.savefig(f"{OUT}/learned_run60.pdf"); plt.close(fig)
# unit statistics at the end: active fraction, top leaf-cell share, correlation of contribution with the label, per level; and ablations
cells = leaf_cells(xe, U); cell_leaf = cells // 2
H1 = np.maximum(xe @ W1e.T, 0); H2 = np.maximum(H1 @ W2e.T, 0); contrib = H2 * Woe.ravel()
stats = []
for u in range(60):
    act = H2[:, u] > 0; frac = act.mean(); top = 0.0; corr = 0.0
    if act.sum() > 20:
        counts = np.bincount(cell_leaf[act], minlength=16); top = counts.max() / counts.sum(); corr = np.corrcoef(contrib[act, u], ye[act])[0, 1]
    stats.append({"unit": u, "level": NODE_LEVEL[r60["unode"][u]], "active_frac": frac, "top_leaf_share": top, "corr_label": corr, "readout": Woe.ravel()[u]})
stats = pd.DataFrame(stats); facts["unit_stats_run60"] = stats.groupby("level").agg(active_frac=("active_frac", "mean"), top_leaf_share=("top_leaf_share", "mean"), corr_label=("corr_label", "mean"), readout_abs=("readout", lambda s: float(np.abs(s).mean()))).round(3).reset_index().to_dict("records")
def err_keep(keep):
    f = np.maximum(np.maximum(xe @ W1e.T, 0) @ W2e[keep].T, 0) @ Woe.ravel()[keep]; return float(((f >= 0) != (ye > 0)).mean())
lev = np.array([NODE_LEVEL[j] for j in r60["unode"]]); facts["ablation_run60"] = {"full": err_keep(np.arange(60)), "L3_only": err_keep(np.where(lev == 3)[0]), "L0-2_only": err_keep(np.where(lev <= 2)[0]), "L0_removed": err_keep(np.where(lev >= 1)[0])}
# the W1 rows at the end: norms and signs per level (probes) and cues
nrm = np.linalg.norm(W1e, axis=1); facts["row_norms_run60_end"] = {f"L{l}": float(nrm[LEVEL_ROWS[l]].mean()) for l in range(4)}; facts["row_norms_run60_end"]["cues"] = float(nrm[r60["cue_rows"]].mean())
facts["w2_run60_end"] = {"veto_on_own_probe_L3": float(np.mean([W2e[u, 2 * r60["unode"][u] + (0 if r60["ublock"][u] * r60["usign"][u] > 0 else 1)] for u in range(60) if lev[u] == 3])), "mean_abs_cue_weight_L3": float(np.abs(W2e[lev == 3][:, 30:]).mean()), "mean_abs_cue_weight_L0": float(np.abs(W2e[lev == 0][:, 30:]).mean())}
# a single level-3 unit: its weights on the probes of its ancestors (the veto chain) and on the cues
u3 = int(np.where(lev == 3)[0][0]); j3 = int(r60["unode"][u3]); anc = [a for a in node_ancestors(j3)]
facts["example_unit"] = {"unit": u3, "node": j3, "ancestors": anc, "weights_on_probes_of_ancestors": {int(a): [round(float(W2e[u3, 2 * a]), 2), round(float(W2e[u3, 2 * a + 1]), 2)] for a in anc}, "max_abs_weight_on_other_probes": float(np.abs(np.delete(W2e[u3, :30], [2 * a + k for a in anc for k in (0, 1)])).max()), "largest_cue_weights": sorted([(round(float(W2e[u3, r]), 2), str(r60["inv"][r])) for r in r60["cue_rows"]], key=lambda t: -abs(t[0]))[:6]}
# ---------------------------------------------------------------- figure: one level-3 unit's incoming weights at the end of run 60
fig, ax = plt.subplots(figsize=(7.2, 2.6)); w = W2e[u3]; cols = [LEVEL_COL[NODE_LEVEL[HIERARCHY_TARGETS[r][0]]] if r < 30 else CUE_COL for r in range(62)]
ax.bar(np.arange(62), w, color=cols, width=0.8); ax.axvline(29.5, color=INK, lw=0.8); ax.set_xticks([14.5, 45.5]); ax.set_xticklabels(["probe rows (coloured by the level of their node)", "cue rows"]); ax.set_ylabel("$W_2$ weight"); ax.set_title(f"incoming $W_2$ weights of unit {u3} (node {j3}, level 3) at epoch 1000 of run 60", fontsize=9)
for r in [2 * a for a in anc] + [2 * j3, r60["cid"][(15, -1)]]: ax.annotate(("$u_{%d}$" % HIERARCHY_TARGETS[r][0]) if r < 30 else "cue $(15,-)$", (r, w[r]), textcoords="offset points", xytext=(0, -10), ha="center", fontsize=8, color=INK)
fig.savefig(f"{OUT}/unit_example.pdf"); plt.close(fig)
# ---------------------------------------------------------------- figure 7: the orbit structure of the symmetric subspace
units = list(zip(r60["unode"].tolist(), r60["ublock"].tolist(), r60["usign"].tolist())); orb = symmetric_orbits(r60["cid"], units)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1]}, constrained_layout=True)
o1, f1 = orb["W1"][0].astype(float), orb["forced_zero"]; o1m = np.ma.masked_where(f1[orb["W1"][0]], o1); rng = np.random.default_rng(3); perm = rng.permutation(orb["n_orbits"])
axes[0].imshow(perm[orb["W1"][0]].T, cmap="tab20", aspect="auto", interpolation="nearest", alpha=0.9); axes[0].imshow(np.ma.masked_where(~f1[orb["W1"][0]], np.ones_like(o1)).T, cmap=LinearSegmentedColormap.from_list("z", ["#0b0b0b", "#0b0b0b"]), aspect="auto", interpolation="nearest")
axes[0].axvline(29.5, color="white", lw=1.2); axes[0].axhline(30.5, color="white", lw=1.2); axes[0].set_xticks([14.5, 45.5]); axes[0].set_xticklabels(["probe rows", "cue rows"]); axes[0].set_yticks([15, 45]); axes[0].set_yticklabels(["in-span\n$u_0..u_{30}$", "off-span\n$o_0..o_{29}$"]); axes[0].set_title("$W_1$ coordinates: orbit (colour), forced zero (black)", fontsize=8.5); axes[0].grid(False)
o2 = orb["W2"][0]; axes[1].imshow(perm[o2], cmap="tab20", aspect="auto", interpolation="nearest", alpha=0.9); axes[1].imshow(np.ma.masked_where(~f1[o2], np.ones_like(o2, dtype=float)), cmap=LinearSegmentedColormap.from_list("z", ["#0b0b0b", "#0b0b0b"]), aspect="auto", interpolation="nearest")
axes[1].axvline(29.5, color="white", lw=1.2); axes[1].set_xticks([14.5, 45.5]); axes[1].set_xticklabels(["probe rows", "cue rows"]); axes[1].set_yticks([1.5, 7.5, 19.5, 43.5]); axes[1].set_yticklabels(["L0", "L1", "L2", "L3"]); axes[1].set_title("$W_2$ coordinates: orbit (colour), forced zero (black)", fontsize=8.5); axes[1].grid(False)
fig.savefig(f"{OUT}/orbits.pdf"); plt.close(fig)
facts["orbits"] = {"n_orbits": int(orb["n_orbits"]), "n_free": int(orb["n_free"]), "n_coordinates": int(62 * 61 + 60 * 62 + 60)}
# ---------------------------------------------------------------- numbers: linear model, capture tables
z = xe @ U[15:].sum(0); t = ye * z
def bisect(f, lo, hi):
    for _ in range(60):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)
c = bisect(lambda c: np.mean(-t / (1 + np.exp(c * t))), 0.01, 10); facts["linear"] = {"c": float(c), "loss": float(np.mean(np.logaddexp(0, -c * t))), "error": float(np.mean(np.sign(z) != ye))}
json.dump(facts, open("notes/handout/facts.json", "w"), indent=1, default=float); print(json.dumps({k: v for k, v in facts.items() if "curve" not in str(k)}, indent=1, default=float)[:6000])
