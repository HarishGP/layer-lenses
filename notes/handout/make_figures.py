"""Figures and numbers for the handout (notes/handout/handout.tex), generated from the saved replay trajectories of the clean comparable
runs 123-130 (designs 60 / 70, plain SGD / symmetrised flow, two phases / one phase; constant step 0.1 for 1000 epochs then a cosine to
0.005 over 1500). Writes notes/handout/figs/*.pdf and notes/handout/facts.json. Run from the repository root with the project venv.
The x-axis of every trajectory is the flow time, the cumulative step size sum_t lr_t (per_epoch csv column flow_time; for older runs it
is rebuilt from the per-epoch lr history)."""
import json, os, pickle, numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from layer_lenses.init_sweep import SweepRunSpec, build_data, load_tree
from layer_lenses.designed_init import HIERARCHY_TARGETS, NODE_LEVEL, node_ancestors, cue_row_identity
from layer_lenses.path_analysis import leaf_cells
from layer_lenses.symmetric_flow import symmetric_orbits

R = "results/designed_init"; OUT = "notes/handout/figs"; dev = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = {"sgd_d60_2ph": "hd2_run131_clean4k_sgd_d60_2ph", "sgd_d70_2ph": "hd2_run132_clean4k_sgd_d70_2ph", "sym_d60_2ph": "hd2_run133_clean4k_sym_d60_2ph", "sym_d70_2ph": "hd2_run134_clean4k_sym_d70_2ph",
        "sgd_d60_1ph": "hd2_run135_clean4k_sgd_d60_1ph", "sgd_d70_1ph": "hd2_run136_clean4k_sgd_d70_1ph", "sym_d60_1ph": "hd2_run137_clean4k_sym_d60_1ph", "sym_d70_1ph": "hd2_run138_clean4k_sym_d70_1ph"}
if os.environ.get("HANDOUT_RUNS") == "2500":                                      # the 2500-epoch versions (runs 123-130)
    RUNS = {k: v.replace("clean4k", "clean").replace("hd2_run13", "hd2_run12").replace("run121", "run129").replace("run122", "run130") for k, v in RUNS.items()}
    RUNS = {"sgd_d60_2ph": "hd2_run123_clean_sgd_d60_2ph", "sgd_d70_2ph": "hd2_run124_clean_sgd_d70_2ph", "sym_d60_2ph": "hd2_run125_clean_sym_d60_2ph", "sym_d70_2ph": "hd2_run126_clean_sym_d70_2ph", "sgd_d60_1ph": "hd2_run127_clean_sgd_d60_1ph", "sgd_d70_1ph": "hd2_run128_clean_sgd_d70_1ph", "sym_d60_1ph": "hd2_run129_clean_sym_d60_1ph", "sym_d70_1ph": "hd2_run130_clean_sym_d70_1ph"}
if os.environ.get("HANDOUT_STANDINS"):                                           # older runs as stand-ins while the clean runs train
    RUNS = {"sgd_d60_2ph": "hd2_run60_sgd_lr1_1000", "sgd_d70_2ph": "hd2_run70_symcue_partner", "sym_d60_2ph": "hd2_run74_sym_pgf_run60", "sym_d70_2ph": "hd2_run73_sym_pgf_two_phase", "sym_d60_1ph": "hd2_run77_sym_pgf_run60_one_phase", "sym_d70_1ph": "hd2_run76_sym_pgf_one_phase"}
TITLE = {"sgd_d60_2ph": "design 60, plain SGD, two phases", "sgd_d70_2ph": "design 70, plain SGD, two phases", "sym_d60_2ph": "design 60, symmetrised flow, two phases", "sym_d70_2ph": "design 70, symmetrised flow, two phases",
         "sgd_d60_1ph": "design 60, plain SGD, one phase", "sgd_d70_1ph": "design 70, plain SGD, one phase", "sym_d60_1ph": "design 60, symmetrised flow, one phase", "sym_d70_1ph": "design 70, symmetrised flow, one phase"}
LEVEL_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; CUE_COL = "#6250d6"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e6e5e1"; ERR_COL = "#e34948"
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f4f4f2", "#eb6834"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 120, "savefig.bbox": "tight"})
tree = load_tree(f"{R}/hd2_run60_sgd_lr1_1000/tree.npz"); U = tree.w_list
xe, ye = build_data(SweepRunSpec(tree_seed=5178, data_seed=999_777, init_seed=0, batch_size=1, n_train=200_000), tree); ye = ye.astype(np.float64)
Xe = torch.tensor(xe, dtype=torch.float32, device=dev); Ye = torch.tensor(ye, dtype=torch.float32, device=dev)
LEVEL_ROWS = {l: [p for p in range(30) if NODE_LEVEL[HIERARCHY_TARGETS[p][0]] == l] for l in range(4)}
FT_UNIT = 1000.0                                                                  # flow time is plotted in units of 1000 (= 10 epochs at step 0.1)

def load(key):
    f = f"{R}/{RUNS[key]}"
    if not os.path.exists(f"{f}/summary.csv"): return None
    z = np.load(f"{f}/replay/weights.npz"); meta = pickle.load(open(f"{f}/replay/meta.pkl", "rb")); d = meta["design"]
    W1, W2, Wo = (z[k].astype(np.float64) for k in ["w__hidden_layers.0.weight", "w__hidden_layers.1.weight", "w__output_layer.weight"]); ep = np.array(z["epochs"])
    L = np.array(meta["out"]["epoch_losses"], float); L = L[~np.isnan(L)]; cr = list(d["cue_rows"]); cid = cue_row_identity(W1[0], tree, cr); inv = {v: k for k, v in cid.items()}
    lr = np.array(meta["out"]["lr_history"], float); lr = lr[~np.isnan(lr)]; steps = meta["data_spec"]["n_train"] // meta["train_config"].batch_size
    ft = np.array(meta["out"].get("flow_time", [np.nan]), float)
    if np.isnan(ft).all(): ft = np.cumsum(lr) * steps                            # older runs: rebuild from the per-epoch step size
    ft = ft[~np.isnan(ft)]; ft = np.concatenate([[0.0], ft[:len(ep) - 1]]) / FT_UNIT   # flow time at every snapshot epoch
    fc = int(d.get("freeze_cues", 500 if "phases" in d and d["phases"].get("cues") else 0)); fc_ft = float(ft[list(ep).index(fc)]) if fc in ep and fc > 0 else None
    return {"key": key, "W1": W1, "W2": W2, "Wo": Wo, "ep": ep, "ft": ft, "loss": L, "d": d, "cue_rows": cr, "cid": cid, "inv": inv, "unode": np.array(d["unit_node"]), "usign": np.array(d["unit_path_sign"]), "ublock": np.array(d["unit_block_sign"]), "freeze": fc, "freeze_ft": fc_ft, "symmetrised": bool(d.get("symmetrised_gradient", False))}

def cosines(run): W = run["W1"]; return np.abs(W / np.linalg.norm(W, axis=2, keepdims=True) @ U.T)
def test_error(run, every=10):
    idx = np.arange(0, len(run["ep"]), every); out = []
    for i in idx:
        W1, W2, Wo = (torch.tensor(run[k][i], dtype=torch.float32, device=dev) for k in ("W1", "W2", "Wo"))
        with torch.no_grad(): f = (torch.relu(torch.relu(Xe @ W1.T) @ W2.T) @ Wo.T).squeeze(1); out.append(float(((f >= 0) != (Ye > 0)).float().mean()))
    return idx, np.array(out)
def captures(run, C):
    """First epoch and flow time at which every probe of a level is within |cos| 0.95 of its own node; same for the cues and their leaves."""
    cap = {}
    for l in range(4):
        ok = np.all([(C[:, p, :].argmax(1) == HIERARCHY_TARGETS[p][0]) & (C[:, p, HIERARCHY_TARGETS[p][0]] > 0.95) for p in LEVEL_ROWS[l]], axis=0); i = int(np.argmax(ok)) if ok.any() else None
        cap[f"L{l}"] = None if i is None else {"epoch": int(run["ep"][i]), "flow_time": float(run["ft"][i])}
    okc = np.all([(C[:, r, :].argmax(1) == run["inv"][r][0]) & (C[:, r, run["inv"][r][0]] > 0.95) for r in run["cue_rows"]], axis=0); i = int(np.argmax(okc)) if okc.any() else None
    cap["cues"] = None if i is None else {"epoch": int(run["ep"][i]), "flow_time": float(run["ft"][i])}
    return cap

facts = {"flow_time_unit": FT_UNIT}; runs = {k: load(k) for k in RUNS}; runs = {k: v for k, v in runs.items() if v is not None}; print("runs available:", list(runs))
# ---------------------------------------------------------------- the tree
fig, ax = plt.subplots(figsize=(7.2, 3.1)); ax.axis("off"); pos = {}
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
# ---------------------------------------------------------------- the two designs' W2 at init
def w2_panel(ax, W2, title, vmax=None, cbar=True):
    vmax = vmax or np.abs(W2).max(); im = ax.imshow(W2, cmap=DIV, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto", interpolation="nearest"); ax.axvline(29.5, color=INK, lw=0.8)
    for b in (3.5, 11.5, 27.5): ax.axhline(b, color=INK, lw=0.5)
    ax.set_xticks([14.5, 45.5]); ax.set_xticklabels(["probe neurons (30)", "cue neurons (32)"]); ax.set_yticks([1.5, 7.5, 19.5, 43.5]); ax.set_yticklabels(["L0", "L1", "L2", "L3"]); ax.set_ylabel("second-layer unit (by node level)"); ax.set_title(title, fontsize=9); ax.grid(False)
    if cbar: plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    return im
r60 = runs.get("sym_d60_2ph") or runs.get("sgd_d60_2ph"); r70 = runs.get("sym_d70_2ph") or runs.get("sgd_d70_2ph")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), constrained_layout=True)
w2_panel(axes[0], r60["W2"][0], "design 60: $W_2$ at init", vmax=1.0); w2_panel(axes[1], r70["W2"][0], "design 70: $W_2$ at init", vmax=1.0)
fig.savefig(f"{OUT}/init_w2.pdf"); plt.close(fig)
# ---------------------------------------------------------------- trajectories (x = flow time)
def traj_panels(axes, run, title, C, idx_err, err, legend=True):
    ft = run["ft"]
    for l in range(4): axes[0].plot(ft, np.mean([C[:, p, HIERARCHY_TARGETS[p][0]] for p in LEVEL_ROWS[l]], axis=0), color=LEVEL_COL[l], lw=1.6, label=f"probes of level {l}")
    axes[0].plot(ft, np.mean([C[:, r, run["inv"][r][0]] for r in run["cue_rows"]], axis=0), color=CUE_COL, lw=1.6, ls="--", label="cues (with own leaf)")
    axes[0].set_ylim(0, 1.02); axes[0].set_ylabel("mean |cos| with own normal"); axes[0].set_title(title, fontsize=9)
    nrm = np.linalg.norm(run["W1"], axis=2)
    for l in range(4): axes[1].plot(ft, nrm[:, LEVEL_ROWS[l]].mean(1), color=LEVEL_COL[l], lw=1.6)
    axes[1].plot(ft, nrm[:, run["cue_rows"]].mean(1), color=CUE_COL, lw=1.6, ls="--"); axes[1].set_ylabel("mean weight norm")
    axes[2].plot(ft[1:], run["loss"][:len(ft) - 1], color=INK, lw=1.4, label="train loss (epoch mean)"); axes[2].plot(ft[idx_err], err, color=ERR_COL, lw=1.4, label="test error")
    axes[2].set_yscale("log"); axes[2].set_ylim(2e-4, 1); axes[2].set_ylabel("loss / error (log)"); axes[2].set_xlabel(f"flow time (cumulative step size, units of {FT_UNIT:.0f})")
    if run["freeze_ft"]:
        for a in axes: a.axvline(run["freeze_ft"], color=INK2, lw=0.7, ls=":")
    if legend: axes[0].legend(fontsize=7, loc="center right", frameon=True, framealpha=0.92, edgecolor="white"); axes[2].legend(fontsize=7, loc="lower left")
COS = {}; ERR = {}
for key, run in runs.items():
    C = cosines(run); idx, err = test_error(run); COS[key] = C; ERR[key] = (idx, err)
    facts[key] = {"folder": RUNS[key], "captures": captures(run, C), "final_loss": float(run["loss"][-1]), "min_loss": float(run["loss"].min()), "test_error_end": float(err[-1]), "epochs": int(run["ep"][-1]), "flow_time_end": float(run["ft"][-1]), "freeze_epochs": run["freeze"], "freeze_flow_time": run["freeze_ft"],
                  "loss_at_unfreeze": float(run["loss"][run["freeze"] - 1]) if run["freeze"] else None, "nodes_covered_end": int(len(set(np.abs(C[-1]).argmax(1)[np.abs(C[-1]).max(1) > 0.95])))}
    if key.endswith("2ph"):
        fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.7), sharex=True); traj_panels(axes, run, f"run {RUNS[key].split('_')[1][3:]}: {TITLE[key]} (dotted line: cues released)", C, idx, err); fig.savefig(f"{OUT}/traj_{key}.pdf"); plt.close(fig)
# one phase vs two phases, and SGD vs symmetrised: probe cosines by level against flow time
def overlay(ax, keys, styles, title, show_cues=True):
    for key, ls in zip(keys, styles):
        if key not in runs: continue
        run = runs[key]; C = COS[key]
        for l in range(4): ax.plot(run["ft"], np.mean([C[:, p, HIERARCHY_TARGETS[p][0]] for p in LEVEL_ROWS[l]], axis=0), color=LEVEL_COL[l], lw=1.4, ls=ls)
        if show_cues: ax.plot(run["ft"], np.mean([C[:, r, run["inv"][r][0]] for r in run["cue_rows"]], axis=0), color=CUE_COL, lw=1.4, ls=ls)
    ax.set_ylim(0, 1.02); ax.set_title(title, fontsize=8.5); ax.set_xlabel(f"flow time (units of {FT_UNIT:.0f})")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True, constrained_layout=True)
overlay(axes[0], ["sgd_d60_2ph", "sym_d60_2ph"], ["-", "--"], "design 60: SGD (solid) and symmetrised (dashed)"); overlay(axes[1], ["sgd_d70_2ph", "sym_d70_2ph"], ["-", "--"], "design 70: SGD (solid) and symmetrised (dashed)")
axes[0].set_ylabel("mean |cos| with own normal"); fig.savefig(f"{OUT}/overlay_sgd_vs_sym.pdf"); plt.close(fig)
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharey=True, sharex=True, constrained_layout=True)
for ax, (a, b, ttl) in zip(axes.ravel(), [("sgd_d60_2ph", "sgd_d60_1ph", "design 60, plain SGD"), ("sym_d60_2ph", "sym_d60_1ph", "design 60, symmetrised flow"), ("sgd_d70_2ph", "sgd_d70_1ph", "design 70, plain SGD"), ("sym_d70_2ph", "sym_d70_1ph", "design 70, symmetrised flow")]):
    overlay(ax, [a, b], ["-", "--"], ttl)
axes[0, 0].set_ylabel("mean |cos| with own normal"); axes[1, 0].set_ylabel("mean |cos| with own normal")
from matplotlib.lines import Line2D
axes[0, 0].legend(handles=[Line2D([], [], color=LEVEL_COL[l], lw=1.6, label=f"probes of level {l}") for l in range(4)] + [Line2D([], [], color=CUE_COL, lw=1.6, label="cues"), Line2D([], [], color=INK, lw=1.2, ls="-", label="two phases"), Line2D([], [], color=INK, lw=1.2, ls="--", label="one phase")], fontsize=6.5, loc="lower right", ncol=2, frameon=True, framealpha=0.92, edgecolor="white"); fig.savefig(f"{OUT}/overlay_phases.pdf"); plt.close(fig)
# loss curves of all runs on one axis (flow time)
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for key, run in runs.items():
    col = LEVEL_COL[0] if "d60" in key else LEVEL_COL[1]; ls = "-" if key.startswith("sym") else ":"; lw = 1.6 if key.endswith("2ph") else 1.0
    ax.plot(run["ft"][1:], run["loss"][:len(run["ft"]) - 1], color=col, ls=ls, lw=lw, label=TITLE[key])
ax.set_yscale("log"); ax.set_ylim(2e-3, 1); ax.set_ylabel("train loss (epoch mean, log)"); ax.set_xlabel(f"flow time (units of {FT_UNIT:.0f})"); ax.legend(fontsize=6.5, ncol=2, loc="lower left"); fig.savefig(f"{OUT}/losses_all.pdf"); plt.close(fig)
# ---------------------------------------------------------------- the learned representation (design 60, symmetrised, two phases, end)
ref = r60; W1e, W2e, Woe = ref["W1"][-1], ref["W2"][-1], ref["Wo"][-1]; lev = np.array([NODE_LEVEL[j] for j in ref["unode"]]); refname = RUNS[ref["key"]].split("_")[1][3:]
fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), constrained_layout=True)
C1 = W1e / np.linalg.norm(W1e, axis=1, keepdims=True) @ U.T; im = axes[0].imshow(C1.T, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1), aspect="auto", interpolation="nearest"); axes[0].axvline(29.5, color=INK, lw=0.8)
axes[0].set_xticks([14.5, 45.5]); axes[0].set_xticklabels(["probe neurons (30)", "cue neurons (32)"]); axes[0].set_yticks([0, 7, 15, 30]); axes[0].set_yticklabels([r"$u_0$", r"$u_7$", r"$u_{15}$", r"$u_{30}$"]); axes[0].set_ylabel("tree normal"); axes[0].set_title(f"run {refname}, end: cosine of each first-layer weight vector with each tree normal", fontsize=9); axes[0].grid(False); plt.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02)
w2_panel(axes[1], W2e, f"run {refname}, end: $W_2$ (vetoes on the own and ancestors' probes; one cue per level-3 unit)")
fig.savefig(f"{OUT}/learned_w1_w2.pdf"); plt.close(fig)
# readout magnitudes by level against flow time (replaces the readout bar chart)
fig, ax = plt.subplots(figsize=(7.2, 2.4))
for l in range(4): ax.plot(ref["ft"], np.abs(ref["Wo"][:, 0, lev == l]).mean(1), color=LEVEL_COL[l], lw=1.6, label=f"units of level {l}")
if ref["freeze_ft"]: ax.axvline(ref["freeze_ft"], color=INK2, lw=0.7, ls=":")
ax.set_ylabel("mean |readout weight|"); ax.set_xlabel(f"flow time (units of {FT_UNIT:.0f})"); ax.set_title(f"run {refname}: readout magnitude by the level of the unit's node", fontsize=9); ax.legend(fontsize=7, loc="upper left"); fig.savefig(f"{OUT}/readout_by_level.pdf"); plt.close(fig)
# one level-3 unit
u3 = int(np.where(lev == 3)[0][0]); j3 = int(ref["unode"][u3]); anc = [a for a in node_ancestors(j3) if a != j3]
fig, ax = plt.subplots(figsize=(7.2, 2.6)); w = W2e[u3]; cols = [LEVEL_COL[NODE_LEVEL[HIERARCHY_TARGETS[r][0]]] if r < 30 else CUE_COL for r in range(62)]
ax.bar(np.arange(62), w, color=cols, width=0.8); ax.axvline(29.5, color=INK, lw=0.8); ax.set_xticks([14.5, 45.5]); ax.set_xticklabels(["probe neurons (coloured by the level of their node)", "cue neurons"]); ax.set_ylabel("$W_2$ weight"); ax.set_title(f"incoming $W_2$ weights of unit {u3} (node {j3}, level 3) at the end of run {refname}", fontsize=9)
big_cues = [r for r in ref["cue_rows"] if abs(w[r]) > 0.5 * abs(w[ref["cue_rows"]]).max()]
for r in [2 * a for a in anc] + [2 * j3, 2 * j3 + 1] + big_cues:
    if abs(w[r]) > 0.3: ax.annotate(("$u_{%d}$" % HIERARCHY_TARGETS[r][0]) if r < 30 else f"cue {ref['inv'][r]}", (r, w[r]), textcoords="offset points", xytext=(0, -10 if w[r] < 0 else 4), ha="center", fontsize=8, color=INK)
fig.savefig(f"{OUT}/unit_example.pdf"); plt.close(fig)
facts["example_unit"] = {"run": refname, "unit": u3, "node": j3, "ancestors": anc, "weights_on_probes_of_ancestors": {int(a): [round(float(W2e[u3, 2 * a]), 2), round(float(W2e[u3, 2 * a + 1]), 2)] for a in anc}, "own_probe_weights": [round(float(W2e[u3, 2 * j3]), 2), round(float(W2e[u3, 2 * j3 + 1]), 2)], "largest_cue_weights": sorted([(round(float(W2e[u3, r]), 2), str(ref["inv"][r])) for r in ref["cue_rows"]], key=lambda t: -abs(t[0]))[:6]}
# unit statistics, ablations, norms
cells = leaf_cells(xe, U); cell_leaf = cells // 2; H1 = np.maximum(xe @ W1e.T, 0); H2 = np.maximum(H1 @ W2e.T, 0); contrib = H2 * Woe.ravel(); stats = []
for u in range(60):
    act = H2[:, u] > 0; frac = act.mean(); top = 0.0; corr = 0.0
    if act.sum() > 20: counts = np.bincount(cell_leaf[act], minlength=16); top = counts.max() / counts.sum(); corr = np.corrcoef(contrib[act, u], ye[act])[0, 1]
    stats.append({"unit": u, "level": int(lev[u]), "active_frac": frac, "top_leaf_share": top, "corr_label": corr, "readout": Woe.ravel()[u]})
stats = pd.DataFrame(stats); facts["unit_stats"] = stats.groupby("level").agg(active_frac=("active_frac", "mean"), top_leaf_share=("top_leaf_share", "mean"), corr_label=("corr_label", "mean"), readout_abs=("readout", lambda s: float(np.abs(s).mean()))).round(3).reset_index().to_dict("records")
def err_keep(keep):
    f = np.maximum(np.maximum(xe @ W1e.T, 0) @ W2e[keep].T, 0) @ Woe.ravel()[keep]; return float(((f >= 0) != (ye > 0)).mean())
facts["ablation"] = {"full": err_keep(np.arange(60)), "L3_only": err_keep(np.where(lev == 3)[0]), "L0-2_only": err_keep(np.where(lev <= 2)[0]), "L0_removed": err_keep(np.where(lev >= 1)[0])}
nrm = np.linalg.norm(W1e, axis=1); facts["norms_end"] = {f"L{l}": float(nrm[LEVEL_ROWS[l]].mean()) for l in range(4)}; facts["norms_end"]["cues"] = float(nrm[ref["cue_rows"]].mean())
facts["w2_end"] = {"veto_on_own_probe_L3": float(np.mean([min(W2e[u, 2 * ref["unode"][u]], W2e[u, 2 * ref["unode"][u] + 1]) for u in range(60) if lev[u] == 3])), "mean_abs_cue_weight_L3": float(np.abs(W2e[lev == 3][:, 30:]).mean())}
# ---------------------------------------------------------------- orbits
units = list(zip(ref["unode"].tolist(), ref["ublock"].tolist(), ref["usign"].tolist())); orb = symmetric_orbits(ref["cid"], units); f1 = orb["forced_zero"]; rng = np.random.default_rng(3); perm = rng.permutation(orb["n_orbits"])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True); BLACK = LinearSegmentedColormap.from_list("z", ["#0b0b0b", "#0b0b0b"])
o1 = orb["W1"][0]; axes[0].imshow(perm[o1].T, cmap="tab20", aspect="auto", interpolation="nearest", alpha=0.9); axes[0].imshow(np.ma.masked_where(~f1[o1], np.ones(o1.shape)).T, cmap=BLACK, aspect="auto", interpolation="nearest")
axes[0].axvline(29.5, color="white", lw=1.2); axes[0].axhline(30.5, color="white", lw=1.2); axes[0].set_xticks([14.5, 45.5]); axes[0].set_xticklabels(["probe neurons", "cue neurons"]); axes[0].set_yticks([15, 45]); axes[0].set_yticklabels(["in-span\n$u_0..u_{30}$", "off-span\n$o_0..o_{29}$"]); axes[0].set_title("$W_1$ coordinates: orbit (colour), forced zero (black)", fontsize=8.5); axes[0].grid(False)
o2 = orb["W2"][0]; axes[1].imshow(perm[o2], cmap="tab20", aspect="auto", interpolation="nearest", alpha=0.9); axes[1].imshow(np.ma.masked_where(~f1[o2], np.ones(o2.shape, dtype=float)), cmap=BLACK, aspect="auto", interpolation="nearest")
axes[1].axvline(29.5, color="white", lw=1.2); axes[1].set_xticks([14.5, 45.5]); axes[1].set_xticklabels(["probe neurons", "cue neurons"]); axes[1].set_yticks([1.5, 7.5, 19.5, 43.5]); axes[1].set_yticklabels(["L0", "L1", "L2", "L3"]); axes[1].set_title("$W_2$ coordinates: orbit (colour), forced zero (black)", fontsize=8.5); axes[1].grid(False)
fig.savefig(f"{OUT}/orbits.pdf"); plt.close(fig); facts["orbits"] = {"n_orbits": int(orb["n_orbits"]), "n_free": int(orb["n_free"]), "n_coordinates": int(62 * 61 + 60 * 62 + 60)}
# ---------------------------------------------------------------- the linear model
z = xe @ U[15:].sum(0); t = ye * z
def bisect(f, lo, hi):
    for _ in range(60):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)
c = bisect(lambda c: np.mean(-t / (1 + np.exp(c * t))), 0.01, 10); facts["linear"] = {"c": float(c), "loss": float(np.mean(np.logaddexp(0, -c * t))), "error": float(np.mean(np.sign(z) != ye))}
json.dump(facts, open("notes/handout/facts.json", "w"), indent=1, default=float)
# ---------------------------------------------------------------- facts.tex: every number the handout quotes, as \fact{run-key}{name}
lines = []
def deff(key, name, value): lines.append("\\expandafter\\def\\csname fact." + key + "." + name + "\\endcsname{" + str(value) + "}")
def cap_str(c): return "--" if c is None else str(c["epoch"])
def capft_str(c): return "--" if c is None else f"{c['flow_time']:.1f}"
for key, v in facts.items():
    if key not in RUNS: continue
    k = key.replace("_", "-"); deff(k, "num", v["folder"].split("_")[1][3:]); deff(k, "loss", f"{v['final_loss']:.4f}"); deff(k, "err", f"{100 * v['test_error_end']:.2f}"); deff(k, "nodes", v["nodes_covered_end"]); deff(k, "ftend", f"{v['flow_time_end']:.0f}")
    deff(k, "lossunfreeze", "--" if v["loss_at_unfreeze"] is None else f"{v['loss_at_unfreeze']:.3f}")
    for l in range(4): deff(k, f"capL{l}", cap_str(v["captures"][f"L{l}"])); deff(k, f"capL{l}ft", capft_str(v["captures"][f"L{l}"]))
    deff(k, "cues", cap_str(v["captures"]["cues"])); deff(k, "cuesft", capft_str(v["captures"]["cues"]))
for st in facts["unit_stats"]:
    l = st["level"]; deff("units", f"activeL{l}", f"{100 * st['active_frac']:.1f}"); deff("units", f"topL{l}", f"{100 * st['top_leaf_share']:.0f}"); deff("units", f"readoutL{l}", f"{st['readout_abs']:.1f}"); deff("units", f"corrL{l}", f"{st['corr_label']:.2f}")
for name, val in facts["ablation"].items(): deff("ablation", name.replace("_", "").replace("-", ""), f"{100 * val:.2f}")
for name, val in facts["norms_end"].items(): deff("norms", name, f"{val:.1f}")
ex = facts["example_unit"]; deff("unit", "num", ex["unit"]); deff("unit", "node", ex["node"]); deff("unit", "ownveto", f"{min(ex['own_probe_weights']):.1f}"); deff("unit", "ancvetoes", ", ".join(f"{min(w):.1f}" for w in ex["weights_on_probes_of_ancestors"].values())); deff("unit", "anclist", ", ".join(f"$u_{{{a}}}$" for a in ex["ancestors"])); deff("unit", "cueveto", f"{ex['largest_cue_weights'][0][0]:.1f}"); deff("unit", "cueid", ex["largest_cue_weights"][0][1].replace("(", "").replace(")", "").replace(" ", "")); deff("unit", "run", ex["run"])
deff("orbits", "n", facts["orbits"]["n_orbits"]); deff("orbits", "free", facts["orbits"]["n_free"]); deff("orbits", "coords", facts["orbits"]["n_coordinates"]); deff("linear", "c", f"{facts['linear']['c']:.2f}"); deff("linear", "loss", f"{facts['linear']['loss']:.3f}"); deff("linear", "err", f"{100 * facts['linear']['error']:.1f}")
r2 = runs["sym_d60_2ph"]; deff("sched", "freezeft", f"{r2['freeze_ft']:.0f}" if r2["freeze_ft"] else "--"); deff("sched", "epochs", int(r2["ep"][-1])); deff("sched", "warm", int(r2["d"].get("warm_epochs", 0))); deff("sched", "freeze", int(r2["freeze"]))
open("notes/handout/facts.tex", "w").write("% generated by make_figures.py: \\fact{key}{name} -> \\csname fact.key.name\\endcsname\n" + "\n".join(lines) + "\n")
print("facts.tex:", len(lines), "macros;", {k: (v["folder"], round(v["final_loss"], 4), round(100 * v["test_error_end"], 2), v["nodes_covered_end"]) for k, v in facts.items() if k in RUNS})
