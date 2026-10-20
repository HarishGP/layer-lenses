"""Summary table of runs 60, 70, 72-78 and the x1 variants (loss, error, capture, probe fates, cues, readout, unit modes by level).
Writes results/designed_init/symmetric_flow_summary.md."""
import numpy as np, pickle
from layer_lenses.designed_init import NODE_LEVEL, HIERARCHY_TARGETS, node_ancestors, cue_row_identity
from layer_lenses.init_sweep import SweepRunSpec, build_data, load_tree
tree = load_tree("results/designed_init/hd2_run60_sgd_lr1_1000/tree.npz"); U = tree.w_list
x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=999_777, init_seed=0, batch_size=1, n_train=100_000), tree); y = y.astype(float)
RUNS = [("60", "hd2_run60_sgd_lr1_1000", "design", "SGD (sample)", "500 + 500", 80), ("74", "hd2_run74_sym_pgf_run60", "design", "symmetrised", "500 + 500", 80), ("74 x1", "hd2_run74_sym_pgf_run60_outx1", "design", "symmetrised", "500 + 500", 1),
        ("75", "hd2_run75_sym_pgf_run60_partner", "design + partner +1", "symmetrised", "500 + 500", 80), ("75 x1", "hd2_run75_sym_pgf_run60_partner_outx1", "design + partner +1", "symmetrised", "500 + 500", 1),
        ("77", "hd2_run77_sym_pgf_run60_one_phase", "design", "symmetrised", "1000, none frozen", 80), ("77 x1", "hd2_run77_sym_pgf_run60_one_phase_outx1", "design", "symmetrised", "1000, none frozen", 1),
        ("78", "hd2_run78_sym_pgf_run60_partner_one_phase", "design + partner +1", "symmetrised", "1000, none frozen", 80), ("78 x1", "hd2_run78_sym_pgf_run60_partner_one_phase_outx1", "design + partner +1", "symmetrised", "1000, none frozen", 1),
        ("70", "hd2_run70_symcue_partner", "majority 0.2/0.1 + partner", "SGD (sample)", "500 + 500", 1), ("72", "hd2_run72_sym_pgf_phase1", "majority 0.2/0.1 + partner", "symmetrised", "500 frozen (phase 1 only)", 1), ("73", "hd2_run73_sym_pgf_two_phase", "majority 0.2/0.1 + partner", "symmetrised", "1000 + 500", 1), ("76", "hd2_run76_sym_pgf_one_phase", "majority 0.2/0.1 + partner", "symmetrised", "1500, none frozen", 1)]
def mode(own, partner):
    if own <= -5 and partner >= 5: return f"both ({own:+.0f}/{partner:+.0f})"
    if own <= -5: return f"veto ({own:+.0f})"
    if partner >= 5: return f"support ({partner:+.0f})"
    return f"inert ({own:+.0f}/{partner:+.0f})"
import glob, os
SWEEP = {"a3_1.5": "alpha_3 = 1.5 (cue bias 0.5)", "a3_4": "alpha_3 = 4 (cue bias 3)", "blocks_x0.5": "cue blocks x 0.5", "blocks_x2": "cue blocks x 2", "a6_0.1": "+ alpha_6 = 0.1 (coherent reading outside the subtree)", "a5_0": "alpha_5 = 0 (open-side blocks only)", "veto_2": "veto -2", "readout_0.4_1.6": "readouts 0.4 (L0-2) / 1.6 (L3)", "a11_8": "readout rate x8"}
for folder in sorted(glob.glob("results/designed_init/hd2_run*_alpha_*"), key=lambda f: int(os.path.basename(f).split("_")[1][3:])):
    base = os.path.basename(folder); n = base.split("_")[1][3:]; key = base[len(f"hd2_run{n}_alpha_"):].rsplit("_", 1)[0]; ph = base.endswith("_2ph")
    RUNS.append((n, base, "design, " + SWEEP[key], "symmetrised", "500 + 500" if ph else "1000, none frozen", 8 if key == "a11_8" else 80))
rows = []
for label, f, init, kind, phases, outs in RUNS:
    z = np.load(f"results/designed_init/{f}/replay/weights.npz"); meta = pickle.load(open(f"results/designed_init/{f}/replay/meta.pkl", "rb")); d = meta["design"]; L = np.array(meta["out"]["epoch_losses"], float); L = L[~np.isnan(L)]; cr = d["cue_rows"]; unode = np.array(d["unit_node"]); fc = int(d.get("freeze_cues", 500 if label in ("60", "70") else 0))
    W = z["w__hidden_layers.0.weight"].astype(float); W2 = z["w__hidden_layers.1.weight"].astype(float); Wo = z["w__output_layer.weight"].astype(float); ep = list(z["epochs"]); cid = cue_row_identity(W[0], tree, cr); inv = {v: k for k, v in cid.items()}
    cosm = lambda e: W[ep.index(e)] / np.linalg.norm(W[ep.index(e)], axis=1, keepdims=True) @ U.T
    first = {}; cue_first = None
    for l in range(4):
        ps = [p for p in range(30) if NODE_LEVEL[HIERARCHY_TARGETS[p][0]] == l]; first[l] = None
        for e in ep:
            Ce = cosm(e)
            if all(abs(Ce[p]).argmax() == HIERARCHY_TARGETS[p][0] and abs(Ce[p]).max() > 0.95 for p in ps): first[l] = int(e); break
    for e in ep:
        Ce = cosm(e)
        if all(abs(Ce[r]).argmax() == inv[r][0] and abs(Ce[r]).max() > 0.95 for r in cr): cue_first = int(e); break
    C = cosm(ep[-1]); nrm = np.linalg.norm(W[-1], axis=1); logit = np.maximum(np.maximum(x @ W[-1].T, 0) @ W2[-1].T, 0) @ Wo[-1].ravel()
    fate = []
    for l in range(4):
        ps = [p for p in range(30) if NODE_LEVEL[HIERARCHY_TARGETS[p][0]] == l]; tops = [int(np.abs(C[p]).argmax()) for p in ps]; cs = [abs(C[p][HIERARCHY_TARGETS[p][0]]) for p in ps]
        on = sum(1 for p, t, c in zip(ps, tops, cs) if t == HIERARCHY_TARGETS[p][0] and c > 0.95); par = sum(1 for p, t in zip(ps, tops) if t in node_ancestors(HIERARCHY_TARGETS[p][0]) and abs(C[p]).max() > 0.95)
        rest = [c for p, t, c in zip(ps, tops, cs) if not (t == HIERARCHY_TARGETS[p][0] and c > 0.95) and not (t in node_ancestors(HIERARCHY_TARGETS[p][0]) and abs(C[p]).max() > 0.95)]
        fate.append(f"L{l}: {on}/{len(ps)} target" + (f", {par} anc" if par else "") + (f", {len(rest)} off (|cos| {np.mean(rest):.2f})" if rest else ""))
    own = np.array([[p for p in range(30) if W2[0][u, p] < 0][0] for u in range(60)]); partner = np.where(own % 2 == 0, own + 1, own - 1)
    modes = [mode(np.mean([W2[-1][u, own[u]] * nrm[own[u]] for u in range(60) if NODE_LEVEL[unode[u]] == l]), np.mean([W2[-1][u, partner[u]] * nrm[partner[u]] for u in range(60) if NODE_LEVEL[unode[u]] == l])) for l in range(4)]
    act = (np.maximum(np.maximum(x @ W[-1].T, 0) @ W2[-1].T, 0) > 0).mean(0); acts = [np.mean([act[u] for u in range(60) if NODE_LEVEL[unode[u]] == l]) for l in range(4)]
    cues_on = sum(1 for r in cr if abs(C[r]).argmax() == inv[r][0] and abs(C[r]).max() > 0.95); cue_cos = np.mean([abs(C[r][inv[r][0]]) for r in cr])
    rows.append([label, init, kind, phases, str(outs), f"{L[-1]:.4f}" + (f" ({L[fc - 1]:.3f})" if fc else ""), f"{((logit >= 0) != (y > 0)).mean() * 100:.2f}%", str(len(set(np.abs(C).argmax(1)[np.abs(C).max(1) > 0.95]))),
                 " / ".join(str(first[l]) if first[l] is not None else "-" for l in range(4)), "; ".join(fate), f"{cues_on}/32" + (f" (by {cue_first})" if cue_first else f" (|cos| {cue_cos:.2f})"), f"{np.abs(Wo[-1]).mean():.1f}", " ; ".join(modes), " / ".join(f"{a:.2f}" for a in acts)])
hdr = ["run", "init", "gradient", "phases (epochs)", "out rate", "loss (at release)", "test err", "nodes", "capture epoch L0/L1/L2/L3", "probe fate by level", "cues on normal", "readout", "unit mode L0 ; L1 ; L2 ; L3 (own/partner x norm)", "activity L0-L3"]
md = "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows)
open("results/designed_init/symmetric_flow_summary.md", "w").write("# Runs 60, 70, 72-78 and the x1 variants: summary (generated 2026-10-19)\n\nUnit mode per level from the mean weight x row norm on the unit's own probe (veto if <= -5) and on its pole partner (support if >= 5); 'both' = a signed-coordinate reader; 'inert' = neither. 'off' probes: not aligned (|cos| > 0.95) with their target or an ancestor.\n\n" + md + "\n"); print(md)
