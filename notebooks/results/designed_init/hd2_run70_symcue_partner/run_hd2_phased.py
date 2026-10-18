"""Run 51 (notebook 2, section 6): run 49's init (62-60-2, readout e = 1.633 / w = 0.5, 10M points) with a PHASED protocol:
  epochs    0- 500: the 32 cue rows frozen (probes, W2, W3, output train);
  epochs  500-1000: the 30 probe rows frozen (cues, W2, W3, output train);
  epochs 1000-1200: nothing frozen.
Three segments of `train_relu_mlp_with_optimizer_snapshots` chained with `start_epoch` and the previous segment's optimizer state, so the
momentum buffers and the cosine schedule (lr 0.01 -> 0.0005 over the 1200 epochs) continue across the phase changes; batch 10k, 1000
steps per epoch, snapshots every epoch. Saved as one replayable run (segments merged).
  run 51: --out hd2_run51_phased_10M --tag hd2_run51_phased_10M_seed0
  run 70: run 68 + --partner-probe-weight 1: every unit also reads the pole partner of its own probe (the other probe of its node) with
          +1: units 0 and 2 (veto on p0) read p1 with +1, units 1 and 3 read p0 with +1, units 4 and 6 read p3, 5 and 7 read p2, etc.
  run 68: --two-layer --momentum 0 --lr 0.1 --phases 500 500 --symmetric-cue-blocks 0.2 0.1 (readouts +-0.816, output rate x1): the
          majority-label blocks with the readouts kept large and no x80, so the coherent initial logit (~5) is shrunk by the layers below.
  run 67 (pilot): --symmetric-cue-blocks 0.2 0.1: every cue-to-unit weight = sign(readout) x pole x a, i.e. the path weight through the unit
          agrees with the majority label of {x : pole u_m . x > 0}; a = 0.2 for the leaves under the unit's node on its open side (the
          side its veto leaves), 0.1 for every other cue (vetoed side and outside the subtree). Probe links -1, other probe weights 0, rows
          not renormalised. Readouts +-(w3_entry x out_weight) as before: the pilot uses +-0.01 so the coherent initial logit is small.
  runs 64-66: run 63's gradient cue blocks with the initial logit under control: 64 --zero-readout (blocks built with the design readout
          signs, then Wout = 0: the readouts regrow with each unit's own sign); 65 --out-scale 1 (no x80 on the output row); 66
          --gradient-cue-scale 0.0075 (blocks of norm 0.0075, bringing the initial logit rms from 6.7 to run 60's 0.05; probe links stay -1).
  run 63: --two-layer --out-scale 80 --momentum 0 --lr 0.1 --phases 500 500 --gradient-cue-blocks: run 60 with every unit's cue block
          replaced by the gradient-descent direction of a single-unit network at the ideal first layer (rows +-u_j at norm sqrt 2, by the
          design's row identities) whose unit reads its own probe with -1, every cue row with +1/100 and nothing else, with the unit's run-60
          readout sign (population gradient over 400k training points, actual gradient; see notebooks/population_gradients_at_inits.ipynb
          section 6). The descent restricted to the 32 cue columns is normalised to norm 1 (the design block's norm), the +1/100 baseline is
          dropped, probe links (-1) and all other weights stay as in run 60; rows renormalised to sqrt 2 as usual. The flip test is skipped.
  run 62: --two-layer --out-scale 80 --momentum 0 --lr 0.1 --phases 500 500 --cue-noise-ratio 1: run 60 with every unit's cue block v_j
          (its 32 cue weights) replaced by v_j + n_j, n_j random in the orthogonal complement of span{v_0..v_59} (the 15-dimensional span
          of the design's side-contrast blocks, so the complement has 17 dimensions), |n_j| = ratio x |v_j|, then rescaled to |v_j|.
          First-order lens prediction: every internal-node contrast of n_j is zero, so the pushes on the probes keep their directions and
          shrink by 1 / sqrt(1 + ratio^2). The flip test is skipped.
  run 61: --two-layer --out-scale 80 --momentum 0 --lr 0.1 --phases 500 --schedule-epochs 1000 --random-cue-block 1: phase 1 of run 60 with
          the cue weights of node 1's four units (4-7) replaced by a random vector v over the 32 cue columns: units 4 and 5 read v, units 6
          and 7 read -v (|v| = the design block's norm; rows renormalised to sqrt 2 as usual). The probes of node 1 (rows 2, 3) lose their
          designed push toward +-u_1; the question is where they go. The flip test is skipped (the init is no longer equivariant).
  run 60: --two-layer --out-scale 80 --momentum 0 --lr 0.1 --phases 500 500: run 56 with plain SGD at lr 0.1 (run 56's effective step
          lr / (1 - momentum) = 0.1), same 1000-epoch schedule: the same integrated effective learning rate as run 56.
  run 59: --two-layer --out-scale 80 --momentum 0 --lr 0.02 --phases 1000 1000: run 56 with plain SGD (no momentum), twice the learning
          rate and twice the epochs (cosine over 2000 epochs; cues frozen 0-1000, free 1000-2000).
  runs 57 / 58: --two-layer --out-scale 80 --phases 500 --schedule-epochs 1000 --project-out 0 (resp. 1): phase 1 of run 56 alone (same lr
          schedule and data order), with every first-layer gradient projected onto the orthogonal complement of u_0 (resp. u_1) by a
          gradient hook, so no row can move along that normal. Tests whether capturing u_0 is needed for u_1 / u_2 (resp. u_1 for its subtree).
  run 56: --two-layer --out-scale 80 --phases 500 500: run 55 with the output row's gradient scaled by 80 (parameter_update_mask value),
          the shared-mode growth rate 30 e^2 + 2 w^2 of run 53's factorised readout at init, so the readout grows at run 53's initial pace.
  run 55: --two-layer --phases 500 500: run 53 WITHOUT the third hidden layer (62-60): each unit is wired straight to the logit with weight
          path sign x (w3_entry x out_weight) = +-0.816, the third layer's effective path weight to the logit at run 53's init, so the
          logit function at init is run 53's (its third-layer ReLUs gate < 3% of inputs). Only the readout's growth dynamics differ.
  run 54: --negate-w2 --phases 500 500: run 53 with W2 negated at init (every probe weight +1 instead of -1, every cue block B -> -B,
          i.e. the same four blocks per node attached to the opposite probe weight sign); the flip equivariance is unchanged.
  run 53: --cue-bias 1.0 --phases 500 500: the bias-1 control of run 52 (same two-phase 1000-epoch schedule).
  run 52: --cue-bias 0.5 --phases 500 500: cues = pole * (v + 0.5 u_j) renormalised (half the level-4 bias), TWO phases: cues frozen for
          500 epochs, then nothing frozen for 500 (cosine over the 1000 epochs, as run 49).
"""
import argparse, json, os, time, numpy as np, pandas as pd, torch
from layer_lenses.designed_init import node_ancestors, NODE_LEVEL, build_hierarchy_model, hierarchy_probe_table, logit_scale, standard_scale_spec, equivariant_quadruple_weights, flip_equivariance_report
from layer_lenses.init_sweep import SweepRunSpec, build_tree, build_data, build_data_chunked, save_tree
from layer_lenses.relu_mlp import ReLUMLP
from layer_lenses.relu_training import ReLUTrainConfig, evaluate_relu_mlp, checkpoint_model_from_state
from layer_lenses.relu_analysis import train_relu_mlp_with_optimizer_snapshots, designed_run_seed_result, save_designed_run
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--data-seed", type=int, default=5178); ap.add_argument("--orth-seed", type=int, default=0)
ap.add_argument("--n-train", type=int, default=10_000_000); ap.add_argument("--phases", type=int, nargs="+", default=[500, 500, 200], help="3 values: epochs with cues frozen, with probes frozen, with nothing frozen; 2 values: cues frozen, nothing frozen"); ap.add_argument("--cue-bias", type=float, default=1.0); ap.add_argument("--negate-w2", action="store_true")
ap.add_argument("--w3-entry", type=float, default=1.633); ap.add_argument("--out-weight", type=float, default=0.5); ap.add_argument("--two-layer", action="store_true", help="run 55: no third layer, Wout = path sign x w3_entry x out_weight per unit"); ap.add_argument("--out-scale", type=float, default=1.0, help="run 56: gradient scale on the output row (mask value)")
ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--momentum", type=float, default=0.9)
ap.add_argument("--random-cue-block", type=int, default=None, help="run 61: node whose four units get cue weights v, v, -v, -v with v random"); ap.add_argument("--cue-block-seed", type=int, default=61)
ap.add_argument("--cue-noise-ratio", type=float, default=0.0, help="run 62: |n_j| / |v_j| for the random cue-block perturbation orthogonal to span{v_j}"); ap.add_argument("--cue-noise-seed", type=int, default=62)
ap.add_argument("--gradient-cue-blocks", action="store_true", help="run 63: cue blocks = descent direction of the single-unit network at the ideal first layer (c = -1, 1/100 on the cues)"); ap.add_argument("--symmetric-cue-blocks", type=float, nargs=2, default=None, metavar=("A_OPEN", "A_OTHER")); ap.add_argument("--partner-probe-weight", type=float, default=0.0, help="run 70: weight from the pole partner of the unit's own probe"); ap.add_argument("--gradient-cue-points", type=int, default=400_000); ap.add_argument("--gradient-cue-scale", type=float, default=1.0, help="run 66: norm of each gradient cue block"); ap.add_argument("--zero-readout", action="store_true", help="run 64: Wout = 0 at init (after the blocks are built with the design readout signs)")
ap.add_argument("--schedule-epochs", type=int, default=None, help="length of the cosine schedule if different from the sum of the phases (to run only the first phases of a longer protocol)")
ap.add_argument("--project-out", type=int, nargs="*", default=[], help="runs 57-58: ODT node indices whose normals are projected out of every first-layer gradient")
args = ap.parse_args(); OUT = f"results/designed_init/{args.out}"; os.makedirs(f"{OUT}/runs", exist_ok=True); T0 = time.time()
N_TRAIN, BS, LR, MOM, ETA_MIN_RATIO, SNAP = args.n_train, 10_000, args.lr, args.momentum, 0.05, 1
assert len(args.phases) in (1, 2, 3); EPOCHS = args.schedule_epochs or sum(args.phases); _b = np.cumsum([0] + args.phases); BOUNDS = [(int(_b[i]), int(_b[i + 1])) for i in range(len(args.phases))]
assert EPOCHS >= sum(args.phases); LAST = int(_b[-1])
PHASE_FROZEN = {3: ["cues", "probes", "none"], 2: ["cues", "none"], 1: ["cues"]}[len(args.phases)]
DESIGN = dict(seed=0, cue_mode="linear", subset_mode="all", units_per_level=(1, 1, 1, 1), width1=80, width2=30, width3=40, cue_offspan_ratio=0.2,
              output_signs="balanced", path_sign_cues=True, layer3_mode="balanced_paths", layer3_majority=15, cue_block_scale="inv_sqrt")
DATA = dict(kind="sweep_chunked", tree_seed=5178, data_seed=args.data_seed, n_train=N_TRAIN, chunk_size=1_000_000, eval_seed=999_777, n_eval=200_000, dim=100, depth=5)
tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); save_tree(tree, f"{OUT}/tree.npz"); U = tree.w_list
x, y = build_data_chunked(SweepRunSpec(tree_seed=5178, data_seed=args.data_seed, init_seed=0, batch_size=1, n_train=N_TRAIN), tree)
xe, ye = build_data(SweepRunSpec(tree_seed=5178, data_seed=999_777, init_seed=0, batch_size=1, n_train=200_000), tree)
tag = args.tag; t0 = time.time()
hs = standard_scale_spec("kaiming", **DESIGN); base, meta = build_hierarchy_model(hs, tree); v = U[15:31].sum(0)
W1 = base.hidden_layers[0].weight.detach().numpy().astype(np.float64).copy(); W2 = base.hidden_layers[1].weight.detach().numpy().astype(np.float64)
W3 = base.hidden_layers[2].weight.detach().numpy().astype(np.float64); Wout = base.output_layer.weight.detach().numpy().astype(np.float64)
for r in meta["cue_rows"]:
    c = W1[r] / np.linalg.norm(W1[r]) @ U.T; m = int(np.abs(c).argmax()); pole = int(np.sign(c[m])); assert m >= 15 and abs(c[m]) > 0.9
    new = pole * (v + args.cue_bias * U[m]); W1[r] = new / np.linalg.norm(new) * np.linalg.norm(W1[r])
q = equivariant_quadruple_weights(W1, W2, W3, Wout, cue_rows=meta["cue_rows"], tree=tree, orth_seed=args.orth_seed)
W1 = q["W1"][:62]; W2 = q["W2"][:, :62]; CUE_ROWS = list(meta["cue_rows"]); PROBE_ROWS = list(range(30))
s_units = np.sign(q["W3"].T @ q["Wout"].ravel()); assert s_units.tolist() == [1, -1, -1, 1] * 15
W3 = np.stack([(s_units > 0).astype(np.float64), (s_units < 0).astype(np.float64)]) * args.w3_entry; Wout = np.array([[args.out_weight, -args.out_weight]])
W1 = W1 / np.linalg.norm(W1, axis=1, keepdims=True) * np.sqrt(2.0); W2 = W2 / np.linalg.norm(W2, axis=1, keepdims=True) * np.sqrt(2.0)
if args.negate_w2: W2 = -W2                                                    # run 54
if args.random_cue_block is not None:                                          # run 61: random cue vector on one node's quadruple
    j = args.random_cue_block; us = [4 * j + t for t in range(4)]; vr = np.random.default_rng(args.cue_block_seed).standard_normal(len(CUE_ROWS)); vr *= np.linalg.norm(W2[us[0], CUE_ROWS]) / np.linalg.norm(vr)
    for u, sgn in zip(us, (1, 1, -1, -1)): W2[u, CUE_ROWS] = sgn * vr
    cue_pole = {r: int(np.sign((W1[r] / np.linalg.norm(W1[r])) @ U.T)[np.abs((W1[r] / np.linalg.norm(W1[r])) @ U.T).argmax()]) for r in CUE_ROWS}; cue_node = {r: int(np.abs((W1[r] / np.linalg.norm(W1[r])) @ U.T).argmax()) for r in CUE_ROWS}
    def side_of(a, m):
        c = m
        while (c - 1) // 2 != a: c = (c - 1) // 2
        return 1 if c == 2 * a + 2 else -1
    contrast = {a: sum(side_of(a, cue_node[r]) * cue_pole[r] * vr[i] for i, r in enumerate(CUE_ROWS) if a in node_ancestors(cue_node[r])) for a in range(15)}
    print(f"random cue block on node {j} (seed {args.cue_block_seed}): side-signed cue contrast of v for each internal node (lens prediction of the push on probes {2 * j}, {2 * j + 1}; the design gives {sum(abs(vr[i]) for i, r in enumerate(CUE_ROWS) if j in node_ancestors(cue_node[r])):.2f} on node {j} and 0 elsewhere): " + ", ".join(f"n{a}: {c:+.2f}" for a, c in contrast.items()), flush=True)
if args.two_layer:                                                             # run 55: the output row is the third layer's linear map at init
    Wout = (s_units * args.w3_entry * args.out_weight)[None, :]; W3 = None
_W3, _Wo = (Wout, np.ones((1, 1))) if args.two_layer else (W3, Wout)           # two-layer: the output row plays W3's part in the flip test
if args.cue_noise_ratio > 0:                                                  # run 62: cue blocks perturbed in the complement of their span
    V = W2[:, CUE_ROWS].copy(); Uv, sv, _ = np.linalg.svd(V, full_matrices=False); rank = int((sv > 1e-8 * sv[0]).sum()); Q = np.linalg.svd(V.T, full_matrices=False)[0][:, :rank]       # orthonormal basis of span{v_j} in R^32
    rng_n = np.random.default_rng(args.cue_noise_seed); N = rng_n.standard_normal(V.shape); N = N - (N @ Q) @ Q.T
    N *= args.cue_noise_ratio * np.linalg.norm(V, axis=1, keepdims=True) / np.linalg.norm(N, axis=1, keepdims=True); Vn = V + N; Vn *= np.linalg.norm(V, axis=1, keepdims=True) / np.linalg.norm(Vn, axis=1, keepdims=True)
    W2[:, CUE_ROWS] = Vn
    print(f"cue-block noise: span{{v_j}} has rank {rank} of 32; |n_j| / |v_j| = {args.cue_noise_ratio}; cue-part norms preserved (max rel. change {np.abs(np.linalg.norm(Vn, axis=1) / np.linalg.norm(V, axis=1) - 1).max():.1e}); cos(v_j + n_j, v_j) = {np.mean(np.sum(Vn * V, 1) / np.linalg.norm(V, axis=1) ** 2):.3f}; max |n_j . v_k| = {np.abs(N @ V.T).max():.1e}", flush=True)
if args.gradient_cue_blocks:                                                  # run 63
    from layer_lenses.designed_init import HIERARCHY_TARGETS, cue_row_identity
    cid = cue_row_identity(W1, tree, CUE_ROWS); W1_ideal = np.zeros_like(W1)
    for p_ in PROBE_ROWS: W1_ideal[p_] = HIERARCHY_TARGETS[p_][1] * U[HIERARCHY_TARGETS[p_][0]] * np.sqrt(2.0)
    for (m_, pole_), r_ in cid.items(): W1_ideal[r_] = pole_ * U[m_] * np.sqrt(2.0)
    Xg = torch.from_numpy(x[:args.gradient_cue_points]).float().cuda(); Yg = torch.from_numpy(y[:args.gradient_cue_points].astype(np.float32)).cuda(); H1 = torch.relu(Xg @ torch.as_tensor(W1_ideal, dtype=torch.float32).cuda().T)
    B_design = W2[:, CUE_ROWS].copy(); new_blocks = np.zeros((60, len(CUE_ROWS)))
    for u_ in range(60):
        w_ = torch.zeros(62); w_[CUE_ROWS] = 0.01; w_[[p_ for p_ in PROBE_ROWS if W2[u_, p_] != 0][0]] = -1.0; w_ = w_.cuda().requires_grad_(True)
        f_ = float(Wout[0, u_]) * torch.relu(H1 @ w_); torch.nn.functional.softplus(-Yg * f_).mean().backward(); d_ = -w_.grad.cpu().numpy()[CUE_ROWS]; new_blocks[u_] = d_ / np.linalg.norm(d_)
    W2[:, CUE_ROWS] = args.gradient_cue_scale * new_blocks
    cos_design = np.sum(new_blocks * B_design, 1) / np.linalg.norm(B_design, axis=1)
    if args.zero_readout: Wout = np.zeros_like(Wout)
    inv = {r_: k for k, r_ in cid.items()}; u0 = 0; j0 = q["unit_node"][u0]
    print(f"gradient cue blocks: cosine with the design blocks mean {cos_design.mean():.3f} (min {cos_design.min():.3f}, max {cos_design.max():.3f}); block norm 1, probe links -1, other probe weights 0", flush=True)
    def side_of_(a_, m_):
        c_ = m_
        while (c_ - 1) // 2 != a_: c_ = (c_ - 1) // 2
        return 1 if c_ == 2 * a_ + 2 else -1
    for u_ in range(4):
        j_ = q["unit_node"][u_]; pr_ = [p_ for p_ in PROBE_ROWS if W2[u_, p_] != 0][0]; qpole = HIERARCHY_TARGETS[pr_][1]
        groups = {"probe side, pole +": [], "probe side, pole -": [], "other side, pole +": [], "other side, pole -": [], "outside, pole +": [], "outside, pole -": []}
        for i_, r_ in enumerate(CUE_ROWS):
            m_, pole_ = inv[r_]; under = j_ in node_ancestors(m_); key = ("outside" if not under else ("probe side" if side_of_(j_, m_) == qpole else "other side")) + f", pole {'+' if pole_ > 0 else '-'}"; groups[key].append(new_blocks[u_, i_])
        print(f"   unit {u_} (node {j_}, probe {pr_} = {'+' if qpole > 0 else '-'}u{j_}, readout {Wout[0, u_]:+.3f}): mean block entry " + ", ".join(f"{k_}: {np.mean(v_):+.3f} (n {len(v_)})" for k_, v_ in groups.items() if v_), flush=True)
if args.symmetric_cue_blocks is not None:                                     # run 67: majority-label cue weights, doubled on the unit's open side under its node
    from layer_lenses.designed_init import HIERARCHY_TARGETS, cue_row_identity
    cid = cue_row_identity(W1, tree, CUE_ROWS); a_open, a_other = args.symmetric_cue_blocks
    def side_of_(a_, m_):
        c_ = m_
        while (c_ - 1) // 2 != a_: c_ = (c_ - 1) // 2
        return 1 if c_ == 2 * a_ + 2 else -1
    for u_ in range(60):
        j_ = int(q["unit_node"][u_]); pr_ = [p_ for p_ in PROBE_ROWS if W2[u_, p_] != 0][0]; open_side = -HIERARCHY_TARGETS[pr_][1]; sgn = np.sign(Wout[0, u_])
        for (m_, pole_), r_ in cid.items(): W2[u_, r_] = sgn * pole_ * (a_open if (j_ in node_ancestors(m_) and side_of_(j_, m_) == open_side) else a_other)
    print(f"symmetric cue blocks a_open {a_open}, a_other {a_other}: block norms by unit level " + ", ".join(f"L{l}: {np.mean([np.linalg.norm(W2[u_, CUE_ROWS]) for u_ in range(60) if NODE_LEVEL[q['unit_node'][u_]] == l]):.3f}" for l in range(4)) + f"; unit 0 cue row: {np.round(W2[0, CUE_ROWS], 2).tolist()}", flush=True)
if args.partner_probe_weight != 0.0:                                          # run 70: each unit reads the other probe of its node with +w
    for u_ in range(60):
        pr_ = [p_ for p_ in PROBE_ROWS if W2[u_, p_] != 0][0]; partner = pr_ + 1 if pr_ % 2 == 0 else pr_ - 1; W2[u_, partner] = args.partner_probe_weight
    print(f"partner-probe weight {args.partner_probe_weight}: unit 0 probe weights {W2[0, :4].tolist()}, unit 1 {W2[1, :4].tolist()}, unit 4 {W2[4, :6].tolist()}; row norms {np.linalg.norm(W2[:4], axis=1).round(3).tolist()}", flush=True)
reports = {j: flip_equivariance_report(W1, W2, _W3, _Wo, tree, j, cue_rows=CUE_ROWS, units_per_node=4) for j in range(15)}
worst = max(max(r.values()) for r in reports.values()); assert args.random_cue_block is not None or args.cue_noise_ratio > 0 or args.gradient_cue_blocks or args.symmetric_cue_blocks is not None or worst < 1e-5, reports
HIDDEN = [62, 60] if args.two_layer else [62, 60, 2]; ARCH = "-".join(map(str, HIDDEN)); model = ReLUMLP(input_dim=100, hidden_dims=HIDDEN, bias=False)
_sd = {"hidden_layers.0.weight": torch.as_tensor(W1, dtype=torch.float32), "hidden_layers.1.weight": torch.as_tensor(W2, dtype=torch.float32), "output_layer.weight": torch.as_tensor(Wout, dtype=torch.float32)}
if not args.two_layer: _sd["hidden_layers.2.weight"] = torch.as_tensor(W3, dtype=torch.float32)
model.load_state_dict(_sd)
if args.project_out:                                                           # runs 57-58: first-layer gradients projected off the chosen normals
    _Up = torch.as_tensor(U[args.project_out], dtype=torch.float32)
    def _project(g):
        Up = _Up.to(g.device); return g - (g @ Up.T) @ Up
    model.hidden_layers[0].weight.register_hook(_project)
lg = logit_scale(model, x, device="cuda")
print(f"{tag}: {ARCH} (run 49's init, cue bias {args.cue_bias}{', W2 negated' if args.negate_w2 else ''}{', no third layer, Wout = path sign x ' + str(args.w3_entry * args.out_weight) if args.two_layer else ''}{', output gradient x ' + str(args.out_scale) if args.out_scale != 1.0 else ''}{', first-layer updates projected off u_' + '/u_'.join(map(str, args.project_out)) if args.project_out else ''}{f', random cue block on node {args.random_cue_block}' if args.random_cue_block is not None else ''}{f', cue-block noise ratio {args.cue_noise_ratio}' if args.cue_noise_ratio > 0 else ''}{f', gradient cue blocks (c = -1, 1/100 cues, ideal W1) x {args.gradient_cue_scale}' if args.gradient_cue_blocks else ''}{', readout zero at init' if args.zero_readout else ''}{f', symmetric cue blocks {args.symmetric_cue_blocks}' if args.symmetric_cue_blocks is not None else ''}{f', partner probe weight {args.partner_probe_weight}' if args.partner_probe_weight else ''}); phases (frozen rows, epochs): {list(zip(PHASE_FROZEN, BOUNDS))}; {N_TRAIN} points, {N_TRAIN // BS} steps per epoch, SGD momentum {MOM} lr {LR} cosine over {EPOCHS} epochs; flip test {worst:.1e}; logit rms {lg['rms_logit']:.4f}", flush=True)
def phase_mask(k):
    mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}
    if PHASE_FROZEN[k] == "cues": mask["hidden_layers.0.weight"][CUE_ROWS] = 0.0
    if PHASE_FROZEN[k] == "probes": mask["hidden_layers.0.weight"][PROBE_ROWS] = 0.0
    if args.out_scale != 1.0: mask["output_layer.weight"][:] = args.out_scale                  # run 56: output gradient x out_scale
    return mask
merged = None; prev = None; cfg = None
for k, (a, b) in enumerate(BOUNDS):
    cfg = ReLUTrainConfig(epochs=EPOCHS, lr=LR, batch_size=BS, seed=1000, device="cuda", snapshot_epochs=tuple(range(a, b + 1, SNAP)), lr_scheduler_eta_min_ratio=ETA_MIN_RATIO,
                          optimizer="sgd", momentum=MOM, parameter_update_mask=phase_mask(k), show_progress=False)
    if prev is not None: model = checkpoint_model_from_state(prev["model"], prev["checkpoint_snapshots"][a])
    out = train_relu_mlp_with_optimizer_snapshots(model=model, x_train=x, y_train=y, config=cfg, start_epoch=a, stop_epoch=b,      # the cosine schedule spans all EPOCHS
                                                  initial_optimizer_states=prev["optimizer_snapshots"][a] if prev is not None else None)
    print(f"phase {k}: epochs {a}-{b} in {(time.time() - t0) / 60:.1f} min; loss at end {np.array(out['epoch_losses'])[b - 1]:.4f}", flush=True)
    if merged is None: merged = out
    else:
        for key in ("checkpoint_snapshots", "optimizer_snapshots", "permutation_generator_snapshots"):
            if isinstance(out.get(key), dict): merged[key].update(out[key])
        for key in ("epoch_losses", "lr_history"):
            L_old, L_new = np.array(merged[key], dtype=float), np.array(out[key], dtype=float); L_old[a:b] = L_new[a:b]; merged[key] = L_old.tolist()
        merged["lr_history_by_group"] = out.get("lr_history_by_group", merged.get("lr_history_by_group")); merged["model"] = out["model"]
    prev = out
out = merged
sr = designed_run_seed_result(out=out, train_config=cfg, model_spec={"input_dim": 100, "hidden_dims": HIDDEN, "bias": False, "activation": "relu"},
                              data_spec=DATA, tree=tree, design={**{k: (list(vv) if isinstance(vv, tuple) else vv) for k, vv in DESIGN.items()}, "scale": "kaiming row norms sqrt(2), readout e / w", "cues": f"pole*(v + {args.cue_bias} u_j) renormalised",
                                                                   "init": "run 49's", "arch": ARCH, "two_layer": args.two_layer, "out_scale": args.out_scale, "schedule_epochs": EPOCHS, "project_out": args.project_out, "random_cue_block": args.random_cue_block, "cue_block_seed": args.cue_block_seed, "cue_noise_ratio": args.cue_noise_ratio, "cue_noise_seed": args.cue_noise_seed, "gradient_cue_blocks": args.gradient_cue_blocks, "gradient_cue_scale": args.gradient_cue_scale, "zero_readout": args.zero_readout, "symmetric_cue_blocks": args.symmetric_cue_blocks, "partner_probe_weight": args.partner_probe_weight, "w3_entry": args.w3_entry, "out_weight": args.out_weight, "n_train": N_TRAIN, "epochs": EPOCHS, "snap": SNAP, "phases": dict(zip(PHASE_FROZEN, BOUNDS)), "cue_bias": args.cue_bias, "negate_w2": args.negate_w2,
                                                                   "orth_seed": args.orth_seed, "data_seed": args.data_seed, "cue_rows": CUE_ROWS, "probe_rows": PROBE_ROWS, "spare_rows": [], "unit_path_sign": q["unit_path_sign"].tolist(), "unit_block_sign": q["unit_block_sign"].tolist(), "unit_node": q["unit_node"].tolist()}, tag=tag)
save_designed_run(sr, f"{OUT}/replay")
ck = out["checkpoint_snapshots"]; epochs = sorted(ck); m_end = out["model"]
tr = evaluate_relu_mlp(model=m_end, x_eval=x, y_eval=y, device="cuda"); te = evaluate_relu_mlp(model=m_end, x_eval=xe, y_eval=ye, device="cuda")
per_epoch = []
for ep in epochs:
    W1e = ck[ep]["hidden_layers.0.weight"].numpy(); t = hierarchy_probe_table(W1e, tree); Ce = W1e / np.linalg.norm(W1e, axis=1, keepdims=True) @ U.T
    per_epoch.append({"epoch": ep, "step": ep * (N_TRAIN // BS), "loss": np.nan if ep == 0 else float(np.mean(out["epoch_losses"][max(0, ep - SNAP):ep])),
                      **{f"on_target_L{l}": float((g.top_node == g.node).mean()) for l, g in t.groupby("level")}, **{f"cos_target_L{l}": float(g.cos_target.mean()) for l, g in t.groupby("level")},
                      "nodes_covered": len(set(np.abs(Ce).argmax(1)[np.abs(Ce).max(1) > 0.95]))})
pe = pd.DataFrame(per_epoch); pe.to_csv(f"{OUT}/per_epoch_{tag}.csv", index=False)
L = np.array(out["epoch_losses"])[:LAST]; first = lambda thr: int(np.nonzero(L <= thr)[0][0] + 1) if (L <= thr).any() else -1
rec = {"tag": tag, "arch": ARCH, "two_layer": args.two_layer, "out_scale": args.out_scale, "schedule_epochs": EPOCHS, "project_out": str(args.project_out), "phases": str(BOUNDS), "n_train": N_TRAIN, "batch": BS, "epochs": EPOCHS, "lr": LR, "logit_rms_init": lg["rms_logit"], "final_epoch_loss": float(L[-1]), "min_epoch_loss": float(np.nanmin(L)),
       "train_log_loss_end": tr["log_loss"], "test_error_end": te["zero_one_loss"], "test_error_strict_end": te["zero_one_loss_strict"], "zero_logit_frac_end": te["zero_logit_fraction"],
       "epoch_to_0.3": first(0.3), "epoch_to_0.2": first(0.2), "epoch_to_0.1": first(0.1), "epoch_to_0.05": first(0.05), "epoch_to_0.03": first(0.03), "epoch_to_0.01": first(0.01),
       **{f"loss_end_phase{k + 1}": float(L[b - 1]) for k, (a, b) in enumerate(BOUNDS)}, "cue_bias": args.cue_bias, "negate_w2": args.negate_w2,
       **{f"on_target_L{l}_end": float(pe[f"on_target_L{l}"].iloc[-1]) for l in range(4)}, "nodes_covered_end": int(pe.nodes_covered.iloc[-1]), "wall_min": (time.time() - t0) / 60}
pd.DataFrame([rec]).to_csv(f"{OUT}/summary.csv", index=False)
json.dump({"arch": ARCH, "two_layer": args.two_layer, "out_scale": args.out_scale, "schedule_epochs": EPOCHS, "project_out": args.project_out, "init": "run 49's", "cue_bias": args.cue_bias, "phases": dict(zip(PHASE_FROZEN, BOUNDS)), "n_train": N_TRAIN, "epochs": EPOCHS, "batch": BS, "lr": LR, "eta_min_ratio": ETA_MIN_RATIO, "momentum": MOM, "snapshot_every_epochs": SNAP,
           "design": {k: (list(vv) if isinstance(vv, tuple) else vv) for k, vv in DESIGN.items()}, "data": DATA}, open(f"{OUT}/manifest.json", "w"), indent=2)
print(f"{tag}: final epoch loss {rec['final_epoch_loss']:.4f} (end of phases: {[round(rec[f'loss_end_phase{k + 1}'], 4) for k in range(len(BOUNDS))]}), test error {te['zero_one_loss']:.4f}; epochs to 0.3/0.2/0.1/0.05/0.03/0.01: {rec['epoch_to_0.3']}/{rec['epoch_to_0.2']}/{rec['epoch_to_0.1']}/{rec['epoch_to_0.05']}/{rec['epoch_to_0.03']}/{rec['epoch_to_0.01']}; "
      f"nodes covered {rec['nodes_covered_end']}/31; on-target end {[round(rec[f'on_target_L{l}_end'], 2) for l in range(4)]}; {rec['wall_min']:.1f} min", flush=True)
print("ALL DONE", flush=True)
