"""The alpha parametrisation of the two-hidden-layer designed inits (runs 60, 68, 69, 70 of notebook 2).

Thirteen signed scalars describe the init and the per-layer learning-rate scales:
  alpha_1  norm of every probe row (30 rows, purely off-span, mutually orthogonal, one per (internal node, pole) in HIERARCHY_TARGETS order)
  alpha_2  norm of every cue row (32 rows, one per (leaf, pole), in-span: pole x (sum of the 16 leaf normals + bias x u_m), renormalised)
  alpha_3  ratio of the cue's component on its own leaf normal to its component on each other leaf normal (= 1 + bias)
  alpha_4  cue weight of a unit on the +cue of a leaf under the unit's node on the unit's open side (the -cue gets the negated weight)
  alpha_5  the same for leaves under the unit's node on the vetoed side
  alpha_6  the same for leaves outside the unit's node's subtree
           alpha_4..6 may be scalars or 4-vectors indexed by the unit's node level (run 60's design equalises the block norm per level)
  alpha_7  weight from a probe to its own units (the veto; -1 in every run)
  alpha_8  weight from a probe to the units of the other probe of its node (the pole partner; 0 or +1)
  alpha_9  readout magnitude of the units of levels 0-2 (units 0-27); alpha_10 the same for level 3 (units 28-59); signs = path signs
  alpha_11 learning-rate scale of the readout relative to W1; alpha_12 the same for W2 (used as parameter_update_mask values)
  alpha_13 in-span fraction of every cue row: cue = alpha_2 (alpha_13 w_in + sqrt(1 - alpha_13^2) o_r) with w_in the unit in-span cue above and
           o_r the row's own off-span direction; the 62 off-span directions (30 probes, 32 cues) are orthonormal (62 < 69 = 100 - 31).
           alpha_13 = 1 is the purely in-span cue of every run up to 106; a random unit vector has in-span fraction sqrt(16 / 100) = 0.4
           on the 16 leaf normals (the cues' in-span part lives in their span), which is the "no tree knowledge" base value.
Sign convention for the cue weights: W2[u, cue(m, pole)] = s_u x pole x alpha, with s_u the unit's readout (path) sign. Run 60's design
(blocks +-B_j shared by the (B, +) and (B, -) units) is alpha_4 = +c, alpha_5 = -c, alpha_6 = 0 in this convention; the majority-label
blocks of runs 68-70 are alpha_4 = 0.2, alpha_5 = 0.1, alpha_6 = 0.1.
The four units of node j are, in order, the types (B, +), (B, -), (-B, -), (-B, +) of the quadruple design: path signs (+, -, -, +), and
the unit reads probe 2j (pole +) when block sign x path sign > 0, else probe 2j + 1 (pole -); its open side is the side opposite to the
pole it vetoes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from layer_lenses.designed_init import HIERARCHY_TARGETS, NODE_LEVEL, node_ancestors

QUAD_TYPES = ((1, 1), (1, -1), (-1, -1), (-1, 1))      # (block sign, path sign) of the four units of a node
LEAVES = list(range(15, 31))


def side_of(node: int, leaf: int) -> int:
    """Side (+1: a_node > 0, -1: a_node < 0) of ``node`` on which the descendant ``leaf`` lies."""
    c = leaf
    while (c - 1) // 2 != node:
        c = (c - 1) // 2
    return 1 if c == 2 * node + 2 else -1


@dataclass
class AlphaInit:
    alpha_1: float = np.sqrt(2.0)
    alpha_2: float = np.sqrt(2.0)
    alpha_3: float = 2.0
    alpha_4: float | tuple = 0.2
    alpha_5: float | tuple = 0.1
    alpha_6: float | tuple = 0.1
    alpha_7: float = -1.0
    alpha_8: float = 0.0
    alpha_9: float = 0.8165
    alpha_10: float = 0.8165
    alpha_11: float = 1.0
    alpha_12: float = 1.0
    alpha_13: float = 1.0


def _by_level(a, level: int) -> float:
    return float(a[level]) if isinstance(a, (tuple, list, np.ndarray)) else float(a)


def build_alpha_init(alphas: AlphaInit, tree, *, orth_seed: int = 0, dim: int = 100) -> dict:
    """Full parameters (W1 62x100, W2 60x62, Wout 1x60) and learning-rate masks from the thirteen alphas.

    Rows 0-29 are the probes in HIERARCHY_TARGETS order, rows 30-61 the cues in (leaf, pole) order (-u_15, +u_15, -u_16, ...), as in the
    runs; units 4j..4j+3 belong to node j. ``orth_seed`` draws the 62 orthonormal off-span directions (``off_span_directions``, one per
    row; the probes' 30 do not depend on the cues' being drawn, so alpha_13 = 1 reproduces the earlier 30-direction inits exactly).
    """
    U = tree.w_list; d = dim
    # first layer
    rng = np.random.default_rng(orth_seed); G = rng.standard_normal((62, d)); G -= (G @ U.T) @ U
    Q, _ = np.linalg.qr(G.T); O = Q.T[:62]                                                  # 62 orthonormal off-span directions
    W1 = np.zeros((62, d)); cue_rows = {}
    for p in range(30):
        W1[p] = alphas.alpha_1 * O[p]
    v = U[15:31].sum(0); r = 30; a13 = float(alphas.alpha_13); off = np.sqrt(max(0.0, 1.0 - a13 ** 2))
    for m in LEAVES:
        for pole in (-1, 1):
            w = v + (alphas.alpha_3 - 1.0) * U[m]; W1[r] = alphas.alpha_2 * (a13 * pole * w / np.linalg.norm(w) + off * O[r]); cue_rows[(m, pole)] = r; r += 1
    # second layer
    W2 = np.zeros((60, 62)); Wout = np.zeros((1, 60)); units = []
    for j in range(15):
        for t, (b, s) in enumerate(QUAD_TYPES):
            u = 4 * j + t; probe = 2 * j if b * s > 0 else 2 * j + 1; veto_pole = HIERARCHY_TARGETS[probe][1]; open_side = -veto_pole
            partner = probe + 1 if probe % 2 == 0 else probe - 1
            W2[u, probe] = alphas.alpha_7; W2[u, partner] = alphas.alpha_8
            lvl = NODE_LEVEL[j]
            for m in LEAVES:
                if j in node_ancestors(m):
                    a = _by_level(alphas.alpha_4 if side_of(j, m) == open_side else alphas.alpha_5, lvl)
                else:
                    a = _by_level(alphas.alpha_6, lvl)
                for pole in (-1, 1):
                    W2[u, cue_rows[(m, pole)]] = s * pole * a
            Wout[0, u] = s * (alphas.alpha_9 if lvl < 3 else alphas.alpha_10); units.append({"unit": u, "node": j, "block_sign": b, "path_sign": s, "probe": probe})
    masks = {"hidden_layers.0.weight": np.ones((62, d), dtype=np.float32), "hidden_layers.1.weight": np.full((60, 62), alphas.alpha_12, dtype=np.float32), "output_layer.weight": np.full((1, 60), alphas.alpha_11, dtype=np.float32)}
    return {"W1": W1, "W2": W2, "Wout": Wout, "update_masks": masks, "cue_rows": cue_rows, "probe_rows": list(range(30)), "units": units, "off_span_directions": O}


RUN_ALPHAS = {
    60: AlphaInit(alpha_4=(2 ** -2.5, 2 ** -2, 2 ** -1.5, 2 ** -1), alpha_5=(-(2 ** -2.5), -(2 ** -2), -(2 ** -1.5), -(2 ** -1)), alpha_6=0.0, alpha_8=0.0, alpha_11=80.0),
    68: AlphaInit(alpha_4=0.2, alpha_5=0.1, alpha_6=0.1, alpha_8=0.0, alpha_11=1.0),
    69: AlphaInit(alpha_4=0.1, alpha_5=0.2, alpha_6=0.2, alpha_8=0.0, alpha_11=1.0),
    70: AlphaInit(alpha_4=0.2, alpha_5=0.1, alpha_6=0.1, alpha_8=1.0, alpha_11=1.0),
}
"""The settings of runs 60, 68, 69 and 70 (alpha_1 = alpha_2 = sqrt 2, alpha_3 = 2, alpha_7 = -1, alpha_9 = alpha_10 = 0.8165, alpha_12 = alpha_13 = 1 in all)."""
