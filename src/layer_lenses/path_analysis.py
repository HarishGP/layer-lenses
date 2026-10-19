"""Path statistics for a two-hidden-layer ReLU net on the depth-5 COB-ODT task.

A path is (first-layer row r, unit u): on the points where both are active it contributes Wout[u] W2[u, r] (w_r . x) to the logit, a linear
function of x restricted to the path's active region. For every one of the 62 x 60 paths this module measures, on a sample: the active
fraction, the leaf-cell histogram of its active points (which of the 32 (level-4 node, label sign) cells they fall in, summarised by the
share of the largest cell and an entropy), the path weight |Wout[u] W2[u, r]| ||w_r||, the mean and rms contribution on the active set,
and the correlation of the contribution with the label there. Rows and units are labelled by their design identities so that the paths
can be grouped (probe -> own unit, cue -> unit of an ancestor, ...).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from layer_lenses.designed_init import HIERARCHY_TARGETS, NODE_LEVEL, node_ancestors


def leaf_cells(x: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Cell index 0..31 of every point: 2 x (level-4 node - 15) + (1 if its label is +1 else 0)."""
    A = x @ U.T; n = np.zeros(len(A), int)
    for _ in range(4):
        n = 2 * n + 1 + (A[np.arange(len(A)), n] > 0)
    return 2 * (n - 15) + (A[np.arange(len(A)), n] > 0).astype(int)


def path_relation(row_node: int, unit_node: int) -> str:
    """Tree relation of a row's node to a unit's node: own, ancestor (the row's node is above the unit's), descendant, or unrelated."""
    if row_node == unit_node:
        return "own"
    if row_node in node_ancestors(unit_node):
        return "ancestor"
    if unit_node in node_ancestors(row_node):
        return "descendant"
    return "unrelated"


def path_statistics(W1: np.ndarray, W2: np.ndarray, Wout: np.ndarray, x: np.ndarray, y: np.ndarray, U: np.ndarray, *, row_node: np.ndarray, row_pole: np.ndarray, unit_node: np.ndarray, cells: np.ndarray | None = None, min_active: int = 20) -> pd.DataFrame:
    """One row per path (unit u, row r) with its active-region statistics on the sample (x, y).

    row_node / row_pole: the node and pole each first-layer row is closest to (or its design identity); unit_node: each unit's node.
    """
    cells = leaf_cells(x, U) if cells is None else cells
    pre1 = x @ W1.T; h1 = np.maximum(pre1, 0); g1 = pre1 > 0; h2 = np.maximum(h1 @ W2.T, 0); g2 = h2 > 0; wo = Wout.ravel(); nrm = np.linalg.norm(W1, axis=1)
    rows = []
    for u in range(W2.shape[0]):
        gu = g2[:, u]
        for r in range(W1.shape[0]):
            act = gu & g1[:, r]; n = int(act.sum()); w = wo[u] * W2[u, r]
            rec = {"unit": u, "row": r, "unit_node": int(unit_node[u]), "unit_level": int(NODE_LEVEL[unit_node[u]]), "row_node": int(row_node[r]), "row_pole": int(row_pole[r]), "row_kind": "probe" if r < 30 else "cue",
                   "relation": path_relation(int(row_node[r]), int(unit_node[u])), "W2": float(W2[u, r]), "readout": float(wo[u]), "path_weight": float(abs(w) * nrm[r]), "signed_path_weight": float(w * nrm[r]), "active_frac": n / len(x)}
            if n >= min_active:
                c = w * pre1[act, r]; cnt = np.bincount(cells[act], minlength=32) / n; p = cnt[cnt > 0]
                rec.update({"contrib_mean": float(c.mean()), "contrib_rms": float(np.sqrt((c ** 2).mean())), "corr_label": float(np.corrcoef(c, y[act])[0, 1]) if c.std() > 0 and y[act].std() > 0 else 0.0,
                            "top_cell_share": float(cnt.max()), "top_cell": int(cnt.argmax()), "cell_entropy_bits": float(-(p * np.log2(p)).sum()), "label_mean_active": float(y[act].mean())})
            else:
                rec.update({"contrib_mean": np.nan, "contrib_rms": np.nan, "corr_label": np.nan, "top_cell_share": np.nan, "top_cell": -1, "cell_entropy_bits": np.nan, "label_mean_active": np.nan})
            rows.append(rec)
    return pd.DataFrame(rows)


def path_group_summary(ps: pd.DataFrame) -> pd.DataFrame:
    """Mean statistics by (row kind, relation, unit level) over the paths with enough active points, plus the share of the total |path weight|."""
    g = ps.groupby(["row_kind", "relation", "unit_level"])
    out = g.agg(n_paths=("row", "size"), active_frac=("active_frac", "mean"), path_weight=("path_weight", "mean"), contrib_rms=("contrib_rms", "mean"), corr_label=("corr_label", "mean"), top_cell_share=("top_cell_share", "mean"), entropy_bits=("cell_entropy_bits", "mean"))
    out["weight_share"] = g["path_weight"].sum() / ps["path_weight"].sum()
    return out.round(3)
