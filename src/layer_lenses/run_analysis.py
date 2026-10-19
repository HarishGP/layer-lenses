"""Run-level analysis helpers shared by the notebooks: loading a replayable designed-init run, end-of-run numbers, trajectory plots
(loss / test error / node coverage, probe cosines, alignment series of chosen first-layer neurons) and the layered network view at an
epoch. Extracted from the setup cell of notebooks/hierarchy_design_runs_2.ipynb so that the anatomy notebook can use them on any run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from layer_lenses.designed_init import HIERARCHY_TARGETS, hierarchy_run_summary, plot_probe_target_cosines
from layer_lenses.init_sweep import SweepRunSpec, build_data, build_tree
from layer_lenses.relu_analysis import (first_layer_alignment_series, first_layer_odt_alignment, load_designed_run, plot_first_layer_alignment_series,
                                         plot_layered_relu_graph, snapshot_state_dict)
from layer_lenses.relu_mlp import ReLUMLP


@dataclass
class AnalysisContext:
    """The tree and the fixed test sample every helper uses (tree seed 5178, test seed 999777, 100k points by default)."""
    results: Path
    tree: object = None
    x_test: np.ndarray = None
    y_test: np.ndarray = None
    n_test: int = 100_000

    def __post_init__(self):
        self.results = Path(self.results)
        if self.tree is None:
            self.tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1))
        if self.x_test is None:
            self.x_test, self.y_test = build_data(SweepRunSpec(tree_seed=5178, data_seed=999_777, init_seed=0, batch_size=1, n_train=self.n_test), self.tree)
        self.X_test = torch.from_numpy(self.x_test).float()


def load_replay_run(ctx: AnalysisContext, folder: str, dims: list[int]) -> tuple[dict, dict]:
    """(seed_result, run dict) of a run written by save_designed_run: all layers at every stored epoch, losses, lr, per-epoch table, summary row."""
    sr = load_designed_run(ctx.results / folder / "replay"); out = sr["out"]; ep = np.array(sorted(out["checkpoint_snapshots"]))
    losses = np.asarray(out["epoch_losses"], dtype=float); losses = losses[~np.isnan(losses)]
    run = {"tag": sr["tag"], "dims": dims, "epochs": ep, "losses": losses, "lr": np.asarray(out["lr_history"], dtype=float),
           "W": {key: np.stack([out["checkpoint_snapshots"][e][key].numpy() for e in ep]) for key in out["checkpoint_snapshots"][ep[0]]}}
    per = ctx.results / folder / f"per_epoch_{sr['tag']}.csv"; summ = ctx.results / folder / "summary.csv"
    run["per_epoch"] = pd.read_csv(per) if per.exists() else None; run["summary"] = pd.read_csv(summ).iloc[0] if summ.exists() else None
    return sr, run


def logits_at(ctx: AnalysisContext, run: dict, epoch: int) -> np.ndarray:
    m = ReLUMLP(input_dim=100, hidden_dims=run["dims"], bias=False); m.load_state_dict(snapshot_state_dict({"epochs": run["epochs"], **{f"w__{k}": v for k, v in run["W"].items()}}, epoch)); m.eval()
    with torch.no_grad():
        return m(ctx.X_test).reshape(-1).numpy()


def findings_table(ctx: AnalysisContext, run: dict) -> tuple[pd.Series, dict]:
    """End-of-run numbers (loss, both test-error rules, zero-logit fraction, coverage, probe outcomes) and the hierarchy summary dict."""
    out = logits_at(ctx, run, int(run["epochs"][-1])); W1 = run["W"]["hidden_layers.0.weight"]; y = ctx.y_test
    s = hierarchy_run_summary(W1[-1], ctx.tree); L = run["losses"]; first = lambda t: int(np.nonzero(L <= t)[0][0] + 1) if (L <= t).any() else None
    rec = {"final epoch loss": round(float(L[-1]), 4), "test error (logit >= 0)": round(float(((out >= 0) != (y > 0)).mean()), 4), "test error (strict)": round(float(((out > 0) != (y > 0)).mean()), 4),
           "zero-logit fraction": round(float((out == 0).mean()), 3), "negative output weights": int((run["W"]["output_layer.weight"][-1] < 0).sum()),
           "epochs to 0.3 / 0.1 / 0.05": f"{first(0.3)} / {first(0.1)} / {first(0.05)}", "aligned rows": f"{s['aligned_rows']} / {s['rows']}", "nodes covered": f"{s['nodes_covered']} / 31, missing {s['missing_nodes']}",
           "own-probe nodes / intended / opposite": f"{s['own_probe_nodes']} / {s['intended_pole']} / {s['opposite_pole_only']}", "missed by own probes": s["missed_by_own_probes"], "both poles": s["both_poles"]}
    return pd.Series(rec, name=run["tag"]), s


def trajectory_plots(ctx: AnalysisContext, run: dict, neuron_ids: list[int], node_ids: list[int] | None = None, start_epoch: int | None = None, end_epoch: int | None = None, stride: int = 1):
    """Loss / test error / coverage over epochs; every probe's cosine to its target; alignment series of the chosen neurons.

    ``stride`` thins the stored epochs for the test-error and coverage curves (a forward pass per epoch).
    """
    tree, y = ctx.tree, ctx.y_test; ep_all = run["epochs"]; lo = ep_all[0] if start_epoch is None else start_epoch; hi = ep_all[-1] if end_epoch is None else end_epoch
    sel = (ep_all >= lo) & (ep_all <= hi)
    if not sel.any():
        raise ValueError(f"no snapshots in epochs [{lo}, {hi}]; stored epochs run {ep_all[0]}..{ep_all[-1]}")
    ep = ep_all[sel]; W1 = run["W"]["hidden_layers.0.weight"][sel]; W2 = run["W"]["hidden_layers.1.weight"][sel]
    loss_epochs = np.arange(1, len(run["losses"]) + 1); lsel = (loss_epochs >= max(lo, 1)) & (loss_epochs <= hi)
    ks = list(range(0, len(ep), stride)) + ([len(ep) - 1] if (len(ep) - 1) % stride else [])
    cov = [hierarchy_run_summary(W1[k], tree)["nodes_covered"] for k in ks]; err = [float(((logits_at(ctx, run, int(ep[k])) >= 0) != (y > 0)).mean()) for k in ks]
    window = "" if start_epoch is None and end_epoch is None else f" (epochs {lo}-{hi})"
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
    axes[0].plot(loss_epochs[lsel], np.asarray(run["losses"])[lsel]); axes[0].set_yscale("log"); axes[0].set_ylabel("epoch train loss")
    axes[1].plot(ep[ks], err); axes[1].set_yscale("log"); axes[1].set_ylabel("test error (logit >= 0)")
    axes[2].plot(ep[ks], cov); axes[2].axhline(31, color="k", ls=":", lw=0.8); axes[2].set_ylabel("nodes covered (|cos| > 0.95)")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.25)
    fig.suptitle(f"{run['tag']}{window}", fontsize=9); fig.tight_layout(); plt.show()
    plot_probe_target_cosines(W1, ep, tree, title=f"{run['tag']}{window}: signed cosine of each probe to its designed target"); plt.show()
    if node_ids is None:
        W_end = W1[-1]; C = (W_end / np.linalg.norm(W_end, axis=1, keepdims=True)) @ tree.w_list.T
        node_ids = sorted({HIERARCHY_TARGETS[i][0] for i in neuron_ids if i < 30} | {int(np.abs(C[i]).argmax()) for i in neuron_ids})
    series = first_layer_alignment_series(W1, ep, tree, neuron_ids=neuron_ids, node_ids=node_ids, fan_out_weights=W2)
    plot_first_layer_alignment_series(series, title=f"{run['tag']}{window}: first-layer neurons {neuron_ids} vs ODT nodes {node_ids}"); plt.show()
    return series


def network_view(ctx: AnalysisContext, run: dict, epoch: int, *, align_threshold: float = 0.5, edge_threshold: float = 0.5, only_aligned_first_layer: bool = False, figsize=(14, 18), mask_arg=None, layer_orders=None) -> pd.DataFrame:
    """Layered graph at one epoch (first-layer rows labelled by their closest ODT node) and the row -> node table (rows aligned at >= align_threshold)."""
    tree = ctx.tree; sd = snapshot_state_dict({"epochs": run["epochs"], **{f"w__{k}": v for k, v in run["W"].items()}}, epoch)
    n_hidden = len(run["dims"]); mask = None
    if only_aligned_first_layer:
        al = first_layer_odt_alignment(state_dict=sd, tree=tree)
        mask = [al["max_abs_cos"].numpy() >= align_threshold] + [np.ones(sd[f"hidden_layers.{i}.weight"].shape[0], dtype=bool) for i in range(1, n_hidden)]
    if mask_arg is not None:
        mask = mask_arg
    alignment, _, fig, ax = plot_layered_relu_graph(state_dict=sd, tree=tree, depth=5, epoch=epoch, node_mask=mask, alignment_abs_threshold=align_threshold, edge_abs_threshold=edge_threshold, figsize=figsize, layer_orders=layer_orders)
    plt.show()
    n = sd["hidden_layers.0.weight"].shape[0]
    tab = pd.DataFrame({"row": np.arange(n), "group": ["probe"] * 30 + ["cue"] * 32 + ["random"] * (n - 62), "designed_target": [HIERARCHY_TARGETS[i] if i < 30 else "" for i in range(n)],
                        "closest_node": alignment["closest_odt_np"], "level": alignment["node_level"], "signed_cos": alignment["signed_closest_cos"].numpy().round(3), "norm": alignment["row_norms"].numpy().round(2)})
    return tab[tab.signed_cos.abs() >= align_threshold].sort_values(["level", "closest_node", "signed_cos"]).reset_index(drop=True)
