"""Sweeps for studying which first-layer neuron captures the ODT root direction u_0.

Two experiment parts share one runner:

- **Part A (invariance).** A few initializations, each trained several times with
  different data samples and batch sizes, storing the first-layer trajectory. Tests
  the hypothesis that the neuron which captures u_0 is a property of the init alone
  for a fixed optimizer.
- **Part B (init sweep).** Many initializations, one run each, storing the full
  parameter trajectory for lens analyses of what made the winner win.

Unlike `run_single_relu_seed`, the tree seed, data seed, init seed and training seed
are independent knobs, and the ODT is built once and passed to the data generator,
so "same ODT, different sample" is expressible. The learning rate is constant (no
cosine schedule) so runs with different batch sizes share the same flow time per
step, and a run stops as soon as its trailing one-epoch training loss falls below
``stop_loss``. Snapshots are indexed by optimizer step, not epoch, for the same reason.

Continuity with earlier runs: `run_single_relu_seed(master_seed=m)` used
``tree/data seed = m``, ``init seed = m + 1000``, ``train seed = m + 2000``. The same
init is obtained here with ``init_seed = m + 1000``; the data differ because the
tree is now built from its own generator.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from layer_lenses.odt import COBODTTree, build_default_cob_odt_tree, generate_cob_odt_data
from layer_lenses.relu_mlp import ReLUMLP
from layer_lenses.relu_training import (
    ReLUTrainConfig,
    _labels_pm1_to_binary,
    build_relu_optimizers,
    relu_training_step,
)
from layer_lenses.training import set_seed

FIRST_LAYER_KEY = "hidden_layers.0.weight"


@dataclass(frozen=True)
class SweepRunSpec:
    """Everything that determines one sweep run."""

    tree_seed: int
    data_seed: int
    init_seed: int
    batch_size: int
    train_seed: int = 0
    dim: int = 100
    depth: int = 5
    n_train: int = 1_000_000
    hidden_dims: tuple[int, ...] = (40, 40, 40)
    activation: str = "relu"
    lr: float = 0.02
    momentum: float = 0.9
    weight_decay: float = 0.0
    optimizer: str = "sgd"
    stop_loss: float = 0.3
    max_steps: int = 60_000
    snapshot_every_steps: int = 100
    store_all_layers: bool = False
    device: str = "cuda"

    @property
    def run_id(self) -> str:
        return (
            f"tree{self.tree_seed}_data{self.data_seed}_init{self.init_seed}"
            f"_bs{self.batch_size}"
        )

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(self.n_train / self.batch_size)


def build_tree(spec: SweepRunSpec) -> COBODTTree:
    """Build the ODT from ``tree_seed`` alone."""
    rng = np.random.default_rng(spec.tree_seed)
    return build_default_cob_odt_tree(dim=spec.dim, depth=spec.depth, rng=rng)


def build_data(spec: SweepRunSpec, tree: COBODTTree) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n_train`` training points from ``data_seed`` on a fixed tree."""
    x, y, _, _ = generate_cob_odt_data(
        num_data=spec.n_train,
        dim=spec.dim,
        depth=spec.depth,
        seed=spec.data_seed,
        threshold=0.0,
        tree=tree,
    )
    return x, y


def build_data_chunked(
    spec: SweepRunSpec, tree: COBODTTree, *, chunk_size: int = 1_000_000
) -> tuple[np.ndarray, np.ndarray]:
    """Like `build_data` but generated in chunks and stored as float32.

    The generator works in float64 and materialises the margins to all 31 nodes, so
    10M points would need ~18 GB of host memory in one call. Chunk ``k`` uses seed
    ``data_seed + k`` on the same tree; the result is float32 inputs and int8 labels.
    A single chunk reproduces `build_data` exactly (up to the float32 cast).
    """
    n_chunks = math.ceil(spec.n_train / chunk_size)
    x = np.empty((spec.n_train, spec.dim), dtype=np.float32)
    y = np.empty(spec.n_train, dtype=np.int8)
    start = 0
    for k in range(n_chunks):
        n = min(chunk_size, spec.n_train - start)
        xk, yk, _, _ = generate_cob_odt_data(
            num_data=n, dim=spec.dim, depth=spec.depth, seed=spec.data_seed + k, threshold=0.0, tree=tree
        )
        x[start : start + n] = xk.astype(np.float32)
        y[start : start + n] = yk.astype(np.int8)
        start += n
    return x, y


def build_init_model(spec: SweepRunSpec) -> ReLUMLP:
    """Construct the model at initialization from ``init_seed`` alone."""
    set_seed(spec.init_seed)
    return ReLUMLP(
        input_dim=spec.dim,
        hidden_dims=list(spec.hidden_dims),
        bias=False,
        activation=spec.activation,
    )


def _stored_parameter_names(model: ReLUMLP, store_all_layers: bool) -> list[str]:
    names = [name for name, _ in model.named_parameters()]
    return names if store_all_layers else [FIRST_LAYER_KEY]


def train_sweep_run(
    spec: SweepRunSpec,
    *,
    tree: COBODTTree,
    x_train: np.ndarray,
    y_train: np.ndarray,
    log: Callable[[str], None] | None = None,
    model: ReLUMLP | None = None,
    freeze_first_layer_rows: list[int] | None = None,
    unfreeze_at_step: int | None = None,
) -> dict[str, Any]:
    """Train one run to ``stop_loss`` or ``max_steps``, snapshotting every k steps.

    ``model`` overrides the init built from ``spec.init_seed`` (e.g. a designed init);
    ``freeze_first_layer_rows`` keeps those first-layer neurons fixed, until
    ``unfreeze_at_step`` if given, after which every parameter trains.

    Returns a dict with ``spec``, ``snapshot_steps``, ``snapshot_loss`` (trailing
    one-epoch mean training loss at each snapshot, NaN before one epoch has
    elapsed), ``weights`` mapping parameter name to an array of shape
    ``(num_snapshots, *param_shape)``, ``final_step``, ``stopped_by`` and
    ``wall_seconds``. Snapshot 0 is the initialization; the last snapshot is the
    state at ``final_step``.
    """
    t0 = time.time()
    device = torch.device(spec.device)
    model = (build_init_model(spec) if model is None else model).to(device)
    stored = _stored_parameter_names(model, spec.store_all_layers)

    mask = None
    if freeze_first_layer_rows:
        mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}
        mask[FIRST_LAYER_KEY][list(freeze_first_layer_rows)] = 0.0
    set_seed(spec.train_seed)
    config = ReLUTrainConfig(
        epochs=1,
        lr=spec.lr,
        batch_size=spec.batch_size,
        seed=spec.train_seed,
        device=spec.device,
        weight_decay=spec.weight_decay,
        lr_scheduler_eta_min_ratio=1.0,
        optimizer=spec.optimizer,
        momentum=spec.momentum,
        parameter_update_mask=mask,
    )
    optimizers = build_relu_optimizers(model, config)  # constant LR, no scheduler
    from layer_lenses.relu_training import _parameter_update_masks
    update_masks = _parameter_update_masks(model=model, config=config, device=device)
    criterion = torch.nn.BCEWithLogitsLoss()

    x = torch.from_numpy(x_train).float().to(device)
    y_bin = _labels_pm1_to_binary(torch.from_numpy(y_train).float().to(device))
    n = x.shape[0]
    steps_per_epoch = spec.steps_per_epoch
    trailing: deque[tuple[float, int]] = deque(maxlen=steps_per_epoch)

    params = dict(model.named_parameters())
    snapshots: dict[str, list[np.ndarray]] = {name: [] for name in stored}
    snapshot_steps: list[int] = []
    snapshot_loss: list[float] = []

    def take_snapshot(step: int) -> float:
        for name in stored:
            snapshots[name].append(params[name].detach().cpu().numpy().copy())
        snapshot_steps.append(step)
        if len(trailing) == steps_per_epoch:
            num = sum(l for l, _ in trailing)
            den = sum(c for _, c in trailing)
            value = num / den
        else:
            value = float("nan")
        snapshot_loss.append(value)
        return value

    take_snapshot(0)
    step = 0
    stopped_by = "max_steps"
    done = False
    while not done:
        for start in range(0, n, spec.batch_size):
            end = min(start + spec.batch_size, n)
            loss = relu_training_step(
                model=model,
                optimizers=optimizers,
                criterion=criterion,
                x_batch=x[start:end],
                y_batch_bin=y_bin[start:end],
                first_layer_projection_vector=None,
                parameter_update_masks=update_masks,
            )
            step += 1
            if unfreeze_at_step is not None and step == unfreeze_at_step and update_masks:
                update_masks = {}
                if log is not None:
                    log(f"  step {step}: frozen rows released")
            trailing.append((float(loss.cpu()) * (end - start), end - start))
            if step % spec.snapshot_every_steps == 0:
                value = take_snapshot(step)
                if log is not None and step % (spec.snapshot_every_steps * 50) == 0:
                    log(f"  step {step:7d}  trailing loss {value:.4f}")
                if value <= spec.stop_loss:
                    stopped_by = "stop_loss"
                    done = True
                    break
            if step >= spec.max_steps:
                done = True
                break
    if snapshot_steps[-1] != step:
        take_snapshot(step)

    return {
        "spec": spec,
        "snapshot_steps": np.asarray(snapshot_steps, dtype=np.int64),
        "snapshot_loss": np.asarray(snapshot_loss, dtype=np.float64),
        "weights": {name: np.stack(arrs) for name, arrs in snapshots.items()},
        "final_step": int(step),
        "stopped_by": stopped_by,
        "wall_seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_run(result: dict[str, Any], path: str | Path) -> Path:
    """Write one run to a ``.npz`` file (weights, steps, losses, spec as JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "snapshot_steps": result["snapshot_steps"],
        "snapshot_loss": result["snapshot_loss"],
        "final_step": np.asarray(result["final_step"]),
        "stopped_by": np.asarray(result["stopped_by"]),
        "wall_seconds": np.asarray(result["wall_seconds"]),
        "spec_json": np.asarray(json.dumps(asdict(result["spec"]))),
    }
    for name, arr in result["weights"].items():
        payload[f"w__{name}"] = arr
    np.savez(path, **payload)
    return path


def load_run(path: str | Path) -> dict[str, Any]:
    """Inverse of `save_run`."""
    with np.load(path, allow_pickle=False) as z:
        spec_dict = json.loads(str(z["spec_json"]))
        spec_dict["hidden_dims"] = tuple(spec_dict["hidden_dims"])
        return {
            "spec": SweepRunSpec(**spec_dict),
            "snapshot_steps": z["snapshot_steps"],
            "snapshot_loss": z["snapshot_loss"],
            "weights": {k[3:]: z[k] for k in z.files if k.startswith("w__")},
            "final_step": int(z["final_step"]),
            "stopped_by": str(z["stopped_by"]),
            "wall_seconds": float(z["wall_seconds"]),
        }


def save_tree(tree: COBODTTree, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, w_list=tree.w_list, b_list=tree.b_list, leaf_labels=tree.leaf_labels)
    return path


def load_tree(path: str | Path) -> COBODTTree:
    with np.load(path) as z:
        return COBODTTree(w_list=z["w_list"], b_list=z["b_list"], leaf_labels=z["leaf_labels"])


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def signed_cosines_with_direction(first_layer: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Signed cosine of every first-layer row with ``direction``: shape ``(S, K)``."""
    u = np.asarray(direction, dtype=np.float64)
    u = u / np.linalg.norm(u)
    W = np.asarray(first_layer, dtype=np.float64)
    norms = np.linalg.norm(W, axis=-1)
    return (W @ u) / np.maximum(norms, 1e-12)


def capture_summary(
    first_layer: np.ndarray,
    snapshot_steps: np.ndarray,
    direction: np.ndarray,
    *,
    threshold: float = 0.95,
    fanout_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Which neurons capture ``direction`` (either sign), and in what order.

    ``first_layer`` has shape ``(S, K, d)``. A neuron "crosses" at the first
    snapshot where ``|cos| >= threshold``. Returns the first two crossers, their
    crossing steps, the gap between them, the winner's signed cosine at crossing and
    at the end, the winner's fan-out norm at the end if ``fanout_weights`` is given
    (the next layer's weight trajectory, shape ``(S, K_next, K)``; the norm of the
    winner's column), and the set of neurons on the direction at the last snapshot.
    ``winner`` is -1 if nothing crossed.
    """
    cos = signed_cosines_with_direction(first_layer, direction)
    crossed = np.abs(cos) >= threshold
    first_idx = np.where(crossed.any(axis=0), crossed.argmax(axis=0), -1)
    order = [k for k in np.argsort(np.where(first_idx < 0, np.iinfo(np.int64).max, first_idx)) if first_idx[k] >= 0]
    out: dict[str, Any] = {
        "winner": -1, "winner_step": -1, "winner_cos_at_cross": float("nan"),
        "winner_cos_end": float("nan"), "winner_fanout_norm_end": float("nan"),
        "runner_up": -1, "runner_up_step": -1, "gap_steps": -1,
        "n_on_direction_end": int(crossed[-1].sum()),
        "on_direction_end": ",".join(str(int(k)) for k in np.nonzero(crossed[-1])[0]),
        "n_crossed_ever": len(order),
    }
    if order:
        w = int(order[0])
        out.update(
            winner=w,
            winner_step=int(snapshot_steps[first_idx[w]]),
            winner_cos_at_cross=float(cos[first_idx[w], w]),
            winner_cos_end=float(cos[-1, w]),
        )
        if fanout_weights is not None:
            out["winner_fanout_norm_end"] = float(np.linalg.norm(np.asarray(fanout_weights[-1])[:, w]))
        if len(order) > 1:
            r = int(order[1])
            out.update(
                runner_up=r,
                runner_up_step=int(snapshot_steps[first_idx[r]]),
                gap_steps=int(snapshot_steps[first_idx[r]] - snapshot_steps[first_idx[w]]),
            )
    return out


def first_layer_agreement(run_a: dict[str, Any], run_b: dict[str, Any]) -> pd.DataFrame:
    """Row-wise cosine between two runs' first layers at their common snapshot steps.

    Both runs must share the init (same neuron indexing). Columns: ``step``,
    ``median_cos``, ``min_cos``, ``argmin_neuron``, ``n_below_0.99``.
    """
    steps_a = run_a["snapshot_steps"]; steps_b = run_b["snapshot_steps"]
    common, ia, ib = np.intersect1d(steps_a, steps_b, return_indices=True)
    A = run_a["weights"][FIRST_LAYER_KEY][ia].astype(np.float64)
    B = run_b["weights"][FIRST_LAYER_KEY][ib].astype(np.float64)
    cos = (A * B).sum(-1) / np.maximum(np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1), 1e-12)
    return pd.DataFrame(
        {
            "step": common,
            "median_cos": np.median(cos, axis=1),
            "min_cos": cos.min(axis=1),
            "argmin_neuron": cos.argmin(axis=1),
            "n_below_0.99": (cos < 0.99).sum(axis=1),
        }
    )


def _fanout_weights(weights: dict[str, np.ndarray]) -> np.ndarray | None:
    """Trajectory of the layer that reads the first layer: ``(S, K_next, K)`` or None."""
    if "hidden_layers.1.weight" in weights:
        return weights["hidden_layers.1.weight"]
    if "output_layer.weight" in weights:
        return weights["output_layer.weight"]
    return None


def summary_row(result: dict[str, Any], tree: COBODTTree, *, threshold: float = 0.95) -> dict[str, Any]:
    """One CSV row per run: spec fields, stopping info, and the u_0 capture summary."""
    spec = result["spec"]
    row: dict[str, Any] = {
        "run_id": spec.run_id,
        **{k: v for k, v in asdict(spec).items() if k != "hidden_dims"},
        "hidden_dims": "x".join(str(h) for h in spec.hidden_dims),
        "final_step": result["final_step"],
        "final_epoch": result["final_step"] / spec.steps_per_epoch,
        "final_loss": float(result["snapshot_loss"][-1]),
        "stopped_by": result["stopped_by"],
        "wall_seconds": result["wall_seconds"],
        "num_snapshots": int(len(result["snapshot_steps"])),
    }
    row.update(
        capture_summary(
            result["weights"][FIRST_LAYER_KEY],
            result["snapshot_steps"],
            tree.w_list[0],
            threshold=threshold,
            fanout_weights=_fanout_weights(result["weights"]),
        )
    )
    return row


# ---------------------------------------------------------------------------
# Sweep drivers
# ---------------------------------------------------------------------------


def part_a_specs(
    *,
    tree_seed: int,
    init_seeds: list[int],
    data_seeds: list[int],
    batch_sizes: list[int],
    **overrides: Any,
) -> list[SweepRunSpec]:
    """Invariance grid: every init x every data seed x every batch size, first layer only."""
    return [
        SweepRunSpec(
            tree_seed=tree_seed, data_seed=d, init_seed=i, batch_size=b,
            train_seed=i + 1000, store_all_layers=False, **overrides,
        )
        for i in init_seeds
        for d in data_seeds
        for b in batch_sizes
    ]


def part_b_specs(
    *,
    tree_seed: int,
    init_seeds: list[int],
    data_seed: int,
    batch_size: int,
    **overrides: Any,
) -> list[SweepRunSpec]:
    """Init sweep: one run per init, full parameter trajectory stored."""
    return [
        SweepRunSpec(
            tree_seed=tree_seed, data_seed=data_seed, init_seed=i, batch_size=batch_size,
            train_seed=i + 1000, store_all_layers=True, **overrides,
        )
        for i in init_seeds
    ]


def run_specs(
    specs: list[SweepRunSpec],
    out_dir: str | Path,
    *,
    skip_existing: bool = True,
    threshold: float = 0.95,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Run every spec, writing ``runs/<run_id>.npz``, ``tree.npz`` and ``summary.csv``.

    Data is generated once per ``(tree_seed, data_seed)`` and reused. The summary
    CSV is rewritten after every run so progress is visible while the sweep runs.
    Runs whose file already exists are skipped when ``skip_existing`` (resume).
    """
    out_dir = Path(out_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if not specs:
        return pd.DataFrame()
    tree_seeds = {s.tree_seed for s in specs}
    if len(tree_seeds) != 1:
        raise ValueError(f"all specs in one sweep must share tree_seed, got {tree_seeds}")
    tree = build_tree(specs[0])
    save_tree(tree, out_dir / "tree.npz")

    summary_path = out_dir / "summary.csv"
    rows: list[dict[str, Any]] = []
    if summary_path.exists():
        rows = pd.read_csv(summary_path).to_dict("records")
    done_ids = {r["run_id"] for r in rows}

    data_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, spec in enumerate(specs):
        path = runs_dir / f"{spec.run_id}.npz"
        if skip_existing and path.exists() and spec.run_id in done_ids:
            log(f"[{i+1}/{len(specs)}] skip {spec.run_id} (exists)")
            continue
        if spec.data_seed not in data_cache:
            # keep at most one dataset per data seed, drop others to bound memory
            if len(data_cache) >= 3:
                data_cache.pop(next(iter(data_cache)))
            data_cache[spec.data_seed] = build_data(spec, tree)
        x, y = data_cache[spec.data_seed]
        log(f"[{i+1}/{len(specs)}] run {spec.run_id}")
        result = train_sweep_run(spec, tree=tree, x_train=x, y_train=y, log=log)
        save_run(result, path)
        row = summary_row(result, tree, threshold=threshold)
        rows = [r for r in rows if r["run_id"] != spec.run_id] + [row]
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        log(
            f"   -> {result['stopped_by']} at step {result['final_step']} "
            f"(loss {result['snapshot_loss'][-1]:.4f}, {result['wall_seconds']:.0f}s); "
            f"u_0 winner {row['winner']} at step {row['winner_step']}, "
            f"runner-up {row['runner_up']} (+{row['gap_steps']} steps)"
        )
    return pd.DataFrame(rows)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="part", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--out", required=True, help="output directory, e.g. results/init_sweep/partA")
        sp.add_argument("--tree-seed", type=int, default=5178)
        sp.add_argument("--dim", type=int, default=100)
        sp.add_argument("--depth", type=int, default=5)
        sp.add_argument("--n-train", type=int, default=1_000_000)
        sp.add_argument("--hidden-dims", type=int, nargs="+", default=[40, 40, 40])
        sp.add_argument("--lr", type=float, default=0.02)
        sp.add_argument("--momentum", type=float, default=0.9)
        sp.add_argument("--weight-decay", type=float, default=0.0)
        sp.add_argument("--optimizer", default="sgd")
        sp.add_argument("--stop-loss", type=float, default=0.3)
        sp.add_argument("--max-steps", type=int, default=60_000)
        sp.add_argument("--snapshot-every-steps", type=int, default=100)
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--no-skip-existing", action="store_true")

    a = sub.add_parser("part-a", help="invariance grid: inits x data seeds x batch sizes")
    common(a)
    a.add_argument("--init-seeds", type=int, nargs="+", required=True)
    a.add_argument("--data-seeds", type=int, nargs="+", required=True)
    a.add_argument("--batch-sizes", type=int, nargs="+", required=True)

    b = sub.add_parser("part-b", help="init sweep: one run per init, full trajectory")
    common(b)
    b.add_argument("--data-seed", type=int, default=5178)
    b.add_argument("--batch-size", type=int, default=10_000)
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--init-seeds", type=int, nargs="+")
    g.add_argument("--n-inits", type=int, help="use init seeds init-seed-start .. +n-inits-1")
    b.add_argument("--init-seed-start", type=int, default=7000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    overrides = dict(
        dim=args.dim, depth=args.depth, n_train=args.n_train, hidden_dims=tuple(args.hidden_dims),
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, optimizer=args.optimizer,
        stop_loss=args.stop_loss, max_steps=args.max_steps,
        snapshot_every_steps=args.snapshot_every_steps, device=args.device,
    )
    if args.part == "part-a":
        specs = part_a_specs(
            tree_seed=args.tree_seed, init_seeds=args.init_seeds, data_seeds=args.data_seeds,
            batch_sizes=args.batch_sizes, **overrides,
        )
    else:
        init_seeds = args.init_seeds or list(range(args.init_seed_start, args.init_seed_start + args.n_inits))
        specs = part_b_specs(
            tree_seed=args.tree_seed, init_seeds=init_seeds, data_seed=args.data_seed,
            batch_size=args.batch_size, **overrides,
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps({"part": args.part, "args": vars(args), "num_runs": len(specs),
                    "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2)
    )
    df = run_specs(specs, out, skip_existing=not args.no_skip_existing)
    print(df[["run_id", "final_step", "final_loss", "stopped_by", "winner", "winner_step", "runner_up", "gap_steps"]].to_string(index=False))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Post-hoc analysis: leading sets, invariance table, init features, rank tests
# ---------------------------------------------------------------------------


def leading_set(
    first_layer: np.ndarray,
    snapshot_steps: np.ndarray,
    direction: np.ndarray,
    *,
    threshold: float = 0.95,
    tie_window_steps: int = 500,
) -> list[int]:
    """Neurons that cross ``|cos| >= threshold`` within ``tie_window_steps`` of the first.

    Sorted by crossing step. Empty if nothing crosses. This is the tie-aware
    replacement for a single "winner": in practice two neurons often reach u_0
    within a couple of hundred steps of each other.
    """
    cos = signed_cosines_with_direction(first_layer, direction)
    crossed = np.abs(cos) >= threshold
    first_idx = np.where(crossed.any(axis=0), crossed.argmax(axis=0), -1)
    if (first_idx < 0).all():
        return []
    steps = np.asarray(snapshot_steps)
    cross_step = np.where(first_idx >= 0, steps[np.maximum(first_idx, 0)], np.iinfo(np.int64).max)
    earliest = cross_step.min()
    members = np.nonzero(cross_step <= earliest + tie_window_steps)[0]
    return [int(k) for k in members[np.argsort(cross_step[members])]]


def sweep_capture_table(
    out_dir: str | Path,
    *,
    threshold: float = 0.95,
    tie_window_steps: int = 500,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """One row per run file under ``out_dir/runs``: spec, stopping info, u_0 capture, leading set.

    Recomputed from the run files, so it does not depend on ``summary.csv``.
    Unreadable files (e.g. still being written) are skipped.
    """
    out_dir = Path(out_dir)
    tree = load_tree(out_dir / "tree.npz")
    rows = []
    for path in sorted((out_dir / "runs").glob("*.npz")):
        try:
            run = load_run(path)
        except Exception as ex:  # noqa: BLE001 - a partially written file mid-sweep
            if log is not None:
                log(f"skipping {path.name}: {ex}")
            continue
        row = summary_row(run, tree, threshold=threshold)
        lead = leading_set(
            run["weights"][FIRST_LAYER_KEY], run["snapshot_steps"], tree.w_list[0],
            threshold=threshold, tie_window_steps=tie_window_steps,
        )
        row["leading_set"] = ",".join(map(str, lead))
        row["n_leading"] = len(lead)
        row["path"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows)


def part_a_invariance_table(
    out_dir: str | Path,
    *,
    reference_data_seed: int,
    reference_batch_size: int,
    threshold: float = 0.95,
    tie_window_steps: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-init summary of whether the u_0 capture depends on data seed or batch size.

    Returns ``(per_init, per_run)``. ``per_init`` has, for each init seed: the number
    of runs, the distinct first crossers, whether all runs agree on the first
    crosser, whether every run's leading set equals the reference run's leading
    set, whether every leading set at least intersects the reference's, and the
    worst row-wise first-layer cosine against the reference run at the reference
    winner's crossing step and at the end of the shorter run. ``per_run`` has the
    per-run rows with the comparison columns added.
    """
    out_dir = Path(out_dir)
    table = sweep_capture_table(out_dir, threshold=threshold, tie_window_steps=tie_window_steps)
    if table.empty:
        return pd.DataFrame(), table
    runs = {r.run_id: load_run(r.path) for r in table.itertuples()}
    per_run_rows = []
    per_init_rows = []
    for init_seed, grp in table.groupby("init_seed"):
        ref = grp[(grp.data_seed == reference_data_seed) & (grp.batch_size == reference_batch_size)]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        ref_run = runs[ref.run_id]
        ref_lead = set(int(k) for k in ref.leading_set.split(",") if ref.leading_set)
        min_cos_at_cross, min_cos_end = [], []
        for r in grp.itertuples():
            row = r._asdict()
            lead = set(int(k) for k in r.leading_set.split(",") if r.leading_set)
            row["same_winner_as_ref"] = int(r.winner == ref.winner)
            row["leading_set_equals_ref"] = int(lead == ref_lead)
            row["leading_set_meets_ref"] = int(bool(lead & ref_lead))
            agreement = first_layer_agreement(ref_run, runs[r.run_id])
            at_cross = agreement[agreement.step <= max(int(ref.winner_step), 0)]
            row["min_row_cos_vs_ref_at_cross"] = float(at_cross.min_cos.iloc[-1]) if len(at_cross) else float("nan")
            row["min_row_cos_vs_ref_end"] = float(agreement.min_cos.iloc[-1])
            row["median_row_cos_vs_ref_end"] = float(agreement.median_cos.iloc[-1])
            per_run_rows.append(row)
            if r.run_id != ref.run_id:
                min_cos_at_cross.append(row["min_row_cos_vs_ref_at_cross"])
                min_cos_end.append(row["min_row_cos_vs_ref_end"])
        per_init_rows.append(
            {
                "init_seed": int(init_seed),
                "n_runs": int(len(grp)),
                "ref_winner": int(ref.winner),
                "ref_winner_step": int(ref.winner_step),
                "ref_leading_set": ref.leading_set,
                "distinct_winners": ",".join(map(str, sorted(set(int(w) for w in grp.winner)))),
                "all_same_winner": int(grp.winner.nunique() == 1),
                "all_leading_sets_equal_ref": int(all(
                    set(int(k) for k in s.split(",") if s) == ref_lead for s in grp.leading_set)),
                "all_leading_sets_meet_ref": int(all(
                    bool(set(int(k) for k in s.split(",") if s) & ref_lead) for s in grp.leading_set)),
                "worst_min_row_cos_at_cross": float(np.nanmin(min_cos_at_cross)) if min_cos_at_cross else float("nan"),
                "worst_min_row_cos_end": float(np.nanmin(min_cos_end)) if min_cos_end else float("nan"),
            }
        )
    return pd.DataFrame(per_init_rows), pd.DataFrame(per_run_rows)


def _state_dict_at(run: dict[str, Any], snapshot_index: int) -> dict[str, torch.Tensor]:
    return {name: torch.from_numpy(arr[snapshot_index]).float() for name, arr in run["weights"].items()}


def odt_level_ids(depth: int) -> np.ndarray:
    """Level of every internal node id (0 = root) for a complete tree of ``depth``."""
    return np.concatenate([np.full(2**L, L) for L in range(depth)])


def init_neuron_features(
    run: dict[str, Any],
    tree: COBODTTree,
    x_feat: np.ndarray,
    y_feat: np.ndarray,
    *,
    device: str = "cpu",
    plateau_loss: float = 0.69,
    threshold: float = 0.95,
    tie_window_steps: int = 500,
) -> pd.DataFrame:
    """Per-first-layer-neuron features of a full-trajectory run, plus the u_0 outcome.

    Features at initialization (snapshot 0): signed cosine with u_0 and its absolute
    value; weight norm; fan-out norm into the next layer; the neuron's population gradient at init on
    ``x_feat`` (norm, |cos| with the layer's dominant gradient direction, level-4
    energy fraction, signed cosine of -grad with u_0); the neuron's lens x blinder
    over ``x_feat`` (mean, mean of |.|, std). Features at the end of the plateau,
    the first snapshot whose trailing loss is <= ``plateau_loss``: signed cosine with
    u_0, norm ratio to init, displacement norm from init, signed cosine of that
    displacement with u_0, fan-out norm. Outcome columns: ``is_winner`` (first
    crosser), ``in_leading_set``, ``cross_step`` (-1 if never).

    Requires a run that stored all layers.
    """
    from layer_lenses.relu_analysis import (
        activation_tensor_by_layer_neuron_data,
        first_layer_neuron_weighted_lens,
        log_loss_gradients,
    )
    from layer_lenses.relu_training import checkpoint_model_from_state

    spec: SweepRunSpec = run["spec"]
    if "output_layer.weight" not in run["weights"]:
        raise ValueError("init_neuron_features needs a run with store_all_layers=True")
    U = tree.w_list.astype(np.float64)
    u0 = U[0] / np.linalg.norm(U[0])
    levels = odt_level_ids(tree.depth)
    deepest = U[levels == levels.max()]

    W0 = run["weights"][FIRST_LAYER_KEY][0].astype(np.float64)
    fan = _fanout_weights(run["weights"])
    fan0 = np.linalg.norm(fan[0].astype(np.float64), axis=0)
    K = W0.shape[0]
    norms0 = np.linalg.norm(W0, axis=1)

    template = build_init_model(spec)
    model0 = checkpoint_model_from_state(template, _state_dict_at(run, 0))
    G = log_loss_gradients(model=model0, x=x_feat, y=y_feat, device=device)[FIRST_LAYER_KEY].numpy().astype(np.float64)
    gnorm = np.linalg.norm(G, axis=1)
    _, _, vt = np.linalg.svd(G, full_matrices=False)
    common = vt[0]
    g_unit = G / np.maximum(gnorm[:, None], 1e-12)
    acts = activation_tensor_by_layer_neuron_data(model=model0, x_eval=x_feat, device=device)
    state0 = model0.state_dict()
    lens = np.stack([
        first_layer_neuron_weighted_lens(state_dict=state0, activation_tensor=acts, first_layer_neuron_id=k)
        for k in range(K)
    ])

    losses = run["snapshot_loss"]
    finite = np.nonzero(np.isfinite(losses) & (losses <= plateau_loss))[0]
    p = int(finite[0]) if len(finite) else len(losses) - 1
    Wp = run["weights"][FIRST_LAYER_KEY][p].astype(np.float64)
    fanp = np.linalg.norm(fan[p].astype(np.float64), axis=0)
    D = Wp - W0
    dnorm = np.linalg.norm(D, axis=1)

    cos_traj = signed_cosines_with_direction(run["weights"][FIRST_LAYER_KEY], u0)
    crossed = np.abs(cos_traj) >= threshold
    first_idx = np.where(crossed.any(axis=0), crossed.argmax(axis=0), -1)
    cross_step = np.where(first_idx >= 0, run["snapshot_steps"][np.maximum(first_idx, 0)], -1)
    lead = leading_set(run["weights"][FIRST_LAYER_KEY], run["snapshot_steps"], u0,
                       threshold=threshold, tie_window_steps=tie_window_steps)
    winner = lead[0] if lead else -1

    return pd.DataFrame(
        {
            "init_seed": spec.init_seed,
            "neuron": np.arange(K),
            "cos_u0_init": W0 @ u0 / np.maximum(norms0, 1e-12),
            "abs_cos_u0_init": np.abs(W0 @ u0) / np.maximum(norms0, 1e-12),
            "norm_init": norms0,
            "fanout_norm_init": fan0,
            "grad_norm_init": gnorm,
            "grad_abs_cos_common_init": np.abs(g_unit @ common),
            "grad_level4_fraction_init": ((G @ deepest.T) ** 2).sum(1) / np.maximum((G ** 2).sum(1), 1e-30),
            "neg_grad_cos_u0_init": -(G @ u0) / np.maximum(gnorm, 1e-12),
            "lens_mean_init": lens.mean(1),
            "lens_abs_mean_init": np.abs(lens).mean(1),
            "lens_std_init": lens.std(1),
            "plateau_end_step": int(run["snapshot_steps"][p]),
            "cos_u0_plateau": Wp @ u0 / np.maximum(np.linalg.norm(Wp, axis=1), 1e-12),
            "abs_cos_u0_plateau": np.abs(Wp @ u0) / np.maximum(np.linalg.norm(Wp, axis=1), 1e-12),
            "norm_ratio_plateau": np.linalg.norm(Wp, axis=1) / np.maximum(norms0, 1e-12),
            "disp_norm_plateau": dnorm,
            "disp_cos_u0_plateau": (D @ u0) / np.maximum(dnorm, 1e-12),
            "abs_disp_cos_u0_plateau": np.abs(D @ u0) / np.maximum(dnorm, 1e-12),
            "fanout_norm_plateau": fanp,
            "cross_step": cross_step,
            "is_winner": (np.arange(K) == winner).astype(int),
            "in_leading_set": np.isin(np.arange(K), lead).astype(int),
        }
    )


def _two_sided_binomial_p(successes: int, trials: int) -> float:
    """Exact two-sided sign-test p-value at p=1/2."""
    if trials == 0:
        return float("nan")
    tail = sum(math.comb(trials, i) for i in range(0, min(successes, trials - successes) + 1)) / 2**trials
    return float(min(1.0, 2 * tail))


def winner_rank_table(
    features: pd.DataFrame,
    *,
    outcome_col: str = "is_winner",
    exclude: tuple[str, ...] = ("init_seed", "neuron", "cross_step", "is_winner", "in_leading_set", "plateau_end_step"),
) -> pd.DataFrame:
    """Where does the winner rank among its siblings, per feature, pooled over inits.

    ``features`` is the concatenation of `init_neuron_features` frames. For every
    numeric feature column: the winner's percentile rank within its own init (0 =
    smallest, 1 = largest), averaged over inits; the fraction of inits where the
    winner is in the top or bottom decile; and an exact two-sided sign-test p-value
    for "winner above the init median". Sorted by p-value.
    """
    cols = [c for c in features.columns if c not in exclude and pd.api.types.is_numeric_dtype(features[c])]
    rows = []
    for c in cols:
        pct, above = [], 0
        for _, grp in features.groupby("init_seed"):
            if grp[outcome_col].sum() != 1:
                continue
            vals = grp[c].to_numpy(dtype=np.float64)
            w = vals[grp[outcome_col].to_numpy() == 1][0]
            rank = (vals < w).sum() + 0.5 * ((vals == w).sum() - 1)
            pct.append(rank / max(len(vals) - 1, 1))
            above += int(w > np.median(vals))
        if not pct:
            continue
        pct = np.asarray(pct)
        rows.append(
            {
                "feature": c,
                "n_inits": len(pct),
                "mean_percentile": pct.mean(),
                "median_percentile": float(np.median(pct)),
                "frac_top_decile": float((pct >= 0.9).mean()),
                "frac_bottom_decile": float((pct <= 0.1).mean()),
                "frac_above_median": above / len(pct),
                "p_sign_test": _two_sided_binomial_p(above, len(pct)),
            }
        )
    return pd.DataFrame(rows).sort_values("p_sign_test").reset_index(drop=True)


def _standardize_within_init(features: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X[N, K, p], winner[N], in_leading[N, K])`` with z-scores per init."""
    X, W, L = [], [], []
    for _, g in features.groupby("init_seed"):
        if g["is_winner"].sum() != 1:
            continue
        g = g.sort_values("neuron")
        x = g[cols].to_numpy(dtype=np.float64)
        x = (x - x.mean(0)) / (x.std(0) + 1e-12)
        X.append(x)
        W.append(int(np.argmax(g["is_winner"].to_numpy())))
        L.append(g["in_leading_set"].to_numpy().astype(bool))
    return np.stack(X), np.asarray(W), np.stack(L)


def _fit_conditional_logit(X: np.ndarray, W: np.ndarray, *, l2: float, iters: int, lr: float) -> np.ndarray:
    """Maximize the mean log-softmax probability of the winner among its siblings.

    Damped Newton iterations on the L2-regularized conditional-logit objective,
    which is smooth and strictly concave; ``iters`` bounds the iteration count and
    ``lr`` is the initial step, halved on any decrease of the objective.
    """
    N, _, p = X.shape
    beta = np.zeros(p)
    idx = np.arange(N)

    def objective(b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        s = X @ b
        s = s - s.max(axis=1, keepdims=True)
        P = np.exp(s)
        P /= P.sum(axis=1, keepdims=True)
        ll = float(np.log(P[idx, W] + 1e-300).mean() - 0.5 * l2 * (b @ b))
        mean_x = (P[:, :, None] * X).sum(axis=1)                     # (N, p)
        grad = (X[idx, W] - mean_x).mean(axis=0) - l2 * b
        cov = np.einsum("nk,nkp,nkq->pq", P, X, X) / N - (mean_x.T @ mean_x) / N
        hess = -cov - l2 * np.eye(p)
        return ll, grad, hess

    ll, grad, hess = objective(beta)
    step = lr if lr > 0 else 1.0
    for _ in range(min(iters, 100)):
        direction = np.linalg.solve(hess, -grad)
        t = min(1.0, step)
        while True:
            cand = beta + t * direction
            ll_new, grad_new, hess_new = objective(cand)
            if ll_new >= ll - 1e-12 or t < 1e-6:
                break
            t *= 0.5
        converged = abs(ll_new - ll) < 1e-10
        beta, ll, grad, hess = cand, ll_new, grad_new, hess_new
        if converged:
            break
    return beta


def winner_prediction_cv(
    features: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    l2: float = 1e-2,
    iters: int = 3000,
    lr: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Can a linear score over per-neuron features pick the winner out of its siblings?

    For each named feature set, fits a conditional-logit model (softmax over the
    neurons of one init) with leave-one-init-out cross-validation and reports the
    held-out top-1 hit rate, the rate at which the top-scored neuron lies in the
    leading set, the median rank of the true winner, and the top-3 hit rate.
    Chance for top-1 is ``1 / width``. Also returns the weights fitted on all
    inits, per feature set, for interpretation. Features are z-scored within each
    init, so signed quantities should be passed as absolute values when the sign
    is irrelevant.
    """
    rows = []
    weights: dict[str, dict[str, float]] = {}
    for name, cols in feature_sets.items():
        X, W, L = _standardize_within_init(features, cols)
        N = len(W)
        top1 = in_lead = 0
        ranks = []
        for i in range(N):
            m = np.arange(N) != i
            beta = _fit_conditional_logit(X[m], W[m], l2=l2, iters=iters, lr=lr)
            order = np.argsort(-(X[i] @ beta))
            top1 += int(order[0] == W[i])
            in_lead += int(L[i][order[0]])
            ranks.append(int(np.nonzero(order == W[i])[0][0]) + 1)
        ranks = np.asarray(ranks)
        rows.append(
            {
                "feature_set": name, "n_inits": N, "n_features": len(cols),
                "top1": top1 / N, "top1_in_leading_set": in_lead / N,
                "median_winner_rank": float(np.median(ranks)), "top3": float((ranks <= 3).mean()),
                "chance_top1": 1.0 / X.shape[1],
            }
        )
        beta_all = _fit_conditional_logit(X, W, l2=l2, iters=iters, lr=lr)
        weights[name] = {c: float(b) for c, b in zip(cols, beta_all)}
    return pd.DataFrame(rows), weights


# ---------------------------------------------------------------------------
# Autograd sensitivity of the u_0 alignment rate along the population flow
# ---------------------------------------------------------------------------


def alignment_rate_sensitivity(
    model: ReLUMLP,
    x: np.ndarray,
    y: np.ndarray,
    direction: np.ndarray,
    *,
    device: str = "cpu",
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """First and second time derivatives, at the current parameters, of each first-layer
    neuron's alignment with ``direction`` and of its relative norm, under population
    gradient flow on the sample ``(x, y)``.

    For neuron k with weight ``w_k``, unit ``u`` = ``direction``, and population
    gradient ``g_k`` (mean BCE-with-logits over the sample), define

        A_k = d/dt |cos(w_k, u)|  = s_k * <-g_k, u - (w_hat_k.u) w_hat_k> / ||w_k||
        B_k = d/dt log ||w_k||     = <-g_k, w_hat_k> / ||w_k||

    with ``s_k`` the sign of ``w_k.u`` (alignment toward the nearer pole). Both are
    functions of all parameters through ``g_k``. Their derivatives along the flow
    ``theta_dot = -grad L`` are the second time derivatives of alignment and log-norm
    at this point, ``A_dot_k`` and ``B_dot_k``. They are computed exactly by
    forward-mode differentiation over the reverse-mode gradient (``torch.func.jvp``
    over ``torch.func.grad``), with the tangent equal to minus the full-sample
    gradient; the sample is processed in chunks, which is exact because every
    quantity is linear in the per-sample loss.

    Returns one row per first-layer neuron: ``sign``, ``A0``, ``A_dot``, ``B0``,
    ``B_dot``, plus ``grad_norm`` (of ``g_k``) and the scalar ``flow_speed``
    (norm of the full gradient). Requires an activation with forward-mode support
    (the default ``"relu"`` is fine).
    """
    import copy

    from torch.func import functional_call, grad, jvp

    dev = torch.device(device)
    m = copy.deepcopy(model).to(dev).eval()
    params = {n: p.detach() for n, p in m.named_parameters()}
    u = torch.as_tensor(np.asarray(direction, dtype=np.float64), dtype=torch.float32, device=dev)
    u = u / u.norm()

    x_t = torch.from_numpy(x).float().to(dev)
    y_bin = _labels_pm1_to_binary(torch.from_numpy(y).float().to(dev))
    N = x_t.shape[0]
    chunks = [(s, min(s + chunk_size, N)) for s in range(0, N, chunk_size)]

    def loss_fn(p: dict[str, torch.Tensor], xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(functional_call(m, p, (xb,)), yb)

    # full-sample gradient, then the flow direction v = -g
    g = {n: torch.zeros_like(p) for n, p in params.items()}
    for s, e in chunks:
        gc = grad(loss_fn)(params, x_t[s:e], y_bin[s:e])
        for n in g:
            g[n] += gc[n] * ((e - s) / N)
    v = {n: -g[n] for n in g}

    w = params[FIRST_LAYER_KEY]
    sign = torch.sign(w @ u)
    sign[sign == 0] = 1.0

    def rates(p: dict[str, torch.Tensor], xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        gk = grad(loss_fn)(p, xb, yb)[FIRST_LAYER_KEY]
        wk = p[FIRST_LAYER_KEY]
        nrm = wk.norm(dim=1, keepdim=True)
        wh = wk / nrm
        tau = (u[None, :] - (wh @ u)[:, None] * wh) / nrm
        a = sign * (-(gk * tau).sum(1))
        b = (-(gk * wh).sum(1)) / nrm[:, 0]
        return torch.stack([a, b])          # (2, K)

    val = torch.zeros(2, w.shape[0], device=dev)
    dot = torch.zeros(2, w.shape[0], device=dev)
    for s, e in chunks:
        xb, yb = x_t[s:e], y_bin[s:e]
        vc, dc = jvp(lambda p: rates(p, xb, yb), (params,), (v,))
        val += vc * ((e - s) / N)
        dot += dc * ((e - s) / N)

    flow_speed = float(torch.sqrt(sum((t ** 2).sum() for t in g.values())))
    return pd.DataFrame(
        {
            "neuron": np.arange(w.shape[0]),
            "sign": sign.cpu().numpy(),
            "A0": val[0].cpu().numpy(),
            "A_dot": dot[0].cpu().numpy(),
            "B0": val[1].cpu().numpy(),
            "B_dot": dot[1].cpu().numpy(),
            "grad_norm": g[FIRST_LAYER_KEY].norm(dim=1).cpu().numpy(),
            "flow_speed": flow_speed,
        }
    )


def alignment_rate_flow_difference(
    model: ReLUMLP,
    x: np.ndarray,
    y: np.ndarray,
    direction: np.ndarray,
    *,
    rel_step: float = 1e-3,
    device: str = "cpu",
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """Finite-difference version of `alignment_rate_sensitivity` along the flow.

    Evaluates the rates ``A_k`` and ``B_k`` at the parameters displaced by
    ``+-eps * v`` with ``v = -grad L`` and ``eps`` chosen so the displacement has
    norm ``rel_step * ||theta||``, and returns the central difference. Unlike the
    forward-mode derivative, which is exact on the current linear piece of the
    ReLU network, a finite step crosses gate boundaries and therefore includes the
    gate-flip part of the population Hessian. The two agree as ``rel_step -> 0`` on
    a fixed sample; where they differ at the step sizes the flow actually takes,
    the difference is the boundary contribution.

    Columns: ``A0``, ``B0`` (at theta), ``A_dot_fd``, ``B_dot_fd``, ``displacement``.
    Uses the model's parameter dtype, so pass a ``.double()`` model for a
    high-precision small-step check.
    """
    import copy

    from torch.func import functional_call, grad

    dev = torch.device(device)
    m = copy.deepcopy(model).to(dev).eval()
    params = {n: p.detach() for n, p in m.named_parameters()}
    dtype = next(iter(params.values())).dtype
    u = torch.as_tensor(np.asarray(direction, dtype=np.float64), dtype=dtype, device=dev)
    u = u / u.norm()
    x_t = torch.from_numpy(x).to(dtype=dtype, device=dev)
    y_bin = _labels_pm1_to_binary(torch.from_numpy(y).to(dtype=dtype, device=dev))
    N = x_t.shape[0]
    chunks = [(s, min(s + chunk_size, N)) for s in range(0, N, chunk_size)]

    def loss_fn(p, xb, yb):
        return torch.nn.functional.binary_cross_entropy_with_logits(functional_call(m, p, (xb,)), yb)

    def full_grad(p):
        g = {n: torch.zeros_like(t) for n, t in p.items()}
        for s, e in chunks:
            gc = grad(loss_fn)(p, x_t[s:e], y_bin[s:e])
            for n in g:
                g[n] += gc[n] * ((e - s) / N)
        return g

    w0 = params[FIRST_LAYER_KEY]
    sign = torch.sign(w0 @ u)
    sign[sign == 0] = 1.0

    def rates(p, g):
        gk = g[FIRST_LAYER_KEY]
        wk = p[FIRST_LAYER_KEY]
        nrm = wk.norm(dim=1, keepdim=True)
        wh = wk / nrm
        tau = (u[None, :] - (wh @ u)[:, None] * wh) / nrm
        return sign * (-(gk * tau).sum(1)), (-(gk * wh).sum(1)) / nrm[:, 0]

    g0 = full_grad(params)
    a0, b0 = rates(params, g0)
    gnorm = torch.sqrt(sum((t ** 2).sum() for t in g0.values()))
    tnorm = torch.sqrt(sum((t ** 2).sum() for t in params.values()))
    eps = float(rel_step * tnorm / gnorm)
    plus = {n: params[n] - eps * g0[n] for n in params}     # theta + eps * v
    minus = {n: params[n] + eps * g0[n] for n in params}    # theta - eps * v
    ap, bp = rates(plus, full_grad(plus))
    am, bm = rates(minus, full_grad(minus))
    return pd.DataFrame(
        {
            "neuron": np.arange(w0.shape[0]),
            "A0": a0.cpu().numpy(),
            "B0": b0.cpu().numpy(),
            "A_dot_fd": ((ap - am) / (2 * eps)).cpu().numpy(),
            "B_dot_fd": ((bp - bm) / (2 * eps)).cpu().numpy(),
            "displacement": float(eps * gnorm),
        }
    )
