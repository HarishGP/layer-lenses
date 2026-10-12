"""Hand-designed initializations that should steer one first-layer neuron toward u_0.

The construction (Harish, 2026-09-28) for a ``[40, 40, 40]`` ReLU MLP on a depth-5
COB-ODT with internal nodes ``u_0 .. u_30`` (level 4 is ``u_15 .. u_30``; nodes
``15..22`` lie under the root's left child, ``23..30`` under its right child):

- **First layer.** Neuron 0 ("the probe") is a small random vector orthogonal to
  every ODT normal. Neurons 1..32 are ``-u_15, +u_15, -u_16, +u_16, ..., -u_30,
  +u_30``. Neurons 33..39 are random. Every row has the same norm ``eps``.
- **Second layer**, entries in ``{-1, 0, +1}``, four groups of ``group_size`` units:

  | group | rows   | weight on the probe | +1 on a subset of first-layer neurons |
  |-------|--------|---------------------|----------------------------------------|
  | A     | 0-9    | +1                  | ``-u_15 .. -u_22``                      |
  | B     | 10-19  | +1                  | ``+u_23 .. +u_30``                      |
  | C     | 20-29  | -1                  | ``-u_23 .. -u_30``                      |
  | D     | 30-39  | -1                  | ``+u_15 .. +u_22``                      |

  The probe weights can be flipped per group (``group_sign``) to test variations.
- **Third layer**: each row a random vector in ``{0, 1}^40``.
- **Output**: the all-ones vector times ``output_scale``.

The label of a point is ``+1`` iff its on-path level-4 split is negative, so the
gates of the level-4 neurons correlate with ``y`` only on the side of the root
that owns them. Through the second-layer wiring that correlation reaches the
probe's lens, which is how a ``u_0`` component can enter the probe's gradient
even though ``E[y x]`` has none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from layer_lenses.odt import COBODTTree
from layer_lenses.relu_mlp import ReLUMLP
from layer_lenses.relu_training import ReLUTrainConfig, train_relu_mlp

PROBE = 0
LEVEL4_FIRST = 15
NUM_LEVEL4 = 16
NUM_RANDOM = 7
WIDTH = 1 + 2 * NUM_LEVEL4 + NUM_RANDOM  # 40


@dataclass(frozen=True)
class DesignedInitSpec:
    """Knobs of the designed initialization."""

    eps: float = 1e-3
    seed: int = 0
    # "random": each candidate joins a unit's subset with prob 1/2 (at least one);
    # "all": every candidate joins every unit of its group.
    subset_mode: str = "random"
    # Sign of the probe's weight into groups A, B, C, D. Base case (+1, +1, -1, -1).
    group_sign: tuple[int, int, int, int] = (1, 1, -1, -1)
    group_size: int = 10
    layer3_density: float = 0.5
    output_scale: float = 1.0
    dim: int = 100
    depth: int = 5


def first_layer_index(node: int, sign: int) -> int:
    """Row index of the first-layer neuron ``sign * u_node`` (node in 15..30)."""
    if not LEVEL4_FIRST <= node < LEVEL4_FIRST + NUM_LEVEL4 or sign not in (-1, 1):
        raise ValueError(f"need a level-4 node and sign +-1, got node={node}, sign={sign}")
    return 1 + 2 * (node - LEVEL4_FIRST) + (1 if sign > 0 else 0)


def first_layer_labels() -> list[str]:
    labels = ["probe"]
    for node in range(LEVEL4_FIRST, LEVEL4_FIRST + NUM_LEVEL4):
        labels += [f"-u{node}", f"+u{node}"]
    labels += [f"rand{i}" for i in range(NUM_RANDOM)]
    return labels


GROUP_CANDIDATES = {
    "A": [(-1, n) for n in range(15, 23)],
    "B": [(+1, n) for n in range(23, 31)],
    "C": [(-1, n) for n in range(23, 31)],
    "D": [(+1, n) for n in range(15, 23)],
}
GROUP_ORDER = ("A", "B", "C", "D")


def build_designed_weights(spec: DesignedInitSpec, tree: COBODTTree) -> dict[str, np.ndarray]:
    """Return the four weight matrices of the designed init as float64 arrays."""
    if spec.depth != 5 or tree.w_list.shape != (31, spec.dim):
        raise ValueError("the design assumes a depth-5 ODT with 31 internal nodes")
    if spec.group_size * 4 != WIDTH:
        raise ValueError(f"4 * group_size must equal {WIDTH}")
    rng = np.random.default_rng(spec.seed)
    U = tree.w_list.astype(np.float64)

    # --- first layer ---------------------------------------------------------
    W1 = np.zeros((WIDTH, spec.dim))
    v = rng.normal(size=spec.dim)
    v -= U.T @ (U @ v)                      # project out every ODT normal
    v -= U.T @ (U @ v)                      # twice, for numerical cleanliness
    W1[PROBE] = v / np.linalg.norm(v)
    for node in range(LEVEL4_FIRST, LEVEL4_FIRST + NUM_LEVEL4):
        W1[first_layer_index(node, -1)] = -U[node]
        W1[first_layer_index(node, +1)] = +U[node]
    for i in range(NUM_RANDOM):
        r = rng.normal(size=spec.dim)
        W1[1 + 2 * NUM_LEVEL4 + i] = r / np.linalg.norm(r)
    W1 *= spec.eps

    # --- second layer --------------------------------------------------------
    W2 = np.zeros((WIDTH, WIDTH))
    for g, name in enumerate(GROUP_ORDER):
        cands = [first_layer_index(n, s) for s, n in GROUP_CANDIDATES[name]]
        for r in range(spec.group_size):
            row = g * spec.group_size + r
            W2[row, PROBE] = spec.group_sign[g]
            if spec.subset_mode == "all":
                chosen = cands
            elif spec.subset_mode == "random":
                mask = rng.random(len(cands)) < 0.5
                if not mask.any():
                    mask[rng.integers(len(cands))] = True
                chosen = [c for c, m in zip(cands, mask) if m]
            else:
                raise ValueError(f"subset_mode must be 'random' or 'all', got {spec.subset_mode!r}")
            W2[row, chosen] = 1.0

    # --- third layer and output ---------------------------------------------
    W3 = (rng.random((WIDTH, WIDTH)) < spec.layer3_density).astype(np.float64)
    Wout = np.full((1, WIDTH), spec.output_scale)
    return {
        "hidden_layers.0.weight": W1,
        "hidden_layers.1.weight": W2,
        "hidden_layers.2.weight": W3,
        "output_layer.weight": Wout,
    }


def build_designed_model(spec: DesignedInitSpec, tree: COBODTTree) -> ReLUMLP:
    """The designed init as a `ReLUMLP` (no biases, plain ReLU)."""
    weights = build_designed_weights(spec, tree)
    model = ReLUMLP(input_dim=spec.dim, hidden_dims=[WIDTH, WIDTH, WIDTH], bias=False)
    model.load_state_dict({k: torch.from_numpy(v).float() for k, v in weights.items()})
    return model


def logit_scale(model: ReLUMLP, x: np.ndarray, device: str = "cpu", n: int = 100_000) -> dict[str, float]:
    """RMS and max |logit| on a sample, to choose ``eps`` so the residue starts near 1/2."""
    m = model.to(device).eval()
    with torch.no_grad():
        z = m(torch.from_numpy(x[:n]).float().to(device)).cpu().numpy()
    return {"rms_logit": float(np.sqrt((z**2).mean())), "max_abs_logit": float(np.abs(z).max())}


def odt_coordinates(vectors: np.ndarray, tree: COBODTTree) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates of rows of ``vectors`` on the ODT normals, and the off-span norm."""
    U = tree.w_list.astype(np.float64)
    V = np.asarray(vectors, dtype=np.float64)
    coords = V @ U.T
    off = np.linalg.norm(V - coords @ U, axis=-1)
    return coords, off


def probe_gradient_report(
    model: ReLUMLP,
    x: np.ndarray,
    y: np.ndarray,
    tree: COBODTTree,
    *,
    device: str = "cpu",
    per_group: bool = True,
) -> dict[str, Any]:
    """Where does the probe's negative population gradient point at this state?

    Returns the ODT coordinates of ``-grad w_probe`` normalized to unit length,
    its cosine with ``u_0``, the off-span fraction, the gradient norm, and, when
    ``per_group``, the same cosine with ``u_0`` when only one second-layer group is
    connected (the other groups' rows zeroed). The per-group numbers are
    approximate: zeroing groups also changes the residue.
    """
    from layer_lenses.relu_analysis import log_loss_gradients

    def report(m: ReLUMLP) -> dict[str, float]:
        g = -log_loss_gradients(model=m, x=x, y=y, device=device)["hidden_layers.0.weight"][PROBE].numpy()
        norm = np.linalg.norm(g)
        coords, off = odt_coordinates(g[None] / max(norm, 1e-30), tree)
        return {"grad_norm": float(norm), "cos_u0": float(coords[0, 0]), "off_span_fraction": float(off[0] ** 2),
                "odt_coords": coords[0]}

    out: dict[str, Any] = {"full": report(model)}
    if per_group:
        base = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        gs = base["hidden_layers.1.weight"].shape[0] // 4
        for g, name in enumerate(GROUP_ORDER):
            sd = {k: v.clone() for k, v in base.items()}
            mask = torch.zeros(WIDTH, 1)
            mask[g * gs:(g + 1) * gs] = 1.0
            sd["hidden_layers.1.weight"] = sd["hidden_layers.1.weight"] * mask
            m = ReLUMLP(input_dim=model.input_dim, hidden_dims=list(model.hidden_dims), bias=False)
            m.load_state_dict(sd)
            out[f"group_{name}"] = report(m)
    return out


def run_designed_flow(
    model: ReLUMLP,
    x: np.ndarray,
    y: np.ndarray,
    *,
    lr: float,
    steps: int,
    momentum: float = 0.0,
    snapshot_every: int = 10,
    device: str = "cpu",
    show_progress: bool = False,
    freeze_first_layer_rows: list[int] | None = None,
) -> dict[str, Any]:
    """Full-batch gradient descent (constant LR, optional heavy-ball) from ``model``.

    Wraps `train_relu_mlp` with ``batch_size = len(x)`` so each epoch is one step,
    and a constant learning rate. Snapshots every ``snapshot_every`` steps hold
    all layers. ``freeze_first_layer_rows`` keeps those first-layer neurons fixed
    (e.g. the cue neurons) via the trainer's parameter update mask.
    """
    mask = None
    if freeze_first_layer_rows:
        mask = {n: np.ones(tuple(p.shape), dtype=np.float32) for n, p in model.named_parameters()}
        mask["hidden_layers.0.weight"][list(freeze_first_layer_rows)] = 0.0
    config = ReLUTrainConfig(
        epochs=steps,
        lr=lr,
        batch_size=len(x),
        seed=0,
        device=device,
        snapshot_epochs=tuple(range(0, steps + 1, snapshot_every)),
        lr_scheduler_eta_min_ratio=1.0,
        optimizer="sgd",
        momentum=momentum,
        show_progress=show_progress,
        parameter_update_mask=mask,
    )
    return train_relu_mlp(model=model, x_train=x, y_train=y, config=config)


def neuron_trajectory_table(out: dict[str, Any], tree: COBODTTree, neuron: int = PROBE) -> pd.DataFrame:
    """Per snapshot: the neuron's cosine with every ODT normal, its norm, off-span fraction."""
    U = tree.w_list.astype(np.float64)
    rows = []
    for step in sorted(out["checkpoint_snapshots"]):
        w = out["checkpoint_snapshots"][step]["hidden_layers.0.weight"][neuron].numpy().astype(np.float64)
        n = np.linalg.norm(w)
        c = (U @ w) / max(n, 1e-30)
        row = {"step": step, "norm": n, "off_span_fraction": float(1 - (c**2).sum())}
        row.update({f"cos_u{j}": float(c[j]) for j in range(U.shape[0])})
        rows.append(row)
    return pd.DataFrame(rows)


def sign_flip_variants() -> list[tuple[int, int, int, int]]:
    """All 16 sign patterns for the probe's weight into groups A, B, C, D."""
    return [(a, b, c, d) for a in (1, -1) for b in (1, -1) for c in (1, -1) for d in (1, -1)]


# ---------------------------------------------------------------------------
# Multi-probe designs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CueGroup:
    """One block of identical second-layer units.

    Each unit gets ``probe_weight`` from probe ``probe`` and reads the level-4 cue
    neurons ``cues`` (pairs ``(sign, node)``). ``cue_mode="relu"`` puts +1 on the
    listed cue neuron; ``"linear"`` puts +1 on it and -1 on its opposite-sign twin,
    so the unit sees ``sign * (u_node . x)`` rather than ``ReLU(sign * u_node . x)``.
    """

    probe: int
    probe_weight: float
    cues: tuple[tuple[int, int], ...]
    cue_mode: str = "relu"
    size: int = 10


@dataclass(frozen=True)
class MultiProbeSpec:
    """A first layer of ``n_probes`` probes, the 32 level-4 cue neurons, and random fill."""

    groups: tuple[CueGroup, ...]
    n_probes: int = 2
    eps: float = 1e-3
    seed: int = 0
    layer3: str = "identity"  # "identity" or "random01"
    output_scale: float = 20.0
    layer3_density: float = 0.5
    width: int = WIDTH
    dim: int = 100
    depth: int = 5


def cue_index(node: int, sign: int, n_probes: int) -> int:
    """First-layer row of the cue neuron ``sign * u_node`` when ``n_probes`` probes come first."""
    return first_layer_index(node, sign) - 1 + n_probes


def multiprobe_labels(spec: MultiProbeSpec) -> list[str]:
    labels = [f"probe{i}" for i in range(spec.n_probes)]
    for node in range(LEVEL4_FIRST, LEVEL4_FIRST + NUM_LEVEL4):
        labels += [f"-u{node}", f"+u{node}"]
    labels += [f"rand{i}" for i in range(spec.width - len(labels))]
    return labels


def build_multiprobe_weights(spec: MultiProbeSpec, tree: COBODTTree) -> dict[str, np.ndarray]:
    """Weights of a multi-probe design; see `CueGroup` and `MultiProbeSpec`."""
    if tree.w_list.shape != (31, spec.dim):
        raise ValueError("the design assumes a depth-5 ODT with 31 internal nodes")
    n_fill = spec.width - spec.n_probes - 2 * NUM_LEVEL4
    if n_fill < 0:
        raise ValueError(f"width {spec.width} too small for {spec.n_probes} probes and 32 cue neurons")
    if sum(g.size for g in spec.groups) > spec.width:
        raise ValueError("group sizes exceed the second-layer width")
    rng = np.random.default_rng(spec.seed)
    U = tree.w_list.astype(np.float64)

    W1 = np.zeros((spec.width, spec.dim))
    for p in range(spec.n_probes):
        v = rng.normal(size=spec.dim)
        v -= U.T @ (U @ v)
        v -= U.T @ (U @ v)
        W1[p] = v / np.linalg.norm(v)
    for node in range(LEVEL4_FIRST, LEVEL4_FIRST + NUM_LEVEL4):
        W1[cue_index(node, -1, spec.n_probes)] = -U[node]
        W1[cue_index(node, +1, spec.n_probes)] = +U[node]
    for i in range(n_fill):
        r = rng.normal(size=spec.dim)
        W1[spec.n_probes + 2 * NUM_LEVEL4 + i] = r / np.linalg.norm(r)
    W1 *= spec.eps

    W2 = np.zeros((spec.width, spec.width))
    row = 0
    for g in spec.groups:
        if not 0 <= g.probe < spec.n_probes:
            raise ValueError(f"group refers to probe {g.probe} but n_probes={spec.n_probes}")
        for _ in range(g.size):
            W2[row, g.probe] = g.probe_weight
            for s, n in g.cues:
                W2[row, cue_index(n, s, spec.n_probes)] += 1.0
                if g.cue_mode == "linear":
                    W2[row, cue_index(n, -s, spec.n_probes)] -= 1.0
                elif g.cue_mode != "relu":
                    raise ValueError(f"cue_mode must be 'relu' or 'linear', got {g.cue_mode!r}")
            row += 1

    if spec.layer3 == "identity":
        W3 = np.eye(spec.width)
    elif spec.layer3 == "random01":
        W3 = (rng.random((spec.width, spec.width)) < spec.layer3_density).astype(np.float64)
    else:
        raise ValueError(f"layer3 must be 'identity' or 'random01', got {spec.layer3!r}")
    Wout = np.full((1, spec.width), spec.output_scale)
    return {
        "hidden_layers.0.weight": W1,
        "hidden_layers.1.weight": W2,
        "hidden_layers.2.weight": W3,
        "output_layer.weight": Wout,
    }


def build_multiprobe_model(spec: MultiProbeSpec, tree: COBODTTree) -> ReLUMLP:
    weights = build_multiprobe_weights(spec, tree)
    model = ReLUMLP(input_dim=spec.dim, hidden_dims=[spec.width] * 3, bias=False)
    model.load_state_dict({k: torch.from_numpy(v).float() for k, v in weights.items()})
    return model


LEFT_NEG = tuple((-1, n) for n in range(15, 23))   # A-type: left subtree, negative side  -> pushes -u_0
RIGHT_POS = tuple((+1, n) for n in range(23, 31))  # B-type: right subtree, positive side -> pushes -u_0
RIGHT_NEG = tuple((-1, n) for n in range(23, 31))  # C-type: right subtree, negative side -> pushes +u_0
LEFT_POS = tuple((+1, n) for n in range(15, 23))   # D-type: left subtree, positive side  -> pushes +u_0


def two_probe_spec(*, cue_mode: str = "relu", size: int = 10, **kw: Any) -> MultiProbeSpec:
    """Probe 0 should run to +u_0 (C- and D-type groups), probe 1 to -u_0 (A- and B-type).

    All probe weights are -1 (informative), no group is shared between probes.
    """
    groups = (
        CueGroup(probe=0, probe_weight=-1.0, cues=RIGHT_NEG, cue_mode=cue_mode, size=size),
        CueGroup(probe=0, probe_weight=-1.0, cues=LEFT_POS, cue_mode=cue_mode, size=size),
        CueGroup(probe=1, probe_weight=-1.0, cues=LEFT_NEG, cue_mode=cue_mode, size=size),
        CueGroup(probe=1, probe_weight=-1.0, cues=RIGHT_POS, cue_mode=cue_mode, size=size),
    )
    return MultiProbeSpec(groups=groups, n_probes=2, **kw)


def multiprobe_gradient_report(
    model: ReLUMLP,
    spec: MultiProbeSpec,
    x: np.ndarray,
    y: np.ndarray,
    tree: COBODTTree,
    *,
    device: str = "cpu",
) -> pd.DataFrame:
    """Per probe: cosine of its negative population gradient with u_0, energy split, and the
    same cosine with each group alone (other groups' rows zeroed; approximate)."""
    from layer_lenses.relu_analysis import log_loss_gradients

    def probe_cos(m: ReLUMLP, p: int) -> dict[str, float]:
        g = -log_loss_gradients(model=m, x=x, y=y, device=device)["hidden_layers.0.weight"][p].numpy()
        c, off = odt_coordinates(g[None] / np.linalg.norm(g), tree)
        return {"cos_u0": float(c[0, 0]), "grad_norm": float(np.linalg.norm(g)),
                "level4_energy": float((c[0, 15:] ** 2).sum()), "off_span": float(off[0] ** 2)}

    base = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bounds, row = [], 0
    for g in spec.groups:
        bounds.append((row, row + g.size)); row += g.size
    rows = []
    for p in range(spec.n_probes):
        rec = {"probe": p, **probe_cos(model, p)}
        for gi, ((a, b), g) in enumerate(zip(bounds, spec.groups)):
            if g.probe != p:
                continue
            sd = {k: v.clone() for k, v in base.items()}
            mask = torch.zeros(spec.width, 1); mask[a:b] = 1.0
            sd["hidden_layers.1.weight"] = sd["hidden_layers.1.weight"] * mask
            m = ReLUMLP(input_dim=spec.dim, hidden_dims=[spec.width] * 3, bias=False); m.load_state_dict(sd)
            rec[f"group{gi}_cos_u0"] = probe_cos(m, p)["cos_u0"]
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Probes for arbitrary internal nodes
# ---------------------------------------------------------------------------


def level4_descendants(node: int, depth: int = 5) -> list[int]:
    """Level-4 internal nodes in the subtree of ``node`` (``node`` itself if at level 4)."""
    last_level_first = 2 ** (depth - 1) - 1  # 15 for depth 5
    frontier = [node]
    while frontier[0] < last_level_first:
        frontier = [c for n in frontier for c in (2 * n + 1, 2 * n + 2)]
    return frontier


def node_probe_groups(
    probe: int,
    node: int,
    pole: int,
    *,
    cue_mode: str = "relu",
    size: int = 10,
    probe_weight: float = -1.0,
) -> tuple[CueGroup, CueGroup]:
    """The balanced pair of informative groups that sends ``probe`` toward ``pole * u_node``.

    For a unit gated on a level-4 cue of sign ``s`` under a child of ``node`` on side
    ``side`` (-1 left, +1 right), the push on the probe toward ``u_node`` has sign
    ``-s * side`` (with a -1 probe weight). So ``pole = +1`` pairs negative-side cues
    under the right child with positive-side cues under the left child, and
    ``pole = -1`` the reverse. Both children lie on the same side of every ancestor,
    so the pair's pushes toward ancestors cancel; using every cue under each child
    cancels the pushes toward descendants as well. At the root this reproduces the
    C/D (pole +1) and A/B (pole -1) groups.
    """
    if pole not in (-1, 1):
        raise ValueError("pole must be +1 or -1")
    if node >= LEVEL4_FIRST:
        raise ValueError("probes target internal nodes above level 4")
    left, right = 2 * node + 1, 2 * node + 2
    if pole > 0:
        first = tuple((-1, n) for n in level4_descendants(right))
        second = tuple((+1, n) for n in level4_descendants(left))
    else:
        first = tuple((-1, n) for n in level4_descendants(left))
        second = tuple((+1, n) for n in level4_descendants(right))
    return (
        CueGroup(probe=probe, probe_weight=probe_weight, cues=first, cue_mode=cue_mode, size=size),
        CueGroup(probe=probe, probe_weight=probe_weight, cues=second, cue_mode=cue_mode, size=size),
    )


SIX_PROBE_TARGETS: tuple[tuple[int, int], ...] = ((0, +1), (0, -1), (1, +1), (1, -1), (2, +1), (2, -1))


def six_probe_spec(
    *,
    cue_mode: str = "relu",
    root_size: int = 4,
    child_size: int = 3,
    **kw: Any,
) -> MultiProbeSpec:
    """Six probes: +-u_0 (groups of ``root_size``), +-u_1 and +-u_2 (groups of ``child_size``).

    With the defaults the twelve groups fill the 40 second-layer rows exactly; the
    first layer is 6 probes, 32 cues and 2 random neurons.
    """
    groups: list[CueGroup] = []
    for p, (node, pole) in enumerate(SIX_PROBE_TARGETS):
        groups += node_probe_groups(p, node, pole, cue_mode=cue_mode, size=root_size if node == 0 else child_size)
    return MultiProbeSpec(groups=tuple(groups), n_probes=len(SIX_PROBE_TARGETS), **kw)


def probe_target_report(
    model: ReLUMLP,
    spec: MultiProbeSpec,
    targets: tuple[tuple[int, int], ...],
    x: np.ndarray,
    y: np.ndarray,
    tree: COBODTTree,
    *,
    device: str = "cpu",
) -> pd.DataFrame:
    """Per probe: cosine of its negative gradient with its signed target, with u_0, u_1, u_2,
    the level-4 energy, and the ODT node of largest |coordinate|."""
    from layer_lenses.relu_analysis import log_loss_gradients

    G = -log_loss_gradients(model=model, x=x, y=y, device=device)["hidden_layers.0.weight"].numpy()
    rows = []
    for p, (node, pole) in enumerate(targets):
        g = G[p] / np.linalg.norm(G[p])
        c, off = odt_coordinates(g[None], tree)
        c = c[0]
        rows.append({"probe": p, "target": f"{'+' if pole > 0 else '-'}u{node}", "cos_target": float(pole * c[node]),
                     "cos_u0": float(c[0]), "cos_u1": float(c[1]), "cos_u2": float(c[2]),
                     "level4_energy": float((c[15:] ** 2).sum()), "off_span": float(off[0] ** 2),
                     "argmax_node": int(np.abs(c).argmax()), "grad_norm": float(np.linalg.norm(G[p]))})
    return pd.DataFrame(rows)


def cue_rows(spec: MultiProbeSpec) -> list[int]:
    """First-layer row indices of the 32 level-4 cue neurons in a multi-probe design."""
    return list(range(spec.n_probes, spec.n_probes + 2 * NUM_LEVEL4))


# ---------------------------------------------------------------------------
# Full hierarchy design: two probes per internal node 0..14
# ---------------------------------------------------------------------------

HIERARCHY_TARGETS: tuple[tuple[int, int], ...] = tuple((node, pole) for node in range(15) for pole in (+1, -1))
NODE_LEVEL = np.array([0] + [1] * 2 + [2] * 4 + [3] * 8 + [4] * 16)


@dataclass(frozen=True)
class HierarchyDesignSpec:
    """Two probes per internal node, 32 cues, random fill; widths (80, 40, 40) by default.

    ``units_per_level`` gives the number of second-layer units per probe by the
    target node's level (root .. level 3). ``subset_mode="random"`` lets every unit
    draw a random subset of its candidate cues with at least one per child;
    ``"all"`` uses every candidate. ``cue_offspan_ratio`` adds a random component
    orthogonal to every ODT normal to each cue neuron (relative to the ODT part).
    Layers 2-4 keep their {-1,0,+1} / {0,1} / ones sign structure, scaled by
    ``layer2_scale``, ``layer3_scale``, ``output_scale`` so their rows have order-one
    norms; ``eps`` is the common norm of every first-layer row.
    """

    eps: float = 0.01
    seed: int = 0
    cue_mode: str = "relu"
    subset_mode: str = "random"
    cue_offspan_ratio: float = 0.2
    units_per_level: tuple[int, int, int, int] = (4, 2, 1, 1)
    width1: int = 80
    width2: int = 40
    width3: int = 40
    layer2_scale: float = 1 / 3
    layer3_density: float = 0.5
    # If true, every third-layer row has exactly round(layer3_density * width2) ones
    # (a random subset per row) instead of independent Bernoulli entries.
    layer3_exact: bool = False
    layer3_scale: float = 1 / np.sqrt(20)
    output_scale: float = 1 / np.sqrt(40)
    # "ones": all output weights positive (the logit can only be positive at init, so
    # y = -1 is representable only by negative third-layer weights or a dead layer).
    # "balanced": alternating +1 / -1 over the third-layer units.
    output_signs: str = "ones"
    # Path-sign-aware cue rule: a unit pushes its probe toward the designed pole only if the
    # unit's initial path to the logit, sum_k Wout[k] W3[k, unit], is positive. If true, units
    # with a negative path get their cue signs flipped (= wired for the opposite pole), so the
    # first-order push points to the designed pole again. Units with a zero path are untouched.
    path_sign_cues: bool = False
    # "none": every cue entry of a unit is +-1 (so a root unit's cue block has norm sqrt(32)
    # against a probe entry of 1, and after row normalisation its probe entry is the smallest).
    # "inv_sqrt": each unit's cue block is scaled by 1/sqrt(number of nonzero cue entries), so
    # every unit has probe entry 1 and cue-block norm 1 before the row rescale, i.e. the same
    # probe weight at every level afterwards.
    cue_block_scale: str = "none"
    # With layer3_exact, repair the third layer by row-count-preserving 2x2 swaps until every
    # unit has a nonzero path to the logit (needed for path_sign_cues to apply to every unit).
    layer3_nonzero_paths: bool = False
    # "random": third layer as above. "balanced_paths" (needs output_signs="balanced", even
    # width2 and width3): every unit connects to exactly width3/2 third-layer neurons; unit 2j
    # to `layer3_majority` of the positive-output neurons and the rest negative-output ones,
    # unit 2j+1 the reverse, so every unit's path to the logit is +-(2*majority - width3/2)
    # quanta with alternating sign. Within a pair the two units use complementary subsets of
    # each output-sign group, so every third-layer row has exactly width2/2 ones.
    layer3_mode: str = "random"
    layer3_majority: int = 15
    # Multiplies the 30 probe rows at the very end of the construction (after the row-norm
    # rescale), so e.g. 0.01 starts the probes near the origin while cues keep their norm.
    probe_scale: float = 1.0
    # If set, every row of every layer is rescaled to these norms (layers 1..4),
    # overriding eps and the layer scales, so the init has the row norms of a standard
    # scheme while keeping the sign structure. See `standard_row_norms`.
    row_norms: tuple[float, float, float, float] | None = None
    dim: int = 100
    depth: int = 5

    @property
    def n_probes(self) -> int:
        return len(HIERARCHY_TARGETS)

    @property
    def cue_first_row(self) -> int:
        return self.n_probes

    @property
    def n_random(self) -> int:
        return self.width1 - self.n_probes - 2 * NUM_LEVEL4


def hierarchy_cue_index(node: int, sign: int, spec: HierarchyDesignSpec) -> int:
    return spec.cue_first_row + 2 * (node - LEVEL4_FIRST) + (1 if sign > 0 else 0)


def hierarchy_unit_cues(node: int, pole: int) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Cue candidates for a single balanced unit: ``(left-child set, right-child set)``.

    Same rule as `node_probe_groups` (push sign ``-s * side``), with both children's
    sets in one unit so the ancestor pushes cancel inside the unit.
    """
    left, right = 2 * node + 1, 2 * node + 2
    if pole > 0:
        return tuple((+1, n) for n in level4_descendants(left)), tuple((-1, n) for n in level4_descendants(right))
    return tuple((-1, n) for n in level4_descendants(left)), tuple((+1, n) for n in level4_descendants(right))


def hierarchy_labels(spec: HierarchyDesignSpec) -> list[str]:
    labels = [f"probe{p}:{'+' if pole > 0 else '-'}u{node}" for p, (node, pole) in enumerate(HIERARCHY_TARGETS)]
    for node in range(LEVEL4_FIRST, LEVEL4_FIRST + NUM_LEVEL4):
        labels += [f"-u{node}", f"+u{node}"]
    labels += [f"rand{i}" for i in range(spec.n_random)]
    return labels


def _repair_zero_unit_paths(W3: np.ndarray, wout: np.ndarray, rng: np.random.Generator, max_iter: int = 10_000) -> np.ndarray:
    """Make every column of a 0/1-structured ``W3`` have a nonzero path ``wout @ W3[:, u]``.

    Uses 2x2 swaps between a positive-output row ``k`` and a negative-output row ``k2`` and
    two columns ``u`` (the zero-path one) and ``v``: ones move from ``(k2, u), (k, v)`` to
    ``(k, u), (k2, v)``, so every row keeps its count. ``v`` is chosen so it does not become
    zero itself. Entries are assumed to share one magnitude (``layer3_exact``).
    """
    W = W3.copy()
    mag = float(np.abs(W[W != 0]).max()) if (W != 0).any() else 1.0
    pos_rows = np.nonzero(wout > 0)[0]; neg_rows = np.nonzero(wout < 0)[0]
    step = 2 * mag * float(np.abs(wout).max())
    for _ in range(max_iter):
        path = W.T @ wout
        zero = np.nonzero(np.isclose(path, 0.0, atol=1e-9))[0]
        if zero.size == 0:
            return W
        u = int(rng.choice(zero))
        candidates = []
        for k in pos_rows:
            if W[k, u] != 0:
                continue
            for k2 in neg_rows:
                if W[k2, u] == 0:
                    continue
                vs = np.nonzero((W[k] != 0) & (W[k2] == 0))[0]
                vs = [v for v in vs if v != u and not np.isclose(path[v] - step, 0.0, atol=1e-9)]
                if vs:
                    candidates.append((k, k2, vs))
        if not candidates:
            raise RuntimeError(f"could not repair the zero path of unit {u}")
        k, k2, vs = candidates[rng.integers(len(candidates))]; v = int(rng.choice(vs))
        W[k, u], W[k2, u], W[k, v], W[k2, v] = mag, 0.0, 0.0, mag
    raise RuntimeError("zero-path repair did not converge")


def build_hierarchy_weights(spec: HierarchyDesignSpec, tree: COBODTTree) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Weights of the full hierarchy design plus metadata (unit -> probe map, row labels)."""
    if tree.w_list.shape != (31, spec.dim):
        raise ValueError("the design assumes a depth-5 ODT with 31 internal nodes")
    if spec.n_random < 0:
        raise ValueError("width1 too small for 30 probes and 32 cues")
    n_units = sum(spec.units_per_level[NODE_LEVEL[node]] for node, _ in HIERARCHY_TARGETS)
    if n_units > spec.width2:
        raise ValueError(f"units_per_level needs {n_units} second-layer rows, width2 is {spec.width2}")
    rng = np.random.default_rng(spec.seed)
    U = tree.w_list.astype(np.float64)

    def offspan(v: np.ndarray) -> np.ndarray:
        v = v - U.T @ (U @ v)
        v = v - U.T @ (U @ v)
        return v

    W1 = np.zeros((spec.width1, spec.dim))
    for p in range(spec.n_probes):
        v = offspan(rng.normal(size=spec.dim))
        W1[p] = v / np.linalg.norm(v)
    for node in range(LEVEL4_FIRST, LEVEL4_FIRST + NUM_LEVEL4):
        for sign in (-1, +1):
            r = offspan(rng.normal(size=spec.dim)); r /= np.linalg.norm(r)
            v = sign * U[node] + spec.cue_offspan_ratio * r
            W1[hierarchy_cue_index(node, sign, spec)] = v / np.linalg.norm(v)
    for i in range(spec.n_random):
        r = rng.normal(size=spec.dim)
        W1[spec.cue_first_row + 2 * NUM_LEVEL4 + i] = r / np.linalg.norm(r)
    W1 *= spec.eps

    W2 = np.zeros((spec.width2, spec.width1))
    unit_probe = np.full(spec.width2, -1, dtype=int)
    row = 0
    for p, (node, pole) in enumerate(HIERARCHY_TARGETS):
        left_set, right_set = hierarchy_unit_cues(node, pole)
        for _ in range(spec.units_per_level[NODE_LEVEL[node]]):
            W2[row, p] = -1.0
            for cand in (left_set, right_set):
                if spec.subset_mode == "all":
                    chosen = list(cand)
                elif spec.subset_mode == "random":
                    mask = rng.random(len(cand)) < 0.5
                    if not mask.any():
                        mask[rng.integers(len(cand))] = True
                    chosen = [c for c, m in zip(cand, mask) if m]
                else:
                    raise ValueError(f"subset_mode must be 'random' or 'all', got {spec.subset_mode!r}")
                for s, n in chosen:
                    W2[row, hierarchy_cue_index(n, s, spec)] += 1.0
                    if spec.cue_mode == "linear":
                        W2[row, hierarchy_cue_index(n, -s, spec)] -= 1.0
                    elif spec.cue_mode != "relu":
                        raise ValueError(f"cue_mode must be 'relu' or 'linear', got {spec.cue_mode!r}")
            unit_probe[row] = p
            row += 1
    if spec.cue_block_scale == "inv_sqrt":
        for u in range(spec.width2):
            if unit_probe[u] < 0:
                continue
            cue_cols = np.ones(spec.width1, dtype=bool); cue_cols[unit_probe[u]] = False
            n_cues = int((W2[u, cue_cols] != 0).sum())
            if n_cues:
                W2[u, cue_cols] /= np.sqrt(n_cues)
    elif spec.cue_block_scale != "none":
        raise ValueError(f"cue_block_scale must be 'none' or 'inv_sqrt', got {spec.cue_block_scale!r}")
    W2 *= spec.layer2_scale
    if spec.layer3_mode == "balanced_paths":
        if spec.output_signs != "balanced":
            raise ValueError("layer3_mode='balanced_paths' needs output_signs='balanced'")
        if spec.width2 % 2 or spec.width3 % 2:
            raise ValueError("layer3_mode='balanced_paths' needs even width2 and width3")
        n_conn = spec.width3 // 2; minority = n_conn - spec.layer3_majority
        if not 0 <= minority < spec.layer3_majority:
            raise ValueError(f"layer3_majority must be in ({n_conn / 2}, {n_conn}], got {spec.layer3_majority}")
        pos_rows = np.arange(0, spec.width3, 2); neg_rows = np.arange(1, spec.width3, 2)   # output_signs="balanced" alternates +,-
        W3 = np.zeros((spec.width3, spec.width2))
        for j in range(spec.width2 // 2):
            a, b = 2 * j, 2 * j + 1
            pa = rng.permutation(pos_rows); W3[pa[: spec.layer3_majority], a] = 1.0; W3[pa[spec.layer3_majority:], b] = 1.0
            nb = rng.permutation(neg_rows); W3[nb[: spec.layer3_majority], b] = 1.0; W3[nb[spec.layer3_majority:], a] = 1.0
        W3 *= spec.layer3_scale
    elif spec.layer3_mode != "random":
        raise ValueError(f"layer3_mode must be 'random' or 'balanced_paths', got {spec.layer3_mode!r}")
    elif spec.layer3_exact:
        k = int(round(spec.layer3_density * spec.width2))
        W3 = np.zeros((spec.width3, spec.width2))
        for r in range(spec.width3):
            W3[r, rng.choice(spec.width2, size=k, replace=False)] = 1.0
        W3 *= spec.layer3_scale
    else:
        W3 = (rng.random((spec.width3, spec.width2)) < spec.layer3_density).astype(np.float64) * spec.layer3_scale
    if spec.output_signs == "ones":
        Wout = np.full((1, spec.width3), spec.output_scale)
    elif spec.output_signs == "balanced":
        Wout = (np.where(np.arange(spec.width3) % 2 == 0, 1.0, -1.0) * spec.output_scale)[None, :]
    else:
        raise ValueError(f"output_signs must be 'ones' or 'balanced', got {spec.output_signs!r}")
    if spec.layer3_nonzero_paths and spec.layer3_mode == "random":
        if not spec.layer3_exact:
            raise ValueError("layer3_nonzero_paths requires layer3_exact=True")
        W3 = _repair_zero_unit_paths(W3, Wout.ravel(), rng)
    unit_path = W3.T @ Wout.ravel()                      # init weight of each unit's path to the logit
    unit_path_sign = np.sign(unit_path).astype(int)
    flipped_units: list[int] = []
    if spec.path_sign_cues:
        for u in range(spec.width2):
            if unit_path_sign[u] < 0 and unit_probe[u] >= 0:
                cue_cols = np.ones(spec.width1, dtype=bool); cue_cols[unit_probe[u]] = False
                W2[u, cue_cols] *= -1.0
                flipped_units.append(u)
    weights = {
        "hidden_layers.0.weight": W1,
        "hidden_layers.1.weight": W2,
        "hidden_layers.2.weight": W3,
        "output_layer.weight": Wout,
    }
    if spec.row_norms is not None:
        for key, target in zip(weights, spec.row_norms):
            M = weights[key]
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            weights[key] = np.where(norms > 0, M / np.maximum(norms, 1e-30) * target, 0.0)
    if spec.probe_scale != 1.0:
        weights["hidden_layers.0.weight"] = weights["hidden_layers.0.weight"].copy()
        weights["hidden_layers.0.weight"][: spec.n_probes] *= spec.probe_scale
    meta = {"unit_probe": unit_probe, "labels": hierarchy_labels(spec), "targets": HIERARCHY_TARGETS,
            "cue_rows": list(range(spec.cue_first_row, spec.cue_first_row + 2 * NUM_LEVEL4)),
            "unit_path_sign": unit_path_sign, "flipped_units": flipped_units, "zero_path_units": [int(u) for u in np.nonzero(unit_path_sign == 0)[0]]}
    return weights, meta


def build_hierarchy_model(spec: HierarchyDesignSpec, tree: COBODTTree) -> tuple[ReLUMLP, dict[str, Any]]:
    weights, meta = build_hierarchy_weights(spec, tree)
    model = ReLUMLP(input_dim=spec.dim, hidden_dims=[spec.width1, spec.width2, spec.width3], bias=False)
    model.load_state_dict({k: torch.from_numpy(v).float() for k, v in weights.items()})
    return model, meta


def hierarchy_probe_table(W1: np.ndarray, tree: COBODTTree) -> pd.DataFrame:
    """Per probe (rows 0..29 of a first layer): signed cosine with its target, the top ODT node,
    the level-4 energy, and the norm."""
    U = tree.w_list.astype(np.float64)
    rows = []
    for p, (node, pole) in enumerate(HIERARCHY_TARGETS):
        w = W1[p].astype(np.float64); n = np.linalg.norm(w); c = U @ w / max(n, 1e-30)
        rows.append({"probe": p, "node": node, "pole": pole, "target": f"{'+' if pole > 0 else '-'}u{node}", "level": int(NODE_LEVEL[node]),
                     "cos_target": float(pole * c[node]), "top_node": int(np.abs(c).argmax()), "top_abs_cos": float(np.abs(c).max()),
                     "level4_energy": float((c[15:] ** 2).sum()), "norm": float(n)})
    return pd.DataFrame(rows)


def standard_row_norms(scheme: str, spec: HierarchyDesignSpec) -> tuple[float, float, float, float]:
    """Row norms (layers 1..4) that a standard random init would give this architecture.

    ``"default"``: the model's own `ReLUMLP.reset_parameters`, std ``1/sqrt(3 fan_in)``,
    so every row has norm ``1/sqrt(3)``. ``"kaiming"``: std ``sqrt(2/fan_in)``, row norm
    ``sqrt(2)``. ``"xavier"``: std ``sqrt(2/(fan_in+fan_out))``, row norm
    ``sqrt(2 fan_in/(fan_in+fan_out))``.
    """
    fans = [(spec.dim, spec.width1), (spec.width1, spec.width2), (spec.width2, spec.width3), (spec.width3, 1)]
    if scheme == "default":
        return tuple(1 / np.sqrt(3) for _ in fans)
    if scheme == "kaiming":
        return tuple(np.sqrt(2.0) for _ in fans)
    if scheme == "xavier":
        return tuple(np.sqrt(2.0 * fi / (fi + fo)) for fi, fo in fans)
    raise ValueError(f"scheme must be 'default', 'kaiming' or 'xavier', got {scheme!r}")


def standard_scale_spec(scheme: str, **kw: Any) -> HierarchyDesignSpec:
    """A `HierarchyDesignSpec` whose rows have the norms of a standard init scheme."""
    base = HierarchyDesignSpec(**kw)
    return HierarchyDesignSpec(**{**base.__dict__, "row_norms": standard_row_norms(scheme, base)})


# ---------------------------------------------------------------------------
# Per-run reports for the hierarchy design (first-layer rows 0..29 are the probes,
# probe 2j -> +u_j, probe 2j+1 -> -u_j; rows 30..61 the cues; the rest random).
# ---------------------------------------------------------------------------


def _first_layer_node_assignment(W1: np.ndarray, tree: COBODTTree, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(top node, signed cosine to it, aligned flag) for every first-layer row."""
    W = np.asarray(W1, dtype=np.float64)
    C = (W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-30)) @ tree.w_list.astype(np.float64).T
    top = np.abs(C).argmax(1)
    signed = C[np.arange(W.shape[0]), top]
    return top, signed, np.abs(signed) > threshold


def probe_node_outcomes(W1: np.ndarray, tree: COBODTTree, threshold: float = 0.95) -> pd.DataFrame:
    """One row per internal node 0..14: what its two probes did and who else sits on the node.

    ``own``: number of the node's own probes aligned to it; ``intended`` / ``opposite``: an own
    probe sits at the designed pole / at the opposite pole; ``others``: other first-layer rows
    aligned to the node; ``probe_plus`` / ``probe_minus``: where the +u_j and -u_j probes ended.
    """
    top, signed, aligned = _first_layer_node_assignment(W1, tree, threshold)

    def where(i: int) -> str:
        return f"{'+' if signed[i] > 0 else '-'}u{top[i]}({abs(signed[i]):.2f})"

    rows = []
    for j in range(15):
        p, m = 2 * j, 2 * j + 1
        own = [(i, np.sign(signed[i])) for i in (p, m) if top[i] == j and aligned[i]]
        rows.append({
            "node": j, "level": int(NODE_LEVEL[j]), "probe_plus": where(p), "probe_minus": where(m), "own": len(own),
            "intended": any((i == p and s > 0) or (i == m and s < 0) for i, s in own),
            "opposite": any((i == p and s < 0) or (i == m and s > 0) for i, s in own),
            "others": int(sum(1 for i in range(W1.shape[0]) if i not in (p, m) and top[i] == j and aligned[i])),
        })
    return pd.DataFrame(rows)


def hierarchy_run_summary(W1: np.ndarray, tree: COBODTTree, threshold: float = 0.95) -> dict[str, Any]:
    """Coverage and probe-outcome counts for one first-layer snapshot."""
    top, signed, aligned = _first_layer_node_assignment(W1, tree, threshold)
    covered = sorted(set(int(n) for n in top[aligned]))
    po = probe_node_outcomes(W1, tree, threshold)
    return {
        "aligned_rows": int(aligned.sum()), "rows": int(W1.shape[0]),
        "nodes_covered": len(covered), "missing_nodes": [n for n in range(tree.w_list.shape[0]) if n not in covered],
        "own_probe_nodes": int((po.own > 0).sum()), "intended_pole": int(po.intended.sum()),
        "opposite_pole_only": int((po.opposite & ~po.intended).sum()), "both_poles": po.loc[po.own == 2, "node"].tolist(),
        "missed_by_own_probes": po.loc[po.own == 0, "node"].tolist(),
    }


def plot_probe_target_cosines(W1_stack: np.ndarray, epochs: np.ndarray, tree: COBODTTree, *, axes=None, title: str | None = None):
    """Signed cosine of every probe to its designed target over the run, one panel per level."""
    U = tree.w_list.astype(np.float64)
    W = np.asarray(W1_stack, dtype=np.float64)
    if axes is None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
    else:
        fig = axes[0].figure
    for p, (node, pole) in enumerate(HIERARCHY_TARGETS):
        w = W[:, p]; c = pole * (w @ U[node]) / np.maximum(np.linalg.norm(w, axis=1), 1e-30)
        ax = axes[int(NODE_LEVEL[node])]
        ax.plot(epochs, c, lw=1.1, ls="-" if pole > 0 else "--", label=f"{'+' if pole > 0 else '-'}u{node}")
    for l, ax in enumerate(axes):
        ax.axhline(0, color="gray", lw=0.6); ax.set_ylim(-1.05, 1.05); ax.set_title(f"level {l} probes", fontsize=9); ax.set_xlabel("epoch"); ax.grid(alpha=0.25)
        ax.legend(fontsize=6, ncol=2 if l >= 2 else 1)
    axes[0].set_ylabel("cosine to designed target")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig, axes


def probe_gradient_stats(grad_W1: np.ndarray, W1: np.ndarray, tree: COBODTTree) -> pd.DataFrame:
    """Where the population gradient wants to move each probe, per probe row.

    ``grad_W1`` is the gradient of the loss w.r.t. the first-layer weight (so the push is
    ``-grad_W1``). Columns: ``grad_norm``; ``cos_push_target`` (cosine of the push with the
    designed signed target); ``cos_push_row`` (with the current row); ``push_along_target``
    (signed component); energy fractions of the push along the target, along the target's
    level-4 descendants, along the other level-4 nodes, along the other internal nodes, and
    off the ODT span; ``push_top`` (the signed ODT node closest to the push) and
    ``row_cos_target`` (where the row currently is).
    """
    U = tree.w_list.astype(np.float64)
    G = -np.asarray(grad_W1, dtype=np.float64); W = np.asarray(W1, dtype=np.float64)
    rows = []
    for p, (node, pole) in enumerate(HIERARCHY_TARGETS):
        g = G[p]; gn = np.linalg.norm(g); c = U @ g; e = c ** 2 / max(gn ** 2, 1e-300)
        desc = level4_descendants(node)
        other_l4 = [n for n in range(LEVEL4_FIRST, LEVEL4_FIRST + NUM_LEVEL4) if n not in desc]
        other_int = [n for n in range(LEVEL4_FIRST) if n != node]
        top = int(np.abs(c).argmax())
        rows.append({
            "probe": p, "target": f"{'+' if pole > 0 else '-'}u{node}", "level": int(NODE_LEVEL[node]), "grad_norm": gn,
            "cos_push_target": float(pole * c[node] / max(gn, 1e-300)), "cos_push_row": float(g @ W[p] / max(gn * np.linalg.norm(W[p]), 1e-300)),
            "push_along_target": float(pole * c[node]),
            "frac_target": float(e[node]), "frac_desc_l4": float(e[desc].sum()), "frac_other_l4": float(e[other_l4].sum()),
            "frac_other_internal": float(e[other_int].sum()), "frac_offspan": float(max(0.0, 1.0 - e.sum())),
            "push_top": f"{'+' if c[top] > 0 else '-'}u{top}", "row_cos_target": float(pole * (U[node] @ W[p]) / max(np.linalg.norm(W[p]), 1e-300)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Lineage of probe deviations in the ODT tree (heap indexing: children of node i are 2i+1, 2i+2;
# the split at node i sends x with w_i . x > 0 to the right child 2i+2)
# ---------------------------------------------------------------------------------------------


def node_ancestors(node: int) -> list[int]:
    """Ancestors of ``node`` from its parent up to the root (empty for the root)."""
    out = []
    while node > 0:
        node = (node - 1) // 2
        out.append(node)
    return out


def node_in_subtree(node: int, root: int, n_nodes: int) -> bool:
    """True if ``node`` lies in the subtree rooted at ``root`` (root included)."""
    return node == root or root in node_ancestors(node)


def node_lineage(target: int, other: int, n_nodes: int = 31) -> dict[str, Any]:
    """Relation of ``other`` to ``target`` in the heap-indexed complete tree.

    ``relation`` is ``same``, ``ancestor`` (``other`` is above ``target``), ``descendant`` (below) or
    ``unrelated``; ``level_gap`` is the level difference (positive = ``other`` deeper); ``side`` says,
    for a descendant, whether it lies in the target's positive-side subtree (``+``: the child
    reached when w_target . x > 0) or the negative one (``-``), and, for an ancestor, on which side
    of ``other`` the target lies.
    """
    if target == other:
        return {"relation": "same", "level_gap": 0, "side": ""}
    if other in node_ancestors(target):
        side = "+" if node_in_subtree(target, 2 * other + 2, n_nodes) else "-"
        return {"relation": "ancestor", "level_gap": int(NODE_LEVEL[other] - NODE_LEVEL[target]), "side": side}
    if target in node_ancestors(other):
        side = "+" if node_in_subtree(other, 2 * target + 2, n_nodes) else "-"
        return {"relation": "descendant", "level_gap": int(NODE_LEVEL[other] - NODE_LEVEL[target]), "side": side}
    return {"relation": "unrelated", "level_gap": int(NODE_LEVEL[other] - NODE_LEVEL[target]), "side": ""}


def lineage_chance(target: int, n_nodes: int = 31) -> float:
    """Fraction of the other nodes that are ancestors or descendants of ``target`` (the chance level for a deviation to be lineage-related)."""
    related = sum(node_lineage(target, o, n_nodes)["relation"] in ("ancestor", "descendant") for o in range(n_nodes) if o != target)
    return related / (n_nodes - 1)


def probe_deviation_lineage(W1: np.ndarray, tree: COBODTTree, threshold: float = 0.95) -> pd.DataFrame:
    """Where each probe row ended and how that node relates to its designed target.

    One row per probe (rows 0-29 of ``W1``): ``target``, ``pole``, ``end_node`` (the ODT node with the
    largest |cosine|), ``end_cos`` (signed), ``captured`` (|cosine| >= threshold), ``deviated``
    (captured on another node), ``relation`` / ``level_gap`` / ``side`` from `node_lineage`, and
    ``chance`` (the fraction of other nodes on the target's lineage).
    """
    n_nodes = tree.w_list.shape[0]
    C = (W1[: len(HIERARCHY_TARGETS)] / np.maximum(np.linalg.norm(W1[: len(HIERARCHY_TARGETS)], axis=1, keepdims=True), 1e-12)) @ tree.w_list.T
    rows = []
    for p, (node, pole) in enumerate(HIERARCHY_TARGETS):
        j = int(np.abs(C[p]).argmax()); c = float(C[p, j]); captured = abs(c) >= threshold
        rel = node_lineage(node, j, n_nodes)
        rows.append({"probe": p, "target": node, "target_level": int(NODE_LEVEL[node]), "pole": pole, "end_node": j, "end_cos": c, "captured": captured, "deviated": captured and j != node,
                     "relation": rel["relation"] if captured else "none", "level_gap": rel["level_gap"], "side": rel["side"], "chance": lineage_chance(node, n_nodes)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Anatomy of second-layer units as leaf / region detectors
# ---------------------------------------------------------------------------------------------


def leaf_wrong_side_detectors(leaf: int, depth: int = 5) -> list[tuple[int, int]]:
    """For leaf ``leaf`` (0-based, left to right), the (node, pole) detector that is OFF on the leaf's side at each ancestor, root first.

    At an ancestor reached through its right child (w . x > 0 on the leaf) the wrong-side detector is -u; through
    the left child, +u. A unit that gives these detectors large negative weights fires only inside the leaf.
    """
    n = leaf + (2 ** depth - 1); out = []
    while n > 0:
        parent = (n - 1) // 2; out.append((parent, -1 if n == 2 * parent + 2 else +1)); n = parent
    return out[::-1]


def unit_leaf_anatomy(W1: np.ndarray, W2: np.ndarray, W3: np.ndarray, Wout: np.ndarray, tree: COBODTTree, x: np.ndarray, *,
                      align_threshold: float = 0.9, veto_threshold: float = -1.0, positive_threshold: float = 0.05) -> pd.DataFrame:
    """One row per second-layer unit: where it fires, and how its fan-in from the first layer builds that region.

    Columns: ``active`` (fraction of ``x`` where the unit is on), ``top_leaf`` and ``share`` (the leaf that holds most of
    its activity and that leaf's share), ``n_veto`` (first-layer rows with weight below ``veto_threshold``),
    ``vetoes_on_chain`` (how many of the top leaf's five wrong-side detectors are among the vetoed rows),
    ``veto_levels`` (levels of the vetoed detectors), ``positive_inputs`` (rows with weight above
    ``positive_threshold``, as detector labels), ``path`` (|W3[:, u] . Wout|). A first-layer row counts as the
    detector of node j with pole s when |cos(row, u_j)| >= ``align_threshold``.
    """
    U = tree.w_list; depth = int(np.log2(U.shape[0] + 1))
    node = np.zeros(len(x), dtype=int)
    for _ in range(depth):
        node = 2 * node + 1 + (np.einsum("nd,nd->n", x, U[node]) > 0).astype(int)
    leaf = node - (2 ** depth - 1); n_leaves = 2 ** depth
    C1 = (W1 / np.maximum(np.linalg.norm(W1, axis=1, keepdims=True), 1e-12)) @ U.T
    det = [(int(np.abs(c).argmax()), int(np.sign(c[np.abs(c).argmax()]))) if np.abs(c).max() >= align_threshold else None for c in C1]
    h1 = np.maximum(x @ W1.T, 0.0); h2 = np.maximum(h1 @ W2.T, 0.0)
    rows = []
    for u in range(W2.shape[0]):
        act = h2[:, u] > 0; cnt = np.bincount(leaf[act], minlength=n_leaves) if act.any() else np.zeros(n_leaves)
        top = int(cnt.argmax()) if act.any() else -1; chain = leaf_wrong_side_detectors(top, depth) if top >= 0 else []
        neg = [t for t in np.argsort(W2[u]) if W2[u, t] < veto_threshold]; pos = [t for t in np.argsort(-W2[u]) if W2[u, t] > positive_threshold]
        neg_det = [det[t] for t in neg if det[t] is not None]
        rows.append({"unit": u, "active": float(act.mean()), "top_leaf": top, "share": float(cnt.max() / act.sum()) if act.any() else np.nan, "n_veto": len(neg),
                     "vetoes_on_chain": int(sum(1 for v in chain if v in neg_det)), "veto_levels": "".join(str(int(NODE_LEVEL[n])) for n, _ in sorted(neg_det, key=lambda z: NODE_LEVEL[z[0]])),
                     "positive_inputs": ", ".join((f"{'+' if det[t][1] > 0 else '-'}u{det[t][0]}" if det[t] else f"row{t}") + f":{W2[u, t]:.2f}" for t in pos[:5]),
                     "path": float(abs(W3[:, u] @ np.asarray(Wout).ravel()))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# Tree flips and the 60-unit equivariant design (run 37)
# ---------------------------------------------------------------------------------------------------------------------


def mirror_node(n: int, j: int) -> int:
    """Image of node ``n`` under the flip at internal node ``j``: the two subtrees of ``j`` are exchanged (heap indexing), every
    other node, ``j`` included, is fixed. An involution."""
    if n == j:
        return n
    path = []; m = n
    while m > j:
        p = (m - 1) // 2; path.append(m - 2 * p); m = p          # 1 = left child (2p + 1), 2 = right child (2p + 2)
    if m != j:
        return n
    path = path[::-1]; path[0] = 3 - path[0]; m = j
    for c in path:
        m = 2 * m + c
    return m


def subtree_nodes(j: int, n_nodes: int = 31) -> list[int]:
    """Strict descendants of ``j`` in heap order."""
    out = []; frontier = [j]
    while frontier:
        m = frontier.pop(0)
        for c in (2 * m + 1, 2 * m + 2):
            if c < n_nodes:
                out.append(c); frontier.append(c)
    return out


def tree_flip_matrix(tree: COBODTTree, j: int) -> np.ndarray:
    """Orthogonal ``Q`` realising the flip at internal node ``j``: ``Q u_j = -u_j``, ``Q u_m = u_{mirror(m)}`` for the descendants
    ``m`` of ``j``, identity on every other direction (the other normals and the off-span complement). The label of the COB-ODT is
    invariant, ``y(Qx) = y(x)``, so ``Q`` is a symmetry of the data law; it acts on a first-layer row ``w`` as ``Q w``."""
    U = tree.w_list; n_nodes = U.shape[0]
    Q = np.eye(U.shape[1]) - 2.0 * np.outer(U[j], U[j])
    for m in subtree_nodes(j, n_nodes):
        Q += np.outer(U[mirror_node(m, j)], U[m]) - np.outer(U[m], U[m])
    return Q


def cue_row_identity(W1: np.ndarray, tree: COBODTTree, cue_rows: list[int] | np.ndarray) -> dict[tuple[int, int], int]:
    """``(node, pole) -> row`` for the cue rows, read off the in-span part of each row (largest |cosine| with a normal)."""
    U = tree.w_list; out = {}
    for r in cue_rows:
        c = W1[r] / np.linalg.norm(W1[r]) @ U.T; m = int(np.abs(c).argmax()); out[(m, int(np.sign(c[m])))] = int(r)
    if len(out) != len(cue_rows):
        raise ValueError("cue rows do not have distinct (node, pole) identities")
    return out


QUAD_TYPES = ((+1, +1), (+1, -1), (-1, -1), (-1, +1))
"""Unit types of the 60-unit design, in order within each node's quadruple: (cue-block sign, net path sign). Types 0 and 2 read the
+pole probe of the node, types 1 and 3 the -pole probe (the block sign times the path sign is the probe pole)."""


def equivariant_quadruple_weights(W1: np.ndarray, W2: np.ndarray, W3: np.ndarray, Wout: np.ndarray, *, cue_rows: list[int] | np.ndarray, tree: COBODTTree,
                                  orth_seed: int = 0) -> dict[str, Any]:
    """Build the 60-unit equivariant init (run 37) from a 30-unit hierarchy design (run 26's layout: probe rows 0..29 in
    ``HIERARCHY_TARGETS`` order feeding units 0..29, cue rows, the rest spare rows).

    * Probe and spare rows become mutually orthogonal, purely off-span vectors (drawn from ``orth_seed``), each keeping its norm.
    * Cue rows are kept as given.
    * Unit ``4j + t`` has cue block ``sign_t * B_j`` (``B_j`` the design's block for node ``j``, shared by the pair) and the probe
      weight of the design on the probe of pole ``sign_t * s_t`` (see ``QUAD_TYPES``).
    * W3: one common column for every net-path-sign ``+`` unit (the design's column of unit 0) and the complementary column for
      every ``-`` unit (the column of unit 1); every third-layer row then has ``width2 / 2`` nonzero entries. ``Wout`` is kept.
    """
    n1 = W1.shape[0]; U = tree.w_list; cue_rows = [int(r) for r in cue_rows]
    probe_rows = list(range(30)); spare_rows = [r for r in range(n1) if r not in probe_rows and r not in cue_rows]
    if (W2[:, spare_rows] != 0).any():
        raise ValueError("the design wires spare rows into W2; the quadruple construction assumes it does not")
    # orthogonal off-span directions for the probes and spare rows
    rng = np.random.default_rng(orth_seed); P = np.eye(U.shape[1]) - U.T @ U
    G = P @ rng.normal(size=(U.shape[1], len(probe_rows) + len(spare_rows))); Qo, _ = np.linalg.qr(G)
    if Qo.shape[1] != len(probe_rows) + len(spare_rows):
        raise ValueError("not enough off-span dimensions for orthogonal probe and spare rows")
    W1n = W1.copy()
    for k, r in enumerate(probe_rows + spare_rows):
        W1n[r] = Qo[:, k] * np.linalg.norm(W1[r])
    # second layer: quadruples
    W2n = np.zeros((60, n1), dtype=W2.dtype)
    for j in range(15):
        a, b = 2 * j, 2 * j + 1
        if not np.array_equal(W2[a, cue_rows], W2[b, cue_rows]):
            raise ValueError(f"the two units of node {j} do not share a cue block")
        B = W2[a, cue_rows]; wa, wb = W2[a, a], W2[b, b]
        if wa != wb or (W2[a] != 0).sum() != len(B[B != 0]) + 1 or (W2[b] != 0).sum() != len(B[B != 0]) + 1:
            raise ValueError(f"unexpected wiring of the units of node {j}")
        for t, (bsign, s) in enumerate(QUAD_TYPES):
            u = 4 * j + t; probe = a if bsign * s > 0 else b
            W2n[u, cue_rows] = bsign * B; W2n[u, probe] = wa
    # third layer: common pattern per path sign
    col_plus, col_minus = W3[:, 0], W3[:, 1]
    if not (((col_plus != 0) ^ (col_minus != 0)).all() and np.sign(col_plus @ Wout.ravel()) > 0 and np.sign(col_minus @ Wout.ravel()) < 0):
        raise ValueError("units 0 and 1 of the design must have complementary W3 columns with net path signs + and -")
    W3n = np.zeros((W3.shape[0], 60), dtype=W3.dtype)
    for j in range(15):
        for t, (_, s) in enumerate(QUAD_TYPES):
            W3n[:, 4 * j + t] = col_plus if s > 0 else col_minus
    unit_path_sign = np.array([s for _ in range(15) for (_, s) in QUAD_TYPES])
    return {"W1": W1n, "W2": W2n, "W3": W3n, "Wout": Wout.copy(), "probe_rows": probe_rows, "cue_rows": cue_rows, "spare_rows": spare_rows,
            "unit_path_sign": unit_path_sign, "unit_block_sign": np.array([b for _ in range(15) for (b, _) in QUAD_TYPES]), "unit_node": np.repeat(np.arange(15), 4)}


def flip_equivariance_report(W1: np.ndarray, W2: np.ndarray, W3: np.ndarray, Wout: np.ndarray, tree: COBODTTree, j: int, *, cue_rows: list[int] | np.ndarray,
                             units_per_node: int = 4) -> dict[str, float]:
    """Apply the tree flip at node ``j`` to an init and compare with the init itself, after relabelling rows and units.

    Rows: ``W1 -> W1 Q^T`` (each row ``w -> Q w``), then the probe rows of the descendants of ``j`` are relabelled to the mirror
    node, the two probe rows of ``j`` swap poles, the cue rows follow their node, spare rows stay. Units: the units of a descendant
    node move to the mirror node with the same type; the units of ``j`` itself exchange the sign of their cue block keeping their
    path sign (types 0 <-> 3, 1 <-> 2 of ``QUAD_TYPES``) when ``units_per_node`` is 4, and keep their index when it is 2 (the 30-unit
    design has no partner, which is the chirality the report exposes). Third-layer neurons and the output are fixed.

    Returns the largest absolute deviations: ``in_span`` (W1's in-span parts), ``off_span_gram`` (the Gram matrix of W1's off-span
    parts, invariant iff some rotation of the complement completes the symmetry), ``w2``, ``w3``, ``wout``.
    """
    U = tree.w_list; n1 = W1.shape[0]; cue_rows = [int(r) for r in cue_rows]
    Q = tree_flip_matrix(tree, j); cue_id = cue_row_identity(W1, tree, cue_rows)
    row_perm = np.arange(n1)
    for n in range(15):
        for q_idx, q in enumerate((+1, -1)):
            r = 2 * n + q_idx; n2 = mirror_node(n, j); q2 = -q if n == j else q
            row_perm[r] = 2 * n2 + (0 if q2 > 0 else 1)
    for (m, q), r in cue_id.items():
        row_perm[r] = cue_id[(mirror_node(m, j), q)]
    n_units = W2.shape[0]; unit_perm = np.arange(n_units)
    for u in range(n_units):
        n, t = divmod(u, units_per_node); n2 = mirror_node(n, j)
        t2 = {0: 3, 1: 2, 2: 1, 3: 0}[t] if (n == j and units_per_node == 4) else t
        unit_perm[u] = units_per_node * n2 + t2
    W1f = W1 @ Q.T; W1n = np.zeros_like(W1); W1n[row_perm] = W1f
    W2n = np.zeros_like(W2); W2n[np.ix_(unit_perm, row_perm)] = W2
    W3n = np.zeros_like(W3); W3n[:, unit_perm] = W3
    Pin = U.T @ U; Poff = np.eye(U.shape[1]) - Pin
    gram = lambda W: (W @ Poff) @ (W @ Poff).T
    return {"in_span": float(np.abs(W1n @ Pin - W1 @ Pin).max()), "off_span_gram": float(np.abs(gram(W1n) - gram(W1)).max()),
            "w2": float(np.abs(W2n - W2).max()), "w3": float(np.abs(W3n - W3).max()), "wout": float(np.abs(Wout - Wout).max())}
