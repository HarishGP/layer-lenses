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
    layer3_scale: float = 1 / np.sqrt(20)
    output_scale: float = 1 / np.sqrt(40)
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
    W2 *= spec.layer2_scale
    W3 = (rng.random((spec.width3, spec.width2)) < spec.layer3_density).astype(np.float64) * spec.layer3_scale
    Wout = np.full((1, spec.width3), spec.output_scale)
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
    meta = {"unit_probe": unit_probe, "labels": hierarchy_labels(spec), "targets": HIERARCHY_TARGETS,
            "cue_rows": list(range(spec.cue_first_row, spec.cue_first_row + 2 * NUM_LEVEL4))}
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
