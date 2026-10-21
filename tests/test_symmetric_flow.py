"""Orbit structure: 157 free orbits for the 62-60 design; symmetrising an equivariant init leaves it unchanged; symmetrising a random perturbation projects it."""
import pickle
from pathlib import Path

import numpy as np
import pytest

from layer_lenses.designed_init import cue_row_identity
from layer_lenses.init_sweep import SweepRunSpec, build_tree
from layer_lenses.symmetric_flow import symmetric_orbits, symmetrise, mirror

RUN = Path(__file__).resolve().parents[1] / "results" / "designed_init" / "hd2_run70_symcue_partner" / "replay"


def test_mirror():
    assert mirror(0, 7) == 11 and mirror(0, 1) == 2 and mirror(1, 3) == 4 and mirror(7, 15) == 16 and mirror(0, 0) == 0 and mirror(3, 1) == 1


def test_orbits_and_projection():
    if not (RUN / "weights.npz").exists():
        pytest.skip("run 70 replay not present")
    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); U = tree.w_list
    z = np.load(RUN / "weights.npz"); meta = pickle.load(open(RUN / "meta.pkl", "rb")); d = meta["design"]
    W1, W2, Wo = (z[k][0].astype(np.float64) for k in ["w__hidden_layers.0.weight", "w__hidden_layers.1.weight", "w__output_layer.weight"])
    cid = cue_row_identity(W1, tree, d["cue_rows"]); units = list(zip(d["unit_node"], d["unit_block_sign"], d["unit_path_sign"]))
    orb = symmetric_orbits(cid, units)
    assert orb["n_free"] == 157 and orb["n_orbits"] == 203
    O = W1[:30] - (W1[:30] @ U.T) @ U; O /= np.linalg.norm(O, axis=1, keepdims=True); B = np.vstack([U, O])
    C = W1 @ B.T                                                                   # 62 x 61 coefficients
    for block, val in (("W1", C), ("W2", W2), ("Wout", Wo.ravel())):
        o, s = orb[block]; sym = symmetrise(val, o, s, orb["forced_zero"], orb["n_orbits"])
        np.testing.assert_allclose(sym, val, atol=1e-6, err_msg=f"{block} of the equivariant init is not fixed by the projection")
    rng = np.random.default_rng(0); R = rng.standard_normal(C.shape); P = symmetrise(R, *orb["W1"], orb["forced_zero"], orb["n_orbits"])
    np.testing.assert_allclose(symmetrise(P, *orb["W1"], orb["forced_zero"], orb["n_orbits"]), P, atol=1e-12)     # a projection
    assert np.linalg.norm(P) < 0.3 * np.linalg.norm(R)                                                               # most of a random direction is removed


def test_orbits_with_cue_off_span_directions():
    """With one off-span direction per row (62), the W1 block has 93 columns, 227 orbits / 181 free in all (24 new, all free: 14 probe x
    cue-direction and 10 cue x cue-direction), and the projection fixes an alpha_13 = 0.4 init."""
    from dataclasses import replace
    from layer_lenses.alpha_init import RUN_ALPHAS, build_alpha_init
    from layer_lenses.designed_init import cue_row_identity
    from layer_lenses.init_sweep import SweepRunSpec, build_tree
    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); U = tree.w_list
    b = build_alpha_init(replace(RUN_ALPHAS[60], alpha_13=0.4), tree); W1 = b["W1"]
    cue_rows = [b["cue_rows"][k] for k in sorted(b["cue_rows"], key=lambda k: b["cue_rows"][k])]
    cid = cue_row_identity(W1, tree, cue_rows); units = [(u["node"], u["block_sign"], u["path_sign"]) for u in b["units"]]
    orb = symmetric_orbits(cid, units, off_rows=list(range(62)))
    assert orb["W1"][0].shape == (62, 93) and orb["n_orbits"] == 227 and orb["n_free"] == 181
    B = np.vstack([U, b["off_span_directions"]]); C = W1 @ B.T
    np.testing.assert_allclose(C @ B, W1, atol=1e-9)
    np.testing.assert_allclose(symmetrise(C, *orb["W1"], orb["forced_zero"], orb["n_orbits"]), C, atol=1e-9)
    for block, val in (("W2", b["W2"]), ("Wout", b["Wout"].ravel())):
        np.testing.assert_allclose(symmetrise(val, *orb[block], orb["forced_zero"], orb["n_orbits"]), val, atol=1e-9)
