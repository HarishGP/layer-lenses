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
