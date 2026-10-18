"""The alpha parametrisation reproduces the saved inits of runs 60 and 70 (W2 and Wout exactly, W1 up to the random off-span basis)."""
import pickle
from pathlib import Path

import numpy as np
import pytest

from layer_lenses.alpha_init import RUN_ALPHAS, build_alpha_init
from layer_lenses.init_sweep import SweepRunSpec, build_tree

RESULTS = Path(__file__).resolve().parents[1] / "results" / "designed_init"
RUNS = {60: "hd2_run60_sgd_lr1_1000", 70: "hd2_run70_symcue_partner", 68: "hd2_run68_symcue_outx1", 69: "hd2_run69_symcue_flipped"}


@pytest.mark.parametrize("run", [60, 68, 69, 70])
def test_alpha_init_reproduces_saved_init(run):
    folder = RESULTS / RUNS[run] / "replay"
    if not (folder / "weights.npz").exists():
        pytest.skip("replay weights not present")
    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1))
    z = np.load(folder / "weights.npz"); meta = pickle.load(open(folder / "meta.pkl", "rb"))
    W1, W2, Wo = (z[k][0].astype(np.float64) for k in ["w__hidden_layers.0.weight", "w__hidden_layers.1.weight", "w__output_layer.weight"])
    out = build_alpha_init(RUN_ALPHAS[run], tree)
    # the saved cue rows are identified the same way (row -> (leaf, pole)); compare in-span parts of W1 and the probes' norms / orthogonality
    U = tree.w_list
    np.testing.assert_allclose(out["W1"][30:] @ U.T, W1[30:] @ U.T, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(out["W1"][:30], axis=1), np.linalg.norm(W1[:30], axis=1), atol=1e-5)
    np.testing.assert_allclose(np.abs(out["W1"][:30] @ U.T).max(), 0.0, atol=1e-6)
    np.testing.assert_allclose(out["W1"][:30] @ out["W1"][:30].T, W1[:30] @ W1[:30].T, atol=1e-5)
    np.testing.assert_allclose(out["W2"], W2, atol=2e-6)
    np.testing.assert_allclose(out["Wout"], Wo, atol=1e-4)
    assert out["update_masks"]["output_layer.weight"][0, 0] == pytest.approx(80.0 if run == 60 else 1.0)
