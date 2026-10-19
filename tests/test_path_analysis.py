import numpy as np

from layer_lenses.init_sweep import SweepRunSpec, build_data, build_tree
from layer_lenses.path_analysis import leaf_cells, path_statistics, path_group_summary


def test_path_statistics_shapes_and_cells():
    tree = build_tree(SweepRunSpec(tree_seed=5178, data_seed=0, init_seed=0, batch_size=1)); U = tree.w_list
    x, y = build_data(SweepRunSpec(tree_seed=5178, data_seed=7, init_seed=0, batch_size=1, n_train=3000), tree)
    cells = leaf_cells(x, U); assert cells.min() >= 0 and cells.max() <= 31 and np.all((cells % 2 == 1) == (y > 0))     # odd cell = label +1
    rng = np.random.default_rng(0); W1 = rng.standard_normal((62, 100)) * 0.2; W2 = rng.standard_normal((6, 62)) * 0.3; Wo = rng.standard_normal((1, 6))
    row_node = np.abs(W1 @ U.T).argmax(1); row_pole = np.sign((W1 @ U.T)[np.arange(62), row_node]).astype(int); unit_node = np.arange(6)
    ps = path_statistics(W1, W2, Wo, x, y.astype(float), U, row_node=row_node, row_pole=row_pole, unit_node=unit_node)
    assert len(ps) == 6 * 62 and set(ps.relation) <= {"own", "ancestor", "descendant", "unrelated"} and (ps.active_frac <= 1).all()
    assert len(path_group_summary(ps)) > 0
