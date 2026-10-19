"""The shared run-analysis helpers load a replay run and produce the end-of-run table (plots are not exercised)."""
from pathlib import Path

import pytest

from layer_lenses.run_analysis import AnalysisContext, findings_table, load_replay_run, logits_at

RESULTS = Path(__file__).resolve().parents[1] / "results" / "designed_init"


def test_load_and_findings():
    if not (RESULTS / "hd2_run60_sgd_lr1_1000" / "replay" / "weights.npz").exists():
        pytest.skip("run 60 replay not present")
    ctx = AnalysisContext(RESULTS, n_test=5_000)
    sr, run = load_replay_run(ctx, "hd2_run60_sgd_lr1_1000", [62, 60])
    assert run["W"]["hidden_layers.0.weight"].shape[1:] == (62, 100) and len(run["losses"]) == 1000
    rec, s = findings_table(ctx, run)
    assert rec["final epoch loss"] == pytest.approx(run["losses"][-1], abs=1e-4) and s["nodes_covered"] == 31
    assert logits_at(ctx, run, 1000).shape == (5_000,)
