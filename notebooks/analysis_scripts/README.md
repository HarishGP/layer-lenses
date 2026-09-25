# Analysis scripts (Sep 22–25, 2026)

Throwaway scripts written by the coding agent during the sessions logged in
`notes/session_log.md`. Every number quoted in those sessions was printed by one of
these. They are evidence, not reusable code: hard-coded paths, no argument parsing,
some redundant pickle reloads.

Run from the `notebooks/` directory with the project interpreter, because the
scripts refer to `results/cache/...` relatively:

```bash
cd notebooks && ../.venv/bin/python analysis_scripts/l4_time.py
```

Inputs are the replayable pickles in `notebooks/results/cache/`; the data split is
regenerated from its seed where needed (about a minute, ~1.6 GB RAM).

| script | what it produced |
|---|---|
| `smoke.py`, `hybrid.py`, `capture_ref.py` | correctness checks for the replay tooling; bitwise-parity hashes for the `train_relu_mlp` refactor (`capture_ref.py <out.json>` twice, diff the outputs) |
| `first_step.py`, `precision.py` | Adam's first step is a sign step; float32 differencing floor on recorded SGD updates |
| `n37.py`, `n37b.py`, `n37c.py` | neuron 37 at epochs 140 and 75: along/perpendicular decomposition, restoring-force test, wobble period, out-of-plane detour during arrival |
| `escape.py`, `escape2.py` | plateau escape direction vs the population gradient at init; level-4 content of that gradient |
| `fb_vs_mb.py`, `fb_vs_mb2.py` | full-batch vs mini-batch SGD on a common flow-time axis; first divergence; diverged neurons |
| `adam_l4.py` | all runs: end metrics, first-layer ODT coverage per level, test error binned by nearest on-path split level; full-batch Adam level-4 capture over time |
| `rowspace.py` | fraction of each ODT normal inside the first-layer row space; per-level response strength |
| `l4_time.py` | first-layer energy per ODT level over training, all four runs (the three-phase level-4 picture) |
| `grad_vs_update.py` | one replayed full-batch step: level-4 share of gradient vs update under SGD and Adam |
