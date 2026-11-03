---
name: init-sweep-u0-winner
description: Sep 28 2026 sweep result: which first-layer neuron captures u_0 is init-determined (35/36 runs) but decided during the plateau dynamics, not by static init features; tooling in init_sweep.py
metadata:
  type: project
---

Two-part sweep (`src/layer_lenses/init_sweep.py`, results in `results/init_sweep/`, summaries in `notebooks/results/init_sweep/`, notebook `notebooks/init_sweep_analysis.ipynb`): Part A, 6 inits x 3 data samples x 2 batch sizes; Part B, 200 inits with full trajectories, constant-LR SGD momentum to loss 0.3.

Findings: the first neuron to reach |cos| 0.95 with u_0 is the same across data samples and batch sizes in 35 of 36 runs; the one exception is a near-tie swapping by 400 steps. Near-ties within 500 steps occur in 65% of inits, so analyses use a leading set. The second capture is not init-determined. A different data sample perturbs the sub-dominant neurons visibly while batch size does not. Static init features predict the winner weakly (top-1 15% vs 2.5% chance; strongest: |cos| with u_0 at init, init gradient norm, lens magnitude); plateau-end displacement predicts it well (top-1 37%, in leading set 58%). The single-neuron anecdote that the winner has a small init gradient was wrong.

Follow-up (same day): the second-order autograd sensitivity of the alignment rate at init (`alignment_rate_sensitivity`, jvp over grad) is a weak predictor (top-1 19% with four dynamical features, 22% with static ones added). Rates evaluated over time show the winner becomes identifiable only between steps 5000 and 8000, a third to a half of the way to capture, during the exponential escape. The best early indicator is the acceleration of norm growth, not the alignment rate. Next candidate: the dominant growing mode of the flow linearized mid-plateau.

**Why:** Harish wants the winner explained by an init condition in the lens-theorem style. The result says the condition is dynamical, decided over the first ~50 epochs of flow, which points at the autograd-sensitivity approach.

**How to apply:** Treat the winner as an init property but look for it in the linearized/plateau dynamics, not in time-zero geometry. Tie handling matters. Runs cost ~42 s each at this config, so hundreds of inits are cheap. See [[user-research-direction]], [[relu-replayable-training]].
