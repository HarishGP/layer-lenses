---
name: path-sign-consensus
description: Path-level version of "output weights never flip" measured on run 26 (2026-10-12): per-unit sign consensus, W2 signs follow row identity
metadata:
  type: project
---

Run 26 (the user's anchor run as of 2026-10-12) has 96 000 three-edge paths, 5720 nonzero at init; 30% of those flip
sign by the end, the 40 output weights never do. Flips come from three classes: W3 minority edges (one-quantum paths
against the unit's net sign: 80 of 150 go through zero, the rest shrink), W2 probe/cue edges pruned through zero, and
12 cue edges that re-grow with the opposite sign because the cue climbed to an ancestor node. Balancedness invariants
drift a lot under momentum SGD, so the persistence is dynamical, not conserved.

**Why:** the user asked how much of the output-weight argument (dL/dw_k = E[(p-y) h_k], feature built through w_k)
extends to paths; the answer is a consensus rule: an edge keeps its sign iff the feature it reads was built with that
sign, so each unit's paths converge to one sign and a W2 edge's sign is tied to the identity of its row.

**How to apply:** tooling is `edge_sign_flips`, `path_products`, `path_sign_summary`, `unit_path_sign_consensus` in
relu_analysis.py and section 2 of hierarchy_cross_run_analysis.ipynb; the user may want the same bookkeeping on runs
11 and 21-25 (dense-family units of run 23) and a step-level replay of the minority flips. See [[designed-init-lens-lesson]].
