---
name: user-research-direction
description: Harish is building a lens-level theory of feature learning in multilayer ReLU nets under population gradient flow; Adam's extra effect (level-4 second wave) is deliberately deferred
metadata:
  type: user
---

Harish (IIT Madras, DSAI) is a researcher building a theory of feature learning across multiple layers of ReLU nets on COB-ODT data. As of 2026-09-25 the scope is fixed: analyse population gradient flow (PGF) from random init, and defer the analysis of what Adam adds. Even a lens-level understanding of why PGF works in multilayer networks is still open, and that is the target.

**Why:** The Sep 24-25 analyses showed that SGD with momentum, full-batch or mini-batch, follows PGF to four decimals through the first feature captures, so PGF is the right object for the coarse hierarchy. Adam's advantage is a separate mechanism (see [[level4-second-wave]]) and Harish chose to set it aside rather than let it widen the theory.

**How to apply:** Frame analysis around the population gradient at checkpoints, not per-step optimizer behaviour. Do not steer toward Adam or mini-batch noise unless asked. Useful facts for the PGF analysis: the linearized flow at init points at the deepest ODT level, so coarse-level capture is a nonlinear effect starting around epoch 40 in the seed-5178 runs; within the arrival window the population gradient is constant across an epoch. Tooling in [[relu-replayable-training]].
