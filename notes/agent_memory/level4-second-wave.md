---
name: level4-second-wave
description: Key empirical finding (Sep 2026): gradient flow learns ODT levels 0-3 and suppresses level 4; Adam's preconditioner starts a second wave that captures level 4 and gives its 10x loss advantage; noise is irrelevant
metadata:
  type: project
---

On the depth-5 COB-ODT ReLU task (seed 5178, [40,40,40]) the first layer's energy along the 16 deepest ODT normals goes through three phases: rise along the init gradient, suppression while coarse features are built, then a second rise that happens under Adam only. SGD, full-batch or mini-batch, never starts the second wave and ends with level-4 energy below init. Full-batch Adam with no noise matches mini-batch Adam and beats mini-batch SGD tenfold, so the difference is the vector field, not sampling noise. Full-batch and mini-batch SGD are identical to 4 decimals through the first feature captures when compared at matched flow time.

**Why:** This is the answer to Harish's question whether population gradient flow explains the phenomena. It does for levels 0-3 and for the order of capture; it does not reach level 4 on any timescale run so far. Details and script names are in the 2026-09-25 session_log entry; scripts in `notebooks/analysis_scripts/`.

**How to apply:** Treat "PGF captures the coarse hierarchy, Adam's per-coordinate normalization enables the fine one" as the working hypothesis. Suggested next experiments: sgd_adam hybrid with first layer on SGD; a much longer full-batch SGD run tracking level-4 energy. See [[user-research-direction]] and [[relu-replayable-training]].
