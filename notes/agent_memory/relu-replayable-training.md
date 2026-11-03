---
name: relu-replayable-training
description: Replay/optimizer-snapshot tooling for ReLU runs, plus the sgd_adam hybrid; training primitives live in relu_training.py and replay in relu_analysis.py
metadata:
  type: project
---

Two related pieces of tooling, added 2026-09-22 and extended 2026-09-23:

- `src/layer_lenses/relu_training.py` owns the training primitives: `build_relu_optimizers`, `relu_training_step`, `build_relu_schedulers`, and the `"sgd_adam"` hybrid that trains hidden layers `[0, sgd_adam_split_layer)` with SGD-with-momentum and everything after with AdamW.
- `src/layer_lenses/relu_analysis.py` owns replay: `run_single_relu_seed_replayable`, `replay_epoch_from_snapshot`, `regenerate_seed_result_data`, `build_seed_model_and_data`, `epoch_replay_summary_table`. Seed results store `data_spec` instead of data arrays, because 1M x 100 floats made pickles exceed 1 GB.

**Why:** Harish wants the optimizer's contribution separated from the population gradient (see [[user-research-direction]]). An earlier version duplicated the training step inside relu_analysis; that duplication was removed on 2026-09-23 so there is now one definition of a step.

**How to apply:** When adding an optimizer variant, extend `build_relu_optimizers` and `optimizer_group_names` in relu_training; the replay path picks it up automatically. `sgd_lr` defaults to `lr`, which leaves SGD layers nearly frozen next to Adam ones. Always run `suggest_sgd_lr_for_adam_parity` first, and set `sgd_weight_decay=0.0` when `sgd_lr` is large, since SGD's decay is coupled and scales with the LR.

Repository size (Oct 10): replay folders are ~90 MB each (124 MB for AdamW runs); the user keeps the repo under 1 GB and no file over 100 MB. Runs 12-14 replay files are untracked (ignored in .gitignore, kept on disk and in ~/researchcodes/layer-lenses-replay-backup/); run 11 and runs 15-21 are in git (825 MiB tracked). Adding a new ~90 MB run will need another run dropped; priority is run 11, then the latest runs. Pre-rewrite history: local branch backup-pre-filter-20261010.
