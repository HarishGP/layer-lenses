# Session Log

Append-only log of agent working sessions. Newest entries go at the **bottom**.

## Purpose

A fresh agent (or a future you) should be able to read the most recent
few entries and know:
- what was being worked on,
- what changed in the repo and why,
- what's still open, and
- the exact framing the user used (via preserved prompts).

This file complements, not replaces:
- `AGENTS.md` — stable project conventions.
- `notes/research_spec.md` — project goals and invariants.
- `notes/paper_mismatches.md` — durable paper-vs-notebook decisions.
- `notes/reproducibility_log.md` — hyperparameter provenance per config.

Code, tests, and configs remain the source of truth for *what currently
runs*. This file captures the *narrative and intent* around changes.

## Entry format

```
## YYYY-MM-DD HH:MM IST — <short session title>

### Goal
One or two lines on what the session set out to do.

### User prompts (verbatim)
- "..."
- "..."

### Changes
- Files touched and a brief note on each.

### Decisions
- Non-obvious decisions, with links to paper_mismatches.md or
  reproducibility_log.md if applicable.

### Current state / where to pick up
- What's working, what's half-done, what's next.

### Open questions
- Anything that needs the user's input or further investigation.
```

Keep entries short. Use IST for all timestamps. If exact historical
times are unknown, keep the date and mark the heading as `(IST)` without
an HH:MM time. Always append new entries at the end of this file only
(never insert between existing entries). If a decision is durable
(paper vs notebook, a
hyperparameter choice, a naming convention), migrate it into the
appropriate long-lived note and link to it from here.

---

## 2026-04-22 (IST) — Retroactive snapshot of repo state

This entry predates the session-log policy and is written from a
cold read of the repo (git log, source files, notes, tests). It is a
best-effort reconstruction, not a transcript. Correct as needed.

### Current state / where to pick up

**Milestone 1 target:** reproduce the shared "DLGN gating hyperplanes
cluster around COB-ODT decision hyperplanes" phenomenon from
Paper 1 (Table 2 setting) and Paper 2 (Fig 2 style scatter).
See `notes/reproducibility_log.md` §`configs/milestone1_dlgn_sf_odt.yaml (planned)`.

**Implemented in `src/first_experiment/`:**
- `odt.py` — COB-ODT synthetic data generation. Unit-sphere `x`,
  QR-orthonormal decision normals, zero bias, deterministic leaf-sign
  assignment by index parity with optional global flip. Includes
  `samples_reaching_node` path utility.
- `dlgn.py` — `DLGNSF` (Paper 2's DLGN / Paper 1's DLGN-SF). Gating
  `sigmoid(beta * V_l x)`, `value_input_mode in {"ones", "x"}`, same
  value-network parameterization in both modes, configurable
  gating/value init scales. Final logit is multiplied by 2 to mimic the
  legacy notebook's `[-z, z]` + cross-entropy behavior under BCE.
- `training.py` — `TrainConfig` (Adam, BCE-with-logits, lr=2e-3,
  batch_size=1024, epochs=200, seed=365 by default), `train_dlgn_sf`
  with per-epoch shuffling, `evaluate_dlgn_sf` (log loss, 0/1 loss,
  accuracy), full-state-dict checkpoint snapshots at
  `snapshot_epochs`, and `effective_gating_weights_from_checkpoint`.
- `parity_debug.py` — utilities for comparing current code against
  the legacy notebook. Used to confirm parity.

**Tests (`tests/`):**
- `test_dlgn_training.py` — forward-shape sanity, parameter-count
  parity across value-input modes, init-scale math, smoke train-loss
  decrease, eval metric shape.
- `test_odt.py` — COB-ODT data-generation checks.

**Notebooks:**
- `notebooks/legacy_dlgn_minimal.ipynb` — minimal legacy reproduction
  (reference).
- `notebooks/scratch.ipynb` — current exploratory workspace (has
  uncommitted changes at the time of this snapshot).
- `notebooks/old_notebooks/` — read-only legacy reference per `AGENTS.md`.

**Configs:** `configs/` is empty. The milestone-1 YAML described in
`reproducibility_log.md` is planned but not yet written.

**Recent git history (most recent first):**
- `f837f83` — testing legacy vs current code variation across seeds (2).
- `23ef54c` — testing legacy vs current code variation across seeds.
- `af0d01e` — reproduced DLGNSF comparable with legacy notebook:
  depth-5 ODT, 80k samples, ~10% error in phase 1.
- `9867ea5` — scaffolding updates.
- `637ad6a` — initial DLGN model + training; scratch notebook testing
  ongoing.
- `3213db1` — ODT path utilities (`samples_reaching_node`).
- `6e5745c` — ODT data-gen code added.
- `3d6b6b6` — milestone-1 scaffolding and reproducibility notes.

### Open questions / known gaps

- No `configs/milestone1_dlgn_sf_odt.yaml` yet; training is currently
  driven from notebooks rather than from a committed config file.
  This is the natural next milestone-1 step.
- No CLI entry point (`AGENTS.md` calls for config-driven runs from the
  command line).
- No analysis-output code yet for `figures/scatter_init.pdf`,
  `figures/scatter_final.pdf`, or the Paper 1 Table 2–style
  `distance_table.csv`.
- `notebooks/scratch.ipynb` and `src/first_experiment/dlgn.py` have
  uncommitted changes as of this snapshot; their intent is not captured
  in git history yet.

### User prompts (verbatim)

None captured for sessions predating this policy. Cursor's local
transcripts in `~/.cursor/projects/.../agent-transcripts/` remain the
fallback for older sessions.

---

## 2026-04-22 (IST) — Establish session-log policy

### Goal
Set up a durable, in-repo context-preservation mechanism so that a new
agent can come up to speed after a Cursor reinstall or agent
termination.

### User prompts (verbatim)
- "This project was built with a previous Cursor agent's help. But an update crashed it and I had to reinstall cursor. I suppose that would leave you with little context on the state of the project. You can look at the markdown notes in the notes folder to get a sense of the project. You can read any other files in the project to build your context.\n\nTo prevent something like this happening again, can you create a conversations.md file in the notes folder with the chats I have with you and a brief summary of your text response (if not the full response)? Is that a good way to maintain the context with AI coding agents, that can be used for getting a new agent upto speed if the old agent has been terminated due to some issue? Or is there a better way? The reprodicibility_log notebook already serves a similar purpose."
- "Ok session_log.md is fine. But do maintain the exact text of my inputs in the chat, even if you don't store your chat outputs, which could be a lot of noise. My text inputs should not be very large even for a long project, simply by human limitations. Any existing note or a new note can be used for this log."

### Changes
- Created `notes/session_log.md` (this file) with format, policy, a
  retroactive state snapshot of the repo as of 2026-04-22, and this
  entry.
- Updated `AGENTS.md` with a working-style rule: agents must append a
  session-log entry (including verbatim user prompts) at the end of any
  non-trivial session.

### Decisions
- **Do not** keep a verbatim log of agent responses. Chat transcripts
  are low-signal-per-token and Cursor already stores them locally.
- **Do** keep verbatim user prompts. They are bounded in volume by
  human input and they preserve framing and intent — the most useful
  signal for a cold-start agent.
- Combine narrative state and verbatim prompts in a single file
  (`session_log.md`) rather than splitting across two files; split
  later if it grows unwieldy.

### Current state / where to pick up
- Policy is set up. Next time a non-trivial code change is made, append
  an entry at the bottom of this file following the format above.
- Milestone-1 work itself is unchanged; see the retroactive snapshot
  above for where that stands.

### Open questions
- None for this meta-session.

---

## 2026-04-22 (IST) — Fix legacy notebook log loss

### Goal
Make minimal edits to the legacy notebook so loss is computed directly
as binary cross-entropy on scalar logits, instead of first converting
to 2-class logits `[-y, y]` for cross-entropy.

### User prompts (verbatim)
- "Can you make minimal changes to the @notebooks/legacy_dlgn_minimal.ipynb  notebook that only fixes the log loss computation? Instead of making an output y into a [-y, y] before computing cross-entropy loss, make a direct binary cross-entropy loss from the model ouptut y. Change all places that use the cross-entropy loss with this modification."

### Changes
- Updated notebook cell 4 (`give_loss`) to use
  `nn.BCEWithLogitsLoss()` directly on `preds`, with float targets
  reshaped to `(N, 1)`.
- Updated notebook cell 13 (`train_dlgn`) to replace all
  `nn.CrossEntropyLoss()` usage and all `torch.cat((-1*..., ...), dim=1)`
  constructions with direct `nn.BCEWithLogitsLoss()` on scalar logits.
- Kept edits minimal and localized to the loss computation path only.

### Decisions
- Targets are cast to `float32` and reshaped to match model logit shape
  for BCE-with-logits compatibility.
- No other notebook logic (model architecture, data generation,
  evaluation/error plotting) was changed.

### Current state / where to pick up
- `notebooks/legacy_dlgn_minimal.ipynb` now has no remaining
  `CrossEntropyLoss` usage and no `[-y, y]` logits construction in the
  training/loss code paths.

### Open questions
- None.

---

## 2026-05-07 13:30 IST — DLGN scatter paper fonts

### Goal
Make DLGN checkpoint scatter plots more readable in paper artifacts.

### User prompts (verbatim)
- "In the DLGN scatter plots, can you increase the font size of the x and y ticks and the axis titles? You can reduce the number of ticks to just 2 or 3."

### Changes
- Updated `plot_scatter_for_checkpoint(...)` in
  `notebooks/DLGN_multiseed_multipath_runner.ipynb`.
- Increased x/y axis label font sizes and tick-label sizes.
- Limited x/y major ticks using `MaxNLocator(nbins=3)`.
- Set an explicit title font size.

### Decisions
- Kept the inset viridis ODT-level colorbar and figure-return behavior unchanged.

### Current state / where to pick up
- Notebook edited only; JSON parsing was validated.

### Open questions
- None.

---

## 2026-05-07 11:57 IST — DLGN scatter colors by ODT level

### Goal
Make DLGN checkpoint scatter plots use ODT node level for color, reducing the
palette from one color per internal node to one color per tree level.

### User prompts (verbatim)
- "In the DLGN scatter plot can you make it so that instead of coloring a node based on the node index, we just color based on the ODT node level? That way we need only 5 colors and not 31. Make changes to the plot_scatter_for_checkpoint function."

### Changes
- Updated `plot_scatter_for_checkpoint(...)` in
  `notebooks/DLGN_multiseed_multipath_runner.ipynb`.
- Converted closest ODT node ids to tree levels via
  `floor(log2(closest + 1))`.
- Switched the scatter color scale to a discrete level-based colormap and
  changed the colorbar label to "Closest ODT node level".

### Decisions
- Derive the number of levels from `odt_normals.shape[0]` rather than hardcoding
  5, while preserving the intended 5-color behavior for depth-5 ODTs.

### Current state / where to pick up
- Notebook edited only; JSON parsing was validated.

### Open questions
- None.

---

## 2026-05-07 12:02 IST — DLGN scatter viridis levels

### Goal
Use a viridis color scale for DLGN scatter plot ODT levels, with root level in
purple and deepest level in yellow.

### User prompts (verbatim)
- "Can you make the colormap be viridis purple for 0 and yelllow for 4?"

### Changes
- Updated `plot_scatter_for_checkpoint(...)` in
  `notebooks/DLGN_multiseed_multipath_runner.ipynb` to use
  `plt.get_cmap("viridis", num_levels)`.

### Decisions
- Keep the discrete level normalization so level `0` maps to the low end of
  viridis and level `num_levels - 1` maps to the high end.

### Current state / where to pick up
- Notebook edited only; JSON parsing was validated.

### Open questions
- None.

---

## 2026-05-07 12:47 IST — DLGN scatter inset colorbar

### Goal
Move the DLGN scatter plot colorbar inside the axes to save horizontal space.

### User prompts (verbatim)
- "Can you bring the colorbar inside the plot to save space?"

### Changes
- Updated `plot_scatter_for_checkpoint(...)` in
  `notebooks/DLGN_multiseed_multipath_runner.ipynb` to use explicit `fig, ax`.
- Replaced the external `plt.colorbar(...)` with an inset axes colorbar using
  `ax.inset_axes(...)`.

### Decisions
- Keep the same discrete viridis ODT-level color scale and ticks.

### Current state / where to pick up
- Notebook edited only; JSON parsing was validated.

### Open questions
- None.

---

## 2026-05-07 12:51 IST — DLGN scatter return figure

### Goal
Make DLGN checkpoint scatter plots easy to save as paper artifacts.

### User prompts (verbatim)
- "Can you have the plot_scatter_for_checkpoint return a fig object so that I can save them as pdf artifacts for a paper?"

### Changes
- Updated `plot_scatter_for_checkpoint(...)` in
  `notebooks/DLGN_multiseed_multipath_runner.ipynb` to return the Matplotlib
  `fig` after displaying it.

### Decisions
- Preserve existing notebook display behavior with `plt.show()` while also
  returning the figure for `fig.savefig(...)`.

### Current state / where to pick up
- Notebook edited only; JSON parsing was validated.

### Open questions
- None.

---

## 2026-05-07 13:10 IST — ReLU scatter figure export

### Goal
Make ReLU first-layer ODT alignment scatter plots space-efficient and easy to
save as PDF artifacts.

### User prompts (verbatim)
- "In ReLU analysis and runner can you do the same? Move the colorbar for the scatterplot inside, and make it return a fig for saving pdf artifacts."

### Changes
- Updated `plot_first_layer_odt_alignment(...)` in
  `src/layer_lenses/relu_analysis.py`.
- Replaced the external colorbar with an inset axes colorbar.
- Changed the return value to `(alignment, fig, ax)`.
- Updated `notebooks/ReLU_runner.ipynb` and `notebooks/scratch.ipynb` call sites
  to unpack `fig`.

### Decisions
- Preserve notebook display behavior by leaving `plt.show()` in the notebooks.
- Update all repository call sites found for the changed return shape.

### Current state / where to pick up
- `src/layer_lenses/relu_analysis.py` has no linter errors.
- Edited notebooks parse as valid JSON.

### Open questions
- None.

---

## 2026-05-01 10:38 IST — Rename package to layer-lenses

### Goal
Rename the project/package from the generic `first-experiment` / `first_experiment`
name to `layer-lenses` / `layer_lenses`.

### User prompts (verbatim)
- "Is there an easy way we could rename the project? \"first-experiment\" is too generic and says nothing about the project. I started it to test out cursor and github and now it has grown quite a bit. 

The project name is there in all python files as an import. Is it too much hassle to change everything? If not, abetter name would be \"layer-lenses\" which is the name of the repo in Github too."
- "Please go ahead and do the rename."

### Changes
- Renamed source package directory from `src/first_experiment/` to `src/layer_lenses/`.
- Updated imports across source files, tests, and active notebooks from
  `first_experiment` to `layer_lenses`.
- Updated project metadata in `pyproject.toml` and `uv.lock` from
  `first-experiment` to `layer-lenses`.
- Updated active documentation references in `README.md`,
  `README_detailed.md`, `AGENTS.md`, and `notes/research_spec.md`.

### Decisions
- Use `layer-lenses` for the distribution/project name and `layer_lenses`
  for the Python import package, since Python imports cannot contain hyphens.
- Left older `first_experiment` references in historical `notes/session_log.md`
  entries untouched except for this new entry.
- Did not rename the local workspace directory (`.../first-experiment`) to avoid
  disrupting the active Cursor workspace.

### Current state / where to pick up
- Tests pass after the rename: `24 passed`.
- No non-historical occurrences of `first_experiment`, `first-experiment`, or
  `First Experiment` remain outside `notes/session_log.md`.

### Open questions
- None.

---

## 2026-05-01 10:33 IST — Add repository README documentation

### Goal
Create project documentation explaining the repository organization, with a concise README and a detailed module/function map.

### User prompts (verbatim)
- "Can you create a a short text for README.md explaining how the files are organised in the repository etc?

Maybe also, add a second more detailed README_detailed.md with even more detailes, e.g. Include a short description of all functions in all the python files etc."

### Changes
- Added root `README.md` with project purpose, repository layout, main code areas, quick start, and source-of-truth notes.
- Added `README_detailed.md` with a detailed directory map and a module/function overview for all files under `src/first_experiment/`.

### Decisions
- Kept `README.md` short and navigational.
- Put the comprehensive function/class inventory in `README_detailed.md` to avoid making the root README too long.

### Current state / where to pick up
- Documentation reflects the current source modules, notebooks, notes, tests, and intended `results/` output root.

### Open questions
- None.

---

## 2026-04-30 19:36 IST — Add neuron-to-leaf activity-vector helper

### Goal
Add a reusable helper in `relu_analysis.py` that converts `neuron_leaf_activity_summary` into a dictionary mapping each neuron to a leaf-ordered activity vector.

### User prompts (verbatim)
- "Can you write a piece of code that takes neuron_leaf_activity_summary as input and creates dictionary with each key corresponding to a neuron, and value corresponding to a numpy array of length equal to the number of leaf nodes in the ODT? Each element in this array should correspond to the total points in the leaf that are active for that neuron."
- "PLease do so, and put it in the @src/first_experiment/relu_analysis.py file."

### Changes
- Updated `src/first_experiment/relu_analysis.py` with:
  - `neuron_leaf_activity_dict(neuron_leaf_activity_summary, *, key_mode='global')`
    returning `(mapping, leaf_order)`.
  - Supports `key_mode='global'` and `key_mode='layer_local'`.
  - Validates required input columns and handles empty input safely.

### Decisions
- Return both mapping and `leaf_order` to make the vector index-to-leaf correspondence explicit and robust in downstream analysis.

### Current state / where to pick up
- Use:
  - `mapping, leaf_order = neuron_leaf_activity_dict(neuron_leaf_activity_summary)`
  - or `key_mode='layer_local'` for `(layer, local_neuron_id)` keys.
- Lints are clean and import check via `uv run python` succeeded.

### Open questions
- None.

---

## 2026-04-30 19:11 IST — Add neuron activity count by ODT leaf

### Goal
Add a reusable helper for counting neuron activity within a specified ODT leaf and use it in the scratch notebook to build an all-neuron/all-leaf activity summary.

### User prompts (verbatim)
- "Write a function which takes a model checkpoint, a neuron id, and a ODT label leafnode id, and gives the total number of points with that label leaf node id for which that neuron is active. Have this function be in relu_analysis.py

Use this function in scratch.ipynb to get an appropriate summary object for all neurons and ODT nodes. Insert this cell just before the start/end metrics cell."

### Changes
- Updated `src/first_experiment/relu_analysis.py`:
  - Added `count_active_points_for_neuron_leaf(...)` to count active points for a given hidden neuron and leaf node.
  - Added supporting internal helpers for precomputing hidden activation masks / leaf IDs and resolving neuron IDs.
  - Added `summarize_neuron_leaf_activity(...)` convenience helper.
- Updated `notebooks/scratch.ipynb`:
  - Inserted a new code cell immediately before the `## Start/end metrics` section that loops all hidden neurons and ODT leaf nodes and builds `neuron_leaf_activity_summary` using `count_active_points_for_neuron_leaf`.
  - Left two previously inserted duplicate cells as no-op comments indicating that the summary moved.

### Decisions
- Accept both global neuron ids (`0..119`) and `(layer_idx, local_neuron_idx)` tuples in the counting helper for flexibility.
- Use leaf IDs in ODT heap indexing (`31..62` for depth 5), matching current notebook conventions.

### Current state / where to pick up
- Run the new pre-start-metrics cell to populate `neuron_leaf_activity_summary`.
- Use the resulting dataframe directly or with `itables.show(...)` for interactive filtering.

### Open questions
- None.

---

## 2026-04-30 16:53 IST — Extract ReLU notebook analysis helpers

### Goal
Move reusable functions out of `notebooks/scratch.ipynb` into importable source code while keeping plots and exploratory analysis in the notebook.

### User prompts (verbatim)
- "I want the plots and analyses to be on the scratch notebook, but would like to move all functions to an appropriate (maybe new or one of the old ones) .py file. \n\nIf any other piece of code in the notebook (not yet as a function) can also be moved into a function that would also be good.\n\nCan you do this reorganisation? "

### Changes
- Added `src/first_experiment/relu_analysis.py` with reusable ReLU/COB-ODT helpers:
  - single-seed ReLU run setup,
  - start/end metric collection,
  - checkpoint model loading,
  - first-layer ODT alignment computation and scatter plotting,
  - layered ReLU graph plotting,
  - masked ReLU forward/eval by ODT leaf,
  - single-neuron ablation responsibility analysis,
  - compact responsibility summary generation.
- Updated `notebooks/scratch.ipynb` to import those helpers and replaced notebook function definitions with calls.
- Replaced inline scatter/graph mechanics with calls to `plot_first_layer_odt_alignment` and `plot_layered_relu_graph`.

### Decisions
- Created a new `relu_analysis.py` rather than extending `analysis.py`, because existing `analysis.py` is focused on multipath DLGN aggregate statistics.
- Kept plotting calls in the notebook, but moved plotting implementations into source so the notebook remains a thin analysis driver.

### Current state / where to pick up
- Re-run the first import/config cell in `notebooks/scratch.ipynb` before running downstream analysis cells.
- `relu_analysis.py` imported successfully through `uv run python`; IDE lints report no errors for the touched notebook and module.

### Open questions
- None.

---

## 2026-04-30 09:29 IST — Add ReLU neuron leaf-responsibility ablation

### Goal
Add notebook tooling to assign hidden ReLU neurons to ODT label leaves by measuring per-leaf log-loss increase after single-neuron ablation.

### User prompts (verbatim)
- "In the cell before the start metrics, I am trying to see if the neurons (all the 120 neurons) can be assigned to one or multiple ODT nodes -- i.e. I want to say \"this neuron is responsible for getting data entering these ODT label leaf nodes classified correctly\".  One way to do this is as follows: have a mask that is all ones, and zero only that neuron and look at all ODT nodes where the log loss is much higher than if we had all neurons active. \n\nCan you write a function which takes in a model and outputs a dictionary with keys being (layer, neuron_id) and values capturing the ODT nodes that neuron is responsible for. Ideally it would also capture the impact of that neuron e.g. how much does the log loss go up by for those ODT nodes etc. \n\nInsert this function and a small script calling it appropriately before the start/end metrics cell."

### Changes
- Inserted a new code cell in `notebooks/scratch.ipynb` before the start/end metrics section.
- Added `neuron_leaf_responsibility_from_ablation`, which compares all-active per-leaf metrics against one-neuron-ablated metrics for all hidden neurons.
- Added an example call that produces `neuron_responsibility`, `neuron_responsibility_df`, `baseline_leaf_metrics`, and a compact `neuron_summary` display.

### Decisions
- Use per-leaf `mean_log_loss` deltas as the primary responsibility signal, with optional thresholds for relative log-loss delta and accuracy drop.
- Keep layer indices zero-based to match `hidden_layers` indexing in `ReLUMLP`.

### Current state / where to pick up
- Run the new notebook cell after the per-leaf masked evaluation helper has been defined and after `seed_result`, `ckpts`, and `EPOCH` exist.
- Tune `min_log_loss_delta` to control how many ODT leaves are attributed to each neuron.

### Open questions
- None.

---

## 2026-04-30 08:52 IST — Add per-leaf masked ReLU log loss

### Goal
Extend the masked ReLU ODT-leaf evaluation helper in the scratch notebook to report average log loss per leaf, not only accuracy.

### User prompts (verbatim)
- "Can you make a change directly in the notebook for also returning the average log loss in each leaf node?"

### Changes
- Updated `notebooks/scratch.ipynb` helper `masked_accuracy_by_odt_leaf` to compute per-example BCE/logistic loss from `{−1,+1}` labels and logits using `np.logaddexp`.
- Added `mean_log_loss` to the returned per-leaf summary table.

### Decisions
- Use the numerically stable `{−1,+1}` logistic-loss form `logaddexp(0, -y * logit)` to match the BCE-with-logits objective used during training.

### Current state / where to pick up
- Re-run the helper-definition cell and then the leaf-accuracy display cell to see the new `mean_log_loss` column.

### Open questions
- None.

---

## 2026-04-25 08:43 IST — Simplify scratch notebook to single-seed focus

### Goal
Remove multi-seed distractions from `scratch.ipynb` and focus all
learning-dynamics analysis on one selected seed.

### User prompts (verbatim)
- "I want to analyse the learning dynamics for a single seed now. Having multiple seeds in all results in other places distracts me. Can you make this modification to the scratch.ipynb notebook?"

### Changes
- Updated `notebooks/scratch.ipynb` to single-seed workflow:
  - Replaced `MASTER_SEEDS` + `all_results` with:
    - `SEED = 5178`
    - `seed_result = None`
  - Replaced training loop over seeds with one run:
    - `seed_result = run_single_seed(SEED)`.
  - Updated scatter cell to read from `seed_result` and use `SEED` in
    title; added guard if training hasn’t run yet.
  - Replaced aggregate multi-seed table cell with single-seed start/end
    table (`metrics_table`).
  - Updated final visualization cell to plot from `metrics_table`
    instead of `agg_table`; added guard if training hasn’t run yet.
  - Updated markdown headings/descriptions to reflect single-seed focus.

### Decisions
- Kept helper function `collect_start_end_metrics(...)` unchanged,
  because it already supports a single seed result input cleanly.

### Current state / where to pick up
- Notebook is now organized around one seed and no longer presents
  multi-seed aggregation outputs in the main flow.
- You can switch seed by changing `SEED` in one place.

### Open questions
- None.

---

## 2026-04-24 19:22 IST — Switch to dense-early quadratic snapshots

### Goal
Capture early-training transitions better by replacing uniform
`SNAPSHOT_STRIDE=200` with a dense-near-init snapshot schedule.

### User prompts (verbatim)
- "Most interesting transitions in the neural network happen early in training. Having snapshots only every 200 epochs does not capture it. Can we capture snapshots more frequently close to init? e.g. quadratically spaced snapshots? 0,1,2,4,7,11,16,22,29, etc?"

### Changes
- Updated `notebooks/scratch.ipynb` config cell:
  - Removed fixed `SNAPSHOT_STRIDE`-based snapshot generation.
  - Added `quadratic_snapshot_epochs(total_epochs, n_points=80)`.
  - Set `SNAPSHOT_EPOCHS` from quadratic spacing over `[0, TOTAL_EPOCHS]`.
  - Added printouts for count + early/late snapshot values for quick
    verification.

### Decisions
- Used quadratic spacing (`frac**2`) to make snapshots much denser near
  initialization and gradually sparser later in training.

### Current state / where to pick up
- Training now stores more early checkpoints while limiting total
  snapshot count.
- Scatter/analysis cells continue to work with `SNAPSHOT_EPOCHS`.

### Open questions
- None.

---

## 2026-04-24 19:17 IST — Add first-layer ReLU vs ODT scatter cell

### Goal
Add a scatter-plot analysis cell in `scratch.ipynb` (before aggregate
analysis) showing first-layer ReLU weight norms against max absolute
cosine similarity to ODT vectors, colored by the best-matching ODT node.

### User prompts (verbatim)
- "Create the code for the scatter plot cell in scratch.ipynb where the weight vector norm of the first layer relu net is plotted w.r.t the max abs cosine similarity to ODT vectors. Color the points based on the ODT node maximising this abs cosine similarity. Similar to the scatter plots for the DLGN made earlier.

Make this in the cell before \"aggregate analysis\"."

### Changes
- Updated `notebooks/scratch.ipynb` cell before “Aggregate analysis”:
  - Validates chosen checkpoint epoch exists.
  - Loads first-layer ReLU weights from checkpoint
    (`hidden_layers.0.weight`).
  - Computes per-neuron weight norm.
  - Computes cosine similarities with ODT normals (`tree.w_list`),
    then max absolute cosine and argmax ODT node index.
  - Renders scatter:
    - x: first-layer weight norm
    - y: max absolute cosine similarity to ODT vectors
    - color: ODT node index maximizing abs cosine
  - Adds colorbar, labels, title, and light grid.

### Decisions
- Implemented with torch ops throughout for consistency with checkpoint
  tensors and numerical stability (`clamp_min(1e-12)` in denominator).

### Current state / where to pick up
- Notebook now has DLGN-style alignment scatter for the ReLU first layer
  in the requested position.

### Open questions
- None.

---

## 2026-04-24 15:34 IST — Add progress option to ReLU training

### Goal
Add an option in ReLU training to show epoch-wise progress.

### User prompts (verbatim)
- "Can you add an option in the relu training for showing progress?"

### Changes
- Updated `src/first_experiment/relu_training.py`:
  - Added `show_progress: bool = False` to `ReLUTrainConfig`.
  - Added tqdm support in `train_relu_mlp(...)`:
    - wraps epoch iteration with a progress bar when enabled
    - shows current epoch loss via `set_postfix`.

### Decisions
- Mirrored DLGN training progress behavior for consistency across
  training modules.

### Current state / where to pick up
- ReLU training now supports progress display with
  `ReLUTrainConfig(show_progress=True)`.
- Validation passed:
  - `./.venv/bin/python -m pytest` -> `24 passed`.

### Open questions
- None.

---

## 2026-04-24 15:30 IST — Move ReLU training routine into Python module

### Goal
Move ReLU training logic out of `scratch.ipynb` into a `.py` training
module while keeping experiment orchestration/data generation in the
notebook.

### User prompts (verbatim)
- "It is better to have the training routine in a .py file . But there is already a training.py file that is for the DLGN, maybe you can reuse the same file or create another file and rename accordingly."

### Changes
- Added `src/first_experiment/relu_training.py`:
  - `ReLUTrainConfig`
  - `train_relu_mlp(...)`
  - `evaluate_relu_mlp(...)`
  - internal PM1-to-binary label conversion helper
- Updated `notebooks/scratch.ipynb`:
  - Replaced notebook-local ReLU train/eval implementations with imports
    from `first_experiment.relu_training`.
  - Kept data generation and seed orchestration in notebook code.
  - `run_single_seed(...)` now calls `train_relu_mlp(...)` with
    `ReLUTrainConfig`.
- Added `tests/test_relu_training.py` for new module coverage.

### Decisions
- Chose a separate `relu_training.py` instead of merging into DLGN
  `training.py` to keep model-specific training paths clean and avoid
  coupling.

### Current state / where to pick up
- ReLU training routine is now reusable and importable from Python code.
- Notebook remains experiment-driver for data generation and analysis.
- Validation passed:
  - `./.venv/bin/python -m pytest` -> `24 passed`.

### Open questions
- None.

---

## 2026-04-24 15:26 IST — Simplify to single-phase ReLU workflow in notebook

### Goal
Reorganize ReLU experiment code to remove multi-phase analysis utilities,
keep only `relu_mlp.py`, and move experiment/training logic into
`scratch.ipynb`.

### User prompts (verbatim)
- "Reorganise the code. There is no need yet for multi-phase analysis. Only a single phase of training is required for now. Keep only the relu_mlp file, and move all experimental details in relu_experiments.py to scratch.ipynb. Have the data generation and training happen in scratch itself using calls to odt.py etc. Remove the relu_experiments.py file."

### Changes
- Deleted `src/first_experiment/relu_experiments.py`.
- Deleted `tests/test_relu_experiments.py` (depended on removed module).
- Updated `notebooks/scratch.ipynb` to a single-phase-only ReLU pipeline:
  - Replaced multi-phase content with direct notebook-local helpers.
  - Imports now use `odt.py`, `training.set_seed`, and `relu_mlp.py`.
  - Added notebook-local training loop (`train_relu_single_phase`) and
    evaluation helper (`evaluate_relu_mlp`).
  - Added notebook-local `run_single_seed` that generates data and trains
    one-phase model per seed.
  - Updated aggregate analysis and plots to compare start vs end
    metrics only (no run-mode axis).

### Decisions
- Kept only `relu_mlp.py` as reusable architecture code, per request.
- Moved all experiment orchestration details (configs, training loop,
  evaluation flow) into `scratch.ipynb`.

### Current state / where to pick up
- ReLU workflow is now single-phase and notebook-driven.
- Test suite passes after cleanup:
  - `./.venv/bin/python -m pytest` -> `22 passed`.

### Open questions
- None.

---

## 2026-04-24 15:18 IST — Add standard ReLU MLP experiment pipeline

### Goal
Create a standard fully connected ReLU architecture path (no dropout,
no normalization), add required experiment files, and repopulate
`scratch.ipynb` for analysis on this architecture.

### User prompts (verbatim)
- "Thanks. Now I need to perform these experiments with a different architecture. Simply the standard fully connected ReLU is sufficient for now. Can you create the required files, and populate the (now empty) scratch.ipynb for performing analysis with it? Choose appropriate hyperparameters as configs (number of hidden layers, number of neurons per layer etc). No need for extra layer magics like dropout or normalization for now."

### Changes
- Added `src/first_experiment/relu_mlp.py`:
  - Introduced `ReLUMLP`, a plain fully connected ReLU classifier with
    scalar-logit output.
- Added `src/first_experiment/relu_experiments.py`:
  - Added `ReLUTrainConfig` and `ReLUExperimentConfig`.
  - Added `evaluate_binary_classifier(...)`.
  - Added `train_relu_mlp(...)` with Adam + cosine schedule + optional
    row-freeze for pruned hidden neurons.
  - Added `run_relu_single_seed(...)` implementing:
    - `two_phase_phase1`
    - `two_phase_phase2_no_prune`
    - `two_phase_phase2_pruned`
    - `one_phase`
- Repopulated `notebooks/scratch.ipynb`:
  - Added config/imports cells for ReLU experiments.
  - Added helper cell to collect start/end checkpoint metrics.
  - Added full multi-seed run cell.
  - Added aggregate per-seed and mean/std analysis tables.
  - Added quick end-metric visualization plots.
- Added `tests/test_relu_experiments.py`:
  - Forward-shape sanity for `ReLUMLP`.
  - Training/evaluation smoke coverage.
  - Single-seed run-mode output coverage.

### Decisions
- Used `hidden_dims=(256, 256, 256)` as a strong but still standard
  baseline for this first ReLU pass.
- Kept training loop simple (no dropout/norm/other layer magic) per
  request.
- Preserved previous experiment framing (one-phase vs two-phase with and
  without pruning) while swapping the architecture.

### Current state / where to pick up
- ReLU experiment path is now available in package code and notebook.
- `scratch.ipynb` is analysis-ready for multi-seed ReLU runs.
- Validation passed:
  - `./.venv/bin/python -m pytest` -> `25 passed`.

### Open questions
- None.

---

## 2026-04-24 10:53 IST — Stop storing split arrays in `all_results`

### Goal
Reduce serialized notebook cache size by removing `x_train`, `x_eval`,
`y_train`, `y_eval` from `all_results` and regenerating splits from
`data_seed` when needed for analysis.

### User prompts (verbatim)
- "Can we avoid storing the x_train, x_eval, y_train, y_eval as part of all_results? We could just generate those from the seeds, right?"
- "Yes, please do so."

### Changes
- Updated `notebooks/multiseed_multipath_runner.ipynb`:
  - Added helper function:
    `regenerate_split_from_seed(data_seed)` to deterministically
    reconstruct train/eval splits from generation settings.
  - Updated `run_single_seed_with_checkpoints(...)` return payload to
    remove the `"data"` block (`x_train`, `y_train`, `x_eval`, `y_eval`)
    from `all_results`.
  - Updated start/end aggregate metrics cell to regenerate split arrays
    from `seed_result["seeds"]["data_seed"]` instead of reading stored
    arrays.

### Decisions
- Kept deterministic regeneration path in-notebook because dataset
  generation is seed-driven and split slicing is fixed.
- Chose to reduce cache payload size rather than storing duplicated
  arrays per seed.

### Current state / where to pick up
- `all_results` is now lighter and still supports scatter and aggregate
  analysis.
- Saved pickle caches will no longer carry train/eval arrays if produced
  after re-running the updated notebook cells.

### Open questions
- None.

---

## 2026-04-23 22:26 IST — Remove redundant re-training API block from notebook

### Goal
Confirm whether the notebook `run_multiseed(...)` section retrains models
and remove it if redundant relative to already-trained checkpoint-based
analysis flow.

### User prompts (verbatim)
- "Does the cell using the run_multiseed API (## Aggregate stats path using the new API (`run_multiseed`)) in the multiseed_multipath_runner.ipynb file train the model again?  Training run over all the run modes takes about 40 minutes per seed.

Does this cell (## Aggregate stats path using the new API (`run_multiseed`)) give me any more functionality than the cells before it and after it does? If not and it actually trains again please remove it."

### Changes
- Updated `notebooks/multiseed_multipath_runner.ipynb`:
  - Removed imports used only by the `run_multiseed(...)` section:
    `ExperimentConfig`, `DataConfig`, `ModelConfig`, `run_multiseed`.
  - Replaced the markdown section text to explicitly mark the
    `run_multiseed` path as removed.
  - Replaced the `run_multiseed(...)` code cell with a placeholder
    comment to prevent accidental full retraining.

### Decisions
- Confirmed `run_multiseed(...)` does retrain all modes for all seeds.
- Since equivalent start/end aggregates are already computed from
  `all_results` (existing checkpoint-rich run), keeping the API block in
  this notebook was redundant and expensive.

### Current state / where to pick up
- Notebook now performs only one training pass (`all_results`) and uses
  those outputs for scatter plots and aggregate start/end tables.
- Accidental duplicate retraining from the removed section is prevented.

### Open questions
- None.

---

## 2026-04-22 (IST) — Add init-time freezing for zeroed gating rows

### Goal
Add init-time parameter freezing support in training, then use it in the
scratch notebook's zeroed Phase-2 branch so zeroed gating vectors remain
frozen during Phase 2.

### User prompts (verbatim)
- "Need functionality in the training routine to freeze some parameters at init. This is to be used in the two phase training second phase to freeze the gating vectors set to zero. Make changes to both the training sub-routine and the scratch notebook to reflect this change."

### Changes
- Updated `src/first_experiment/training.py`:
  - Added `TrainConfig.freeze_parameter_names` for exact-name parameter
    freezing at initialization.
  - Added `TrainConfig.freeze_zero_gating_rows` to freeze all-zero
    gating rows detected at init.
  - Added validation for unknown entries in `freeze_parameter_names`
    (raises `ValueError`).
  - Implemented row-level gradient masking hooks for zeroed gating rows.
  - Added post-step projection to keep frozen gating rows exactly zero.
  - Expanded train function docstring to document freezing controls.
- Updated `tests/test_dlgn_training.py`:
  - Added `test_train_dlgn_sf_freeze_zero_gating_rows_keeps_zeroed_rows_fixed`.
  - Added `test_train_dlgn_sf_unknown_frozen_parameter_raises`.
  - Ran tests: `./.venv/bin/python -m pytest tests/test_dlgn_training.py -q`
    passed (`10 passed`).
- Updated `notebooks/scratch.ipynb`:
  - In the zeroed Phase-2 branch, introduced
    `train_cfg_phase2_zeroed` with `freeze_zero_gating_rows=True`.
  - Wired `out_two_phase_phase2_zeroed` to use this config.
  - Updated run selector map so
    `two_phase_phase2_zeroed` uses `train_cfg_phase2_zeroed`.
  - Updated selector title text to indicate rows are both pre-zeroed and
    frozen.

### Decisions
- Kept freezing behavior opt-in and backward compatible (defaults remain
  non-freezing).
- Used row-level freezing for the gating weights to match the notebook
  use case without requiring architecture changes.

### Current state / where to pick up
- Two-phase branch now supports:
  - Phase 2 without zeroing (`out_two_phase_phase2_nozero`)
  - Phase 2 with pre-zeroed + frozen low-norm gating rows
    (`out_two_phase_phase2_zeroed`)
- Training routine supports both generic named-parameter freezing and
  automatic freezing of zero-initialized gating rows.

### Open questions
- None.

---

## 2026-04-22 (IST) — Add alternate zeroed-gating Phase 2 branch

### Goal
Extend the two-phase notebook pipeline so Phase 1 is shared, then run
two different Phase-2 branches: with and without pre-zeroing low-norm
gating vectors, alongside the one-phase baseline.

### User prompts (verbatim)
- "Build an alternate Phase 2 for the two phase training. The main difference is that before the second phase begins, all gating neurons whose weight vector norm is less than a threshold is zeroed. It also uses the same Phase 1.\n\nSo now there are 4 training runs that has to happen:\n\nPhase 1\nPhase 2 after zeroing some gating vectors\nPhase 2 without any zeroing of gate vectors\nSingle phase."

### Changes
- Updated `notebooks/scratch.ipynb` two-phase section:
  - Added `GATING_NORM_ZERO_THRESHOLD = 0.2`.
  - Kept the same Phase-1 output (`out_two_phase_phase1`) as the shared
    starting point.
  - Added Phase-2 branch A:
    `out_two_phase_phase2_nozero` (no pre-zeroing).
  - Added Phase-2 branch B:
    `out_two_phase_phase2_zeroed` after setting to zero all gating-layer
    rows with norm below threshold.
  - Kept compatibility alias:
    `out_two_phase_phase2 = out_two_phase_phase2_nozero`.
- Updated variable-reference markdown cell to document:
  - `out_two_phase_phase2_nozero`
  - `out_two_phase_phase2_zeroed`
  - `GATING_NORM_ZERO_THRESHOLD`
- Updated metrics run selector cell:
  - Added `RUN_MODE` options for both Phase-2 variants:
    `two_phase_phase2_nozero` and `two_phase_phase2_zeroed`.

### Decisions
- Used `copy.deepcopy(...)` to ensure both Phase-2 branches start from
  the same Phase-1 state without mutating each other.
- Performed gating-row zeroing under `torch.no_grad()` so it is a direct
  parameter edit before Phase-2 training begins.

### Current state / where to pick up
- Notebook now supports four runs:
  - `out_two_phase_phase1`
  - `out_two_phase_phase2_nozero`
  - `out_two_phase_phase2_zeroed`
  - `out_one_phase`

### Open questions
- None.

---

## 2026-04-22 (IST) — Correct IST format and chronology (no exact times)

### Goal
Switch session-log headings to IST conventions, correct chronology from
context, and leave historical times blank where exact HH:MM is unknown.

### User prompts (verbatim)
- "Use IST for the time stamps. Some of the time stamps are clearly wrong and end up giving the updates in the wrong order. (e.g. LR scheduling log is the latest entry prior to this meta conversation) If you don't have the exact times the previous log entries were made, please re-order them correctly from context and leave the times for those blank."

### Changes
- Updated heading guidance to IST usage with explicit fallback:
  date + `(IST)` when exact HH:MM is unknown.
- Converted existing same-day headings to date-only `(IST)` style where
  earlier times were uncertain.
- Reordered entries so chronology follows context:
  pre-metrics cleanup → variable reference → metrics selector →
  LR scheduler → append/timestamp meta fix.

### Decisions
- Avoided fabricating exact historical times; used date-only `(IST)` for
  prior entries that could not be timed reliably.

### Current state / where to pick up
- Session log now reads in correct chronological order and uses IST
  convention without guessed timestamps.

### Open questions
- None.

---

## 2026-04-22 (IST) — Scratch notebook pre-metrics cleanup

### Goal
Apply minimal fixes in `notebooks/scratch.ipynb` before the "Metrics at
each snapshot epoch" section: reproducible seeding (including model
init), clearer two-phase/one-phase naming, and consistent checkpoint
creation setup.

### User prompts (verbatim)
- "I have built a pipeline in scratch.ipynb to compare the two-phase training (gating regularisation followed by no-regularisation with an optional not-yet implemented pruning step in-between) to the no-regularisation one phase only training. In scratch.ipynb. Can you go over it and see if it serves the intended purpose? You can suggest minor renaming/consistency edits. \n\nOne other issue I noticed is reproducibility with random seeds. Randomness is used in three places -- data-generation, model initialisation and training mini-batch selection. Only data-gen and SGD randomness are seeded. The init is not. Can you suggest fixes for that too?"
- "Yes, make these minimal edits up to the training/creation of these checkpoint snapshots for both the workflows. I have not yet started to go through the plotting/comparisons yet. So only fix the issues pointed out by you in the cells before the \"Metrics at each snapshot epoch\" markdown cell."

### Changes
- Updated imports/config cell to include `set_seed` and explicit seed
  variables: `DATA_SEED`, `INIT_SEED`, `TRAIN_SEED_PHASE1`,
  `TRAIN_SEED_PHASE2`, `TRAIN_SEED_ONE_PHASE`.
- Switched data generation to use `seed=DATA_SEED`.
- Added explicit initialization seeding before two-phase model creation
  (`set_seed(INIT_SEED)`), and renamed model variable to
  `model_two_phase`.
- Updated the two-phase training cell to:
  - use fair epoch accounting (`TOTAL_EPOCHS`, split into
    `PHASE1_EPOCHS` and `PHASE2_EPOCHS`),
  - use renamed run outputs (`out_two_phase_phase1`,
    `out_two_phase_phase2`),
  - use separate training seeds for phase 1 and phase 2 configs.
- Renamed one markdown header to "one-phase no-regularization run".
- Updated the one-phase training cell to:
  - reseed initialization before creating `model_one_phase`,
  - use `train_cfg_one_phase` and `out_one_phase`,
  - keep temporary aliases (`out`, `train_cfg`, `out_phase2`) for
    downstream compatibility with existing metric/plot cells.

### Decisions
- Kept edits strictly before the metrics markdown section as requested.
- Preserved backward-compatible aliases to avoid breaking later cells
  before a dedicated plotting/comparison cleanup pass.

### Current state / where to pick up
- Both workflows now have explicit checkpoint-producing training paths
  with controlled data/init/train randomness.
- The metrics and plotting sections remain intentionally untouched and
  still reference compatibility aliases.

### Open questions
- None.

---

## 2026-04-22 (IST) — Add scratch variable reference cell

### Goal
Add a short markdown reference cell before the metrics section in
`notebooks/scratch.ipynb` listing analysis-relevant variable names and
their meanings.

### User prompts (verbatim)
- "Can you create a markdown cell before the metric plotting cell to give the variable names (out_one_phase, train_cfg_one_phase etc) that are relevant for analysis later with a brief description?"

### Changes
- Inserted a new markdown cell immediately before the metrics markdown
  section in `notebooks/scratch.ipynb`.
- The cell documents:
  - `out_one_phase`, `train_cfg_one_phase`
  - `out_two_phase_phase1`, `train_cfg_phase1`
  - `out_two_phase_phase2`, `train_cfg_phase2`
  - compatibility aliases (`out`, `train_cfg`, `out_phase2`)

### Decisions
- Kept the text concise and analysis-focused, without modifying any
  metric/plot code.

### Current state / where to pick up
- Notebook now includes an explicit pre-metrics variable reference for
  downstream analysis.

### Open questions
- None.

---

## 2026-04-22 (IST) — Metrics run selector in scratch notebook

### Goal
Replace manual metrics-cell assignment with a small selectable block for
choosing one-phase or either two-phase stage, including automatic
`fig.suptitle`.

### User prompts (verbatim)
- "Can you make a small piece of code that assigns out, train_cfg, and sets fig.suptitle based on one of three options: one phase, two phase-phase1 or two-phase-phase2 ? Something that can be pasted at the top of the metrics cell to replace the current manual assignment?"
- "Please do."

### Changes
- Updated the metrics plotting code cell in `notebooks/scratch.ipynb`
  to add:
  - `RUN_MODE` selector with options:
    `\"one_phase\"`, `\"two_phase_phase1\"`, `\"two_phase_phase2\"`
  - `run_map` dictionary mapping each mode to:
    `out`, `train_cfg`, and title text
  - mode validation (`ValueError` for invalid mode)
  - dynamic assignment:
    `out = run_map[RUN_MODE][\"out\"]`,
    `train_cfg = run_map[RUN_MODE][\"train_cfg\"]`,
    `FIG_SUPTITLE = run_map[RUN_MODE][\"title\"]`
  - dynamic suptitle:
    `fig.suptitle(FIG_SUPTITLE)`

### Decisions
- Kept all downstream metric/evaluation logic untouched; only the manual
  top-of-cell selection block was replaced.

### Current state / where to pick up
- Metrics cell now supports one-line workflow switching via `RUN_MODE`
  without editing variable names manually.

### Open questions
- None.

---

## 2026-04-22 (IST) — Add LR scheduler to training routine

### Goal
Add a learning-rate scheduler to the core training routine with a
reasonable decay behavior and test coverage.

### User prompts (verbatim)
- "In the training routine, can you add a learning rate scheduler that decays appropriately?"

### Changes
- Updated `src/first_experiment/training.py`:
  - Added `TrainConfig.lr_scheduler` with options `\"none\"` (default)
    and `\"cosine\"`.
  - Added `TrainConfig.lr_scheduler_eta_min_ratio` to control cosine
    minimum LR (`eta_min = lr * ratio`).
  - Added scheduler setup in `train_dlgn_sf` and per-epoch
    `scheduler.step()` call after each training epoch.
  - Added `lr_history` to the returned training output dictionary.
  - Added validation error for unsupported scheduler names.
- Updated `tests/test_dlgn_training.py`:
  - Added `test_train_dlgn_sf_cosine_scheduler_decays_learning_rate`
    to verify LR decreases across epochs when cosine scheduling is
    enabled.
- Ran tests:
  - `./.venv/bin/python -m pytest tests/test_dlgn_training.py -q`
    passed (`8 passed`).

### Decisions
- Kept behavior backward-compatible by defaulting scheduler to
  `\"none\"`; existing experiments keep fixed LR unless explicitly
  configured.
- Used cosine annealing as the built-in decay strategy because it
  decays smoothly and needs only epoch count + minimum LR ratio.

### Current state / where to pick up
- Training now supports both fixed-LR and cosine-decay modes through
  `TrainConfig`.
- Existing notebook/config code can opt in with
  `lr_scheduler=\"cosine\"`.

### Open questions
- None.

---

## 2026-04-22 (IST) — Fix append order and add timestamps in session log

### Goal
Fix session-log ordering issues (entries being inserted in the middle)
and include a time stamp in heading dates.

### User prompts (verbatim)
- "The session logs are being inserted in the middle and not appended. Can you also add a time stamp to the date there?"

### Changes
- Updated session-log heading format template from
  `YYYY-MM-DD` to `YYYY-MM-DD HH:MM`.
- Added an explicit instruction in this file: always append new entries
  at the end, never insert between existing entries.
- Updated existing 2026-04-22 entry headings to include time stamps.
- Reordered misplaced recent entries so newest entries appear lower in
  the file (append-style progression).

### Decisions
- Used timestamped headings consistently to avoid ambiguity and make
  ordering easier to audit.

### Current state / where to pick up
- `notes/session_log.md` now has timestamped headings and explicit
  append-only guidance.

### Open questions
- None.

---

## 2026-04-23 21:53 IST — Refactor multipath workflow into scriptable APIs

### Goal
Implement the approved refactor plan to move multipath DLGN experiment
logic from notebook orchestration into stable Python modules, including
multi-seed execution, descriptive statistics, artifact IO, and tests.

### User prompts (verbatim)
- "Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos."
- "log the changes in session_log."

### Changes
- Added `src/first_experiment/experiment_config.py`:
  - Introduced experiment-level contracts:
    `SeedOffsets`, `SeedBundle`, `DataConfig`, `ModelConfig`,
    `ExperimentConfig`, `EvalMetrics`, `PruningSummary`,
    `RegimeResult`, `SeedRunResult`.
  - Encoded branch-specific `TrainConfig` builders for phase-1,
    phase-2 (no-zero), phase-2 (zeroed/frozen), and one-phase runs.
- Added `src/first_experiment/experiment_runner.py`:
  - Implemented `run_one_phase`, `run_two_phase_nozero`,
    `run_two_phase_zeroed`, and `run_all_regimes_for_seed`.
  - Ported notebook branch logic (shared phase-1 base model, optional
    pre-phase-2 zeroing by norm threshold, freeze-compatible training).
  - Added pruning invariants (`rows_zeroed_per_layer`,
    `rows_still_zero_after_training`).
- Added `src/first_experiment/analysis.py`:
  - Implemented descriptive aggregation across seeds:
    `mean`, `std`, `n`, `sem`, `ci95_low`, `ci95_high`.
  - Added baseline-delta summary helper
    `compute_delta_vs_baseline(...)`.
- Added `src/first_experiment/io.py`:
  - Added run-root creation, manifest writing, per-seed JSON output,
    and summary CSV writing under `results/`.
- Added `src/first_experiment/multiseed.py`:
  - Implemented API-first multi-seed orchestration (`run_multiseed`)
    with optional artifact persistence.
- Added tests:
  - `tests/test_experiment_runner.py`
  - `tests/test_multiseed_analysis.py`
  - Coverage includes seed-fairness invariants, zeroed-branch freeze
    invariants, descriptive schema checks, and artifact writes.
- Validation run:
  - `./.venv/bin/python -m pytest tests/test_experiment_runner.py tests/test_multiseed_analysis.py tests/test_dlgn_training.py`
  - Result: `15 passed`.

### Decisions
- Kept implementation API-first (as requested) while preserving a
  deterministic artifact structure for later CLI wiring.
- Scoped statistics to descriptive summaries + 95% CI for this
  milestone; deferred inferential tests/effect sizes.
- Used normal-approximation 95% CI in `analysis.py` for initial
  reviewer-facing summaries; this can be replaced with t-based CI in a
  later refinement pass if required.

### Current state / where to pick up
- Multipath orchestration is now available from importable Python APIs
  in `src/first_experiment/`.
- Multi-seed runs and summary table generation are available via
  `run_multiseed(...)`, with optional writing to `results/`.
- New tests for runner/analysis contracts pass alongside existing
  training tests.
- Next practical step: adapt `notebooks/scratch.ipynb` to consume these
  APIs and read saved outputs instead of owning execution logic.

### Open questions
- Whether to switch CI computation from normal-approximation to t-based
  intervals for small sample counts.
- Whether to add hypothesis testing/effect sizes in the next milestone
  for reviewer rebuttal material.

---

## 2026-04-23 22:05 IST — Create fresh multi-seed notebook template

### Goal
Provide a copy-paste-ready fresh notebook for running 5-seed multipath
experiments with lower checkpoint frequency, scatter support per
seed/mode/checkpoint, and aggregate statistics.

### User prompts (verbatim)
- "Let's say my goal is to run the experiments in scratch.ipynb for 5 different seeds. Can you give me code that can be copy pasted into a fresh notebook? Intersperse python cells with markdown cells describing the code cells below.

Potential issues include running out of memory becuase we store checkpoints every 20 epochs in all training runs. Maybe reduce that to every 200 epochs for convenience. I need to have the capability to get the scatter plots of the checkpoint epochs of any of the 5 seeds, and all the run modes using that seed config. I also need aggregate statistics such as training loss, test error at the beginning and end of all the run modes for all the chosen seeds.

Other settings such as epoch numbers, depth of tree, number of data points can be taken from the curent scratch.ipynb."
- "Please do so. While you are at it, just create the notebook also and put these cells in it."

### Changes
- Added new notebook:
  `notebooks/multiseed_multipath_runner.ipynb`.
- Populated notebook with interleaved markdown/python cells covering:
  - 5-seed multipath training workflow.
  - Scratch-aligned defaults (`DIM`, `DEPTH`, `N_TRAIN`,
    `TOTAL_EPOCHS`, hidden dims, `BETA`).
  - Reduced checkpoint cadence (`SNAPSHOT_STRIDE=200`).
  - Per-seed checkpoint-rich execution for scatter plots.
  - Scatter helper for any seed/mode/checkpoint:
    `plot_scatter_for_checkpoint(...)`.
  - Aggregate summaries via new API modules:
    `ExperimentConfig` + `run_multiseed(...)` for descriptive/final
    summaries and baseline deltas.
  - Explicit start/end checkpoint metrics aggregation
    (`train_log_loss`, `test_error`) across all seeds and modes.

### Decisions
- Kept two complementary paths in the notebook:
  - direct checkpoint-heavy run path for scatter flexibility;
  - `run_multiseed(...)` API path for script-aligned aggregate outputs.
- Retained scratch defaults while only lowering snapshot frequency to
  reduce memory pressure.

### Current state / where to pick up
- Notebook is ready to run end-to-end.
- User can change `MASTER_SEEDS` and `SNAPSHOT_STRIDE` in one place.
- Scatter and aggregate tables are both wired.

### Open questions
- None.

---

## 2026-04-25 10:13 IST — ReLU layers-as-lenses extension plan

### Goal
Draft a high-level research plan to extend the "Layers as Lenses"
narrative from DLGN to standard ReLU MLPs on COB-ODT data (gates,
paths, neuron-level dynamics) and persist it in a referenceable note.

### User prompts (verbatim)
- "My goal in this research project is to explain, predict and understand where and how neural network parameters move during training. The layers as lenses paper (one of the two papers in research_spec.md and in the old_notebooks folder) tries to do this for a simple single path DLGN and extends it to multi-path DLGNs. \n\nMy goal now is to do similar analysis for ReLU networks and maybe extend it further. I need to extend the notion of gates in DLGN (which are simply halfspaces in the input space) to ReLU networks. And paths in DLGNs (which are simply a product of gates) also to ReLU networks. These are no longer as simple in ReLU MLPs. But the phenomena they cause are similar. I believe the learning dynamics (even in ReLU MLPs) can be explained using the concept of a positive feedback between the dual roles of a neuron. \n\nTo that end I have taken a single successful ReLU run on a ODT dataset for analysis in scratch.ipynb. I am interested in studyng neuron level evolution of gates, as well as the evolution of paths. \n\nDo read the layers as lenses paper again (available both as pdf and tex file in the project) and the scratch.ipynb notebook and its results and think of a high level plan of action for hypotheses and ways of testing/visualising these. No need for code suggestions right now. Take your time and thnk deeply. We will have more conversations and brainstorming of ideas once you are done."
- "Great, this sounds like a good starting point. Move this to research_spec.md so that I can refer to it easier in future conversations."

### Changes
- Re-read `notebooks/old_notebooks/layers_as_lenses/Layers_as_Lenses.tex`
  and `notebooks/scratch.ipynb` to anchor the plan in current state.
- Appended a new section to `notes/research_spec.md`:
  "Extension Plan: Layers-as-Lenses Narrative for ReLU MLPs". Covers
  recap of paper claims, conceptual translation of gates/paths/lens to
  ReLU, 15 numbered hypotheses (Tiers A–E), 9 visualisation ideas,
  suggested sequencing, pitfalls, and open decisions.

### Decisions
- Plan kept conceptual only (no code changes) per user's explicit
  instruction.
- Hypotheses organised into tiers by ambition so future sessions can
  pick a subset to operationalise without re-deriving the framing.

### Current state / where to pick up
- Research charter for the ReLU analysis phase lives in
  `notes/research_spec.md` under the "Extension Plan" section.
- `notebooks/scratch.ipynb` still holds the single-seed ReLU run that
  the plan refers to (one successful seed, snapshot-rich training,
  level-coloured first-layer scatter already implemented).
- No code/test changes were made this session.

### Open questions
- Which of H1–H6 to prioritise for the first deep dive.
- Whether to introduce a `bias=False` parallel ReLU run for cleaner
  first-layer alignment comparisons.
- Whether to define ReLU "paths" as first-class objects in
  `src/first_experiment/` or keep them as analysis-time constructs
  inside the notebook for now.

---

## 2026-04-25 10:21 IST — Fix Markdown LaTeX rendering in research spec

### Goal
Make the math in `notes/research_spec.md` render correctly in Cursor's
Markdown viewer.

### User prompts (verbatim)
- "Latex is not rendered properly in the markdown viewer on cursor. Can you check why? e.g. in researchspec.md."
- "Please do so."

### Changes
- Updated LaTeX expressions in `notes/research_spec.md` to avoid custom
  paper macros unsupported by Markdown renderers.
- Replaced shorthand macros with standard forms used by KaTeX/MathJax:
  `\w, \x, \bu, \m, \E, \1` -> `\mathbf{w}, \mathbf{x}, \mathbf{u},
  \mathbf{m}, \mathbb{E}, \mathbb{1}`.
- Fixed one intermediate replacement artifact (`\mathbf{m}ath...`) and
  verified no unsupported shorthand macros remain.

### Decisions
- Keep equations renderer-friendly in notes by avoiding dependency on
  `.tex` preamble macros.

### Current state / where to pick up
- The math in `notes/research_spec.md` now uses standard Markdown-safe
  LaTeX and should render in Cursor.

### Open questions
- None.

---

## 2026-04-25 10:24 IST — Diagnose markdown LaTeX rendering

### Goal
Identify why LaTeX was not rendering in Cursor markdown preview and apply a fix.

### User prompts (verbatim)
- "No I still don't see proper rendering. Can you create a small markdown test file with a simple latex command?"
- "No this also does not render. How to fix this?"

### Changes
- Added `notes/latex_render_test.md` with minimal inline and block math checks.
- Updated Cursor user settings at
  `~/Library/Application Support/Cursor/User/settings.json` to include:
  - `"markdown.math.enabled": true`

### Decisions
- Treat issue as renderer/settings-level (not file content syntax), because even minimal test math did not render.

### Current state / where to pick up
- Project markdown files contain valid KaTeX-style math delimiters (`$...$`, `$$...$$`).
- Cursor now has explicit markdown math enabled in user settings.
- If still not visible, next check is whether user is viewing source editor vs markdown preview pane and reload window.

### Open questions
- None.

---

## 2026-04-25 10:39 IST — Add notebook copy for math-rendered research plan

### Goal
Provide a renderable (equation-friendly) version of the ReLU extension plan,
since Cursor markdown preview does not render LaTeX reliably in `.md` files.

### User prompts (verbatim)
- "It still fails. Maybe relevant: Markdown cells in .ipynb files seem to render fine."
- "Yes please do so."

### Changes
- Added `notes/research_spec_math.ipynb` containing the ReLU extension plan
  as markdown content with equations for notebook rendering.
- Updated `notes/research_spec.md` to add a pointer line under the
  "Extension Plan" heading:
  - Math-rendered copy: `notes/research_spec_math.ipynb`.

### Decisions
- Keep `notes/research_spec.md` as the canonical plain-text spec and use
  the notebook as the render-focused companion for math-heavy reading.

### Current state / where to pick up
- Use `notes/research_spec_math.ipynb` for rendered equations in Cursor.
- Use `notes/research_spec.md` for text-first reference and search.

### Open questions
- None.

---

## 2026-05-01 11:16 IST — Fix layer-lenses rename fallout

### Goal
Find and fix errors caused by the project/package rename from
`first-experiment` / `first_experiment` to `layer-lenses` /
`layer_lenses`.

### User prompts (verbatim)
- "I did a project rename from "first-experiment" to "layer-lenses" and that has caused some errors. Can you find what they are and fix them?"

### Changes
- Added `src/layer_lenses/cli.py` so the `layer-lenses` console script
  declared in `pyproject.toml` resolves to an importable entry point.
- Added `tests/test_cli.py` covering CLI help and argument forwarding to
  the existing multi-seed runner without launching a full experiment.
- Updated `README.md` to document `uv run python -m pytest`, which works
  in the current environment where `uv run pytest` does not find a
  console executable.

### Decisions
- Kept the CLI minimal and API-backed: it builds an `ExperimentConfig`
  from explicit command-line overrides and delegates to `run_multiseed`.
- Left historical `first_experiment` references in older session-log
  entries unchanged as narrative history.

### Current state / where to pick up
- `uv run python -m pytest` passes.
- `uv run layer-lenses --help` works.
- A tiny `uv run layer-lenses ...` smoke run works and prints summary
  metrics.

### Open questions
- None.

---

## 2026-05-01 11:18 IST — Fix scratch notebook import path

### Goal
Fix `ModuleNotFoundError: No module named 'layer_lenses'` when running
the first code cell in `notebooks/scratch.ipynb`.

### User prompts (verbatim)
- "I get ModuleNotFoundError: No module named 'layer_lenses' if I run scratch.ipynb first cell."

### Changes
- Updated the first code cell in `notebooks/scratch.ipynb` to locate the
  repository root and prepend its `src/` directory to `sys.path` before
  importing `layer_lenses`.

### Decisions
- Used an in-notebook path setup because notebook kernels are not
  guaranteed to have the local `src/` package installed editably.

### Current state / where to pick up
- Import smoke test from the `notebooks/` working directory passes via
  `uv run python`.

### Open questions
- None.

---

## 2026-05-01 11:22 IST — Use project kernel for scratch notebook

### Goal
Replace the ad hoc notebook `sys.path` workaround with an environment-level
fix for importing `layer_lenses`.

### User prompts (verbatim)
- "Ok, but this seems adhoc. We didn't need this block of code when the project was named "first-experiment". Is there some python environment or path level fix to this instead of a code block in the notebook?"

### Changes
- Registered a user Jupyter kernelspec named `layer-lenses` with display
  name `Python (layer-lenses)` pointing to this repo's `.venv` Python.
- Removed the `sys.path` setup block from the first code cell in
  `notebooks/scratch.ipynb`.
- Updated `notebooks/scratch.ipynb` kernelspec metadata to request the
  `layer-lenses` kernel.

### Decisions
- Treat notebook imports as an environment/kernel selection issue, not
  notebook code responsibility.

### Current state / where to pick up
- Select `Python (layer-lenses)` as the notebook kernel if Cursor does not
  switch automatically.
- The registered kernel points at
  `/Users/harishguruprasad/research-code/projects/layer-lenses/.venv/bin/python3`.

### Open questions
- None.

---

## 2026-05-01 11:36 IST — Repair renamed virtual environment

### Goal
Fix stale virtual-environment paths left behind by renaming the project
directory/package from `first-experiment` to `layer-lenses`.

### User prompts (verbatim)
- "There are still some virtual environment settings linking to "first-experiment". Doing source .venv/bin/activate in anoutside terminal still gets me to an environment called "first-experiment" where "python" command is not found. More worryingly, even in a new terminal at the home directory with no virtual environment activated, "python" command is still not found? Something seems to have broken, it was not this way before the rename."

### Changes
- Updated ignored local `.venv` metadata and activation scripts from the
  old absolute path `.../first-experiment/.venv` to
  `.../layer-lenses/.venv`.
- Updated generated `.venv/bin/*` console-script shebangs that still
  pointed at the old venv path.
- Removed the stale local `.venv/bin/first-experiment` console script.
- Restored the README test command to `uv run pytest` now that the
  console-script shebang is fixed.

### Decisions
- Treated the activated `python` failure as a local venv relocation issue:
  the source package rename was correct, but generated venv files store
  absolute paths and needed repair.
- Did not change global shell startup files; outside any venv this machine
  currently has `python3` on PATH, while `python` is not on PATH.

### Current state / where to pick up
- `source .venv/bin/activate` now sets `VIRTUAL_ENV` to the
  `layer-lenses` project path and exposes `.venv/bin/python`.
- `uv run pytest` passes.
- `uv run jupyter kernelspec list` works and shows the project kernels.

### Open questions
- Whether to add a global `python` alias/symlink or conda PATH setup is a
  separate shell preference decision.

---

## 2026-05-01 11:42 IST — Remove provisional CLI

### Goal
Remove the command-line entry point added during rename cleanup because it
is not needed yet.

### User prompts (verbatim)
- "I don't care about a command line entry point yet. Please remove unnecessary files."

### Changes
- Removed `[project.scripts]` from `pyproject.toml`.
- Deleted `src/layer_lenses/cli.py`.
- Deleted `tests/test_cli.py`.
- Refreshed `uv.lock`.

### Decisions
- Keep the package importable and notebook/test focused for now; add a CLI
  later only when there is a concrete command-line workflow to support.

### Current state / where to pick up
- `uv run pytest` passes with 24 tests.

### Open questions
- None.

---

## 2026-05-01 12:06 IST — Rehome misplaced helpers

### Goal
Audit functions that were sitting in odd modules and move or remove
misplaced/redundant helpers.

### User prompts (verbatim)
- "Some of the functions in the files are in odd locations. e.g. "odt_leaf_ids_for_x" is in ReLU analysis, when it has nothiing to do with ReLU. Are there any other similar functions, or redundant functions? If so put them in their right place or remove them. COrrect notebook and other code that refers to such functions."

### Changes
- Moved `odt_leaf_ids_for_x` from `relu_analysis.py` to `odt.py`.
- Added an ODT regression test checking leaf IDs against
  `samples_reaching_node` leaf masks.
- Moved `checkpoint_model_from_state` from `relu_analysis.py` to
  `relu_training.py`.
- Updated `notebooks/scratch.ipynb` imports and `README_detailed.md`.
- Removed duplicated hidden-activation precompute logic by making
  `_precompute_hidden_activation_masks_and_leaf_ids` compose
  `_precompute_hidden_activation_masks`.

### Decisions
- Kept ReLU/ODT interaction analyses in `relu_analysis.py` when they
  genuinely combine ReLU activation behavior with ODT leaves.

### Current state / where to pick up
- `uv run pytest` passes with 25 tests.

### Open questions
- None.

---

## 2026-05-01 15:13 IST — Speed up neuron-leaf activity counting

### Goal
Reduce redundant computation in the notebook section after
"Neuron activity breakdown" where active points are counted for every
neuron/leaf pair.

### User prompts (verbatim)
- "Can you make the counting active points for every neuron more efficient?  the cell after "Neuron activity breakdown" in @notebooks/scratch.ipynb ? I can see that there is significant redundant computation happening there."

### Changes
- Rewrote `summarize_neuron_leaf_activity` in `src/layer_lenses/relu_analysis.py`
  to use vectorized aggregation (`np.add.at`) per layer instead of nested
  `(neuron, leaf)` calls to `count_active_points_for_neuron_leaf`.
- Updated `notebooks/scratch.ipynb`:
  - the post-heading cell now calls `summarize_neuron_leaf_activity(...)`
    directly;
  - the next cell now uses `neuron_leaf_activity_dict(...)` instead of a
    manual dataframe filtering loop.
- Added `tests/test_relu_analysis.py` to validate the vectorized summary
  against scalar `count_active_points_for_neuron_leaf` spot checks.

### Decisions
- Kept `count_active_points_for_neuron_leaf` as a simple scalar utility for
  targeted inspection, while making bulk counting use the vectorized path.

### Current state / where to pick up
- `uv run pytest` passes with 26 tests.
- Notebook cells now route bulk activity counting through reusable,
  optimized library functions.

### Open questions
- None.

---

## 2026-05-01 18:21 IST — Add activation tensor helper

### Goal
Add a reusable helper that returns hidden-neuron activity as a boolean
array indexed by layer, neuron, and data point.

### User prompts (verbatim)
- "Can you add a function in @src/layer_lenses/relu_analysis.py that takes in a model check point and x_eval and returns a boolean numpy array of shape (layers, neurons, num_data)? The output array corresponds to which data points are active for the corresponding neuron."

### Changes
- Added `activation_tensor_by_layer_neuron_data(...)` to
  `src/layer_lenses/relu_analysis.py`.
- The function returns a boolean tensor with shape
  `(num_hidden_layers, max_hidden_width, num_data)`.
- For layers narrower than `max_hidden_width`, trailing neuron slots are
  padded with `False`.
- Added tests in `tests/test_relu_analysis.py` checking shape, dtype,
  padding semantics, and consistency with
  `active_point_mask_for_neuron(...)`.

### Decisions
- Used max-width padding so the return value is always a dense 3D NumPy
  array even when hidden-layer widths differ.

### Current state / where to pick up
- `uv run pytest` passes with 27 tests.

### Open questions
- None.

---

## 2026-05-01 18:23 IST — Add notebook activation-tensor demo cell

### Goal
Add a notebook cell after "Neuron activity breakdown" that demonstrates the
new activation-tensor helper and prints quick layer-wise summaries.

### User prompts (verbatim)
- "Yes, please do so after the "Neuron activity breakdown" markdown cell."

### Changes
- Inserted a new code cell immediately after the
  "Neuron activity breakdown" markdown cell in `notebooks/scratch.ipynb`.
- The cell computes `activation_tensor` via
  `activation_tensor_by_layer_neuron_data(...)`, prints its shape, and
  reports per-layer active-fraction summary stats.

### Decisions
- Kept output concise and added an optional commented snippet for inspecting
  one neuron's boolean mask over evaluation points.

### Current state / where to pick up
- Notebook section order now is:
  markdown heading -> activation tensor demo cell -> bulk summary cell.

### Open questions
- None.

---

## 2026-05-01 23:06 IST — Add checkpoint inner-product plot cell

### Goal
Add a notebook cell after the first-layer norm vs `|cosine|` scatter plot
to track checkpoint-wise inner products for a selected first-layer neuron
and ODT internal node.

### User prompts (verbatim)
- "Can you add a cell just after the scatter plot of neuron norms vs |cosine| with another plotting cell? It should take in a first layer neuron id, and an internal ODT node and plot the inner product of the the corresponding incoming weight vector with the ODT node vector corresponding to the internal node. x-axis is the checkpoint epochs. Use the previously generated seed_result object."

### Changes
- Inserted a new code cell in `notebooks/scratch.ipynb` immediately after
  the scatter plot section.
- The cell:
  - takes `FIRST_LAYER_NEURON_ID` and `ODT_INTERNAL_NODE_ID`,
  - validates both indices,
  - computes `dot(hidden_layers.0.weight[neuron], tree.w_list[node])` for
    each checkpoint epoch,
  - plots inner product vs checkpoint epoch.

### Decisions
- Kept the cell independent and explicit so users can quickly tune neuron
  and node ids without modifying library code.

### Current state / where to pick up
- Notebook section order is now:
  scatter plot cell -> inner-product-over-epochs cell -> graph visualisation.

### Open questions
- None.

---

## 2026-05-01 23:10 IST — ReLU training: SGD option

### Goal
Allow ReLU training to use plain (no-momentum) SGD instead of AdamW when requested.

### User prompts (verbatim)
- "Can you write an option in the ReLU training setup to just do simple gradient descent (without momentum) instead of Adam?"

### Changes
- Extended `ReLUTrainConfig` with `optimizer: str = "adamw"` (`"adamw"` / `"adam"` or `"sgd"` / `"gd"`).
- `train_relu_mlp` uses `torch.optim.SGD(..., momentum=0.0)` for the SGD path; cosine LR schedule unchanged.
- Threaded `optimizer` through `run_single_relu_seed` in `relu_analysis.py`.
- Added `test_train_relu_mlp_sgd_smoke` in `tests/test_relu_training.py`.
- Noted the option in `README_detailed.md`.

### Decisions
- SGD is implemented as `torch.optim.SGD` with `momentum=0` (vanilla mini-batch updates when `batch_size` < n).

### Current state / where to pick up
- `uv run pytest` passes with 28 tests.

### Open questions
- None.

---

## 2026-05-01 23:15 IST — Log-loss parameter gradients helper

### Goal
Expose a reusable helper that computes gradients of the training log loss with
respect to all model parameters on a given `(x, y)` batch without optimizer updates.

### User prompts (verbatim)
- "Write a function in @src/layer_lenses/relu_analysis.py that takes in a model, and a train/eval set x,y, and computes the gradient for all parameters on this set for the log loss. It makes no updates. It retrurns the gradients as an appropriate dictionary."

### Changes
- Added `log_loss_gradients(...)` to `src/layer_lenses/relu_analysis.py`.
- Loss matches training: mean `binary_cross_entropy_with_logits` with
  `y` in `{-1,+1}` mapped to `{0,1}`.
- Returns `dict[str, torch.Tensor]` aligned with `model.state_dict()` keys;
  values are CPU detached clones; model gradients cleared afterward.
- Added `test_log_loss_gradients_keys_and_shapes` in
  `tests/test_relu_analysis.py`.
- Documented in `README_detailed.md`.

### Decisions
- Use **mean** loss over the provided set so gradient magnitude matches a
  single full-batch training step with default mean reduction.

### Current state / where to pick up
- `uv run pytest` passes with 29 tests.

### Open questions
- None.

---

## 2026-05-03 09:51 IST — Project first-layer ReLU gradients

### Goal
Add an experimental training restriction for ReLU MLPs: project first-layer
incoming-weight gradients to the orthogonal complement of a chosen vector
before optimizer updates.

### User prompts (verbatim)
- "Can you write a modification to the training procedure in the relu_training.py file, where gradients for the first layer neurons are projected to the orthogonal complement of the root node ODT vector u_0 before being used? I want to see if training still succeeds despite this extra restriction."

### Changes
- Added `first_layer_grad_orthogonal_to` to `ReLUTrainConfig`.
- Added gradient projection in `train_relu_mlp` after `loss.backward()` and
  before `optimizer.step()`, applied row-wise to `hidden_layers.0.weight.grad`.
- Added `project_first_layer_grad_orthogonal_to_root` to
  `run_single_relu_seed`, which passes `tree.w_list[0]` as the projection vector.
- Added a regression test verifying the one-step SGD first-layer update has
  zero component along the projection vector.
- Updated `README_detailed.md`.

### Decisions
- Keep the training primitive generic: `relu_training.py` accepts any
  projection vector, while `relu_analysis.py` provides the ODT-root
  convenience flag.
- Only first-layer weights are projected; first-layer biases, if present,
  are unchanged because the restriction is on incoming weight vectors.

### Current state / where to pick up
- `uv run pytest` passes with 30 tests.

### Open questions
- Whether AdamW's coordinate-wise adaptive scaling is desirable for this
  experiment; SGD preserves the projected update direction most directly.

---

## 2026-05-06 07:47 IST — Allow provided ReLU initialization

### Goal
Let `run_single_relu_seed` train from a caller-provided ReLU MLP initialization
instead of always constructing one from `master_seed + 1000`.

### User prompts (verbatim)
- "Can you modify run_single_relu_seed so that it can optionally take a model init as as anrgument instead of always choosing init based on a random seed?"

### Changes
- Added optional `model_init` to `run_single_relu_seed`.
- Validates that provided initial models match `dim`, `hidden_dims`, and `bias`.
- Deep-copies provided initial models before training so the source init remains reusable.
- Added a regression test in `tests/test_relu_analysis.py`.
- Updated `README_detailed.md`.

### Decisions
- Preserve existing seed-based initialization when `model_init` is omitted.
- Keep `init_seed` in the returned seed metadata for backward compatibility, even when
  the provided model path does not use it.

### Current state / where to pick up
- `uv run pytest tests/test_relu_analysis.py tests/test_relu_training.py` passes with 8 tests.

### Open questions
- None.

---

## 2026-05-06 07:51 IST — Coordinate-axis ReLU ODT option

### Goal
Allow `run_single_relu_seed` to generate COB-ODT data from coordinate-axis
hyperplanes instead of a random orthogonal basis.

### User prompts (verbatim)
- "Also, add an option in the ReLU single seed training function so that the generating ODT hyperplaes corespond to the co-ordinate axis instead of a random orthogonal set."

### Changes
- Added `build_axis_aligned_cob_odt_tree` in `src/layer_lenses/odt.py`.
- Added `odt_hyperplanes` to `run_single_relu_seed`, with options
  `"random_orthogonal"` and `"coordinate_axes"`.
- Added a regression test checking that the coordinate-axis path uses the first
  standard basis vectors as ODT normals.
- Updated `README_detailed.md`.

### Decisions
- Kept `"random_orthogonal"` as the default to preserve existing behavior.
- Built the coordinate-axis case as an explicit tree passed through
  `generate_cob_odt_data(tree=...)`, so existing data validation remains active.

### Current state / where to pick up
- `uv run pytest tests/test_relu_analysis.py tests/test_relu_training.py` passes with 9 tests.

### Open questions
- None.

---

## 2026-05-06 08:33 IST — ReLU parameter update masks

### Goal
Let ReLU training freeze individual scalar parameter entries via a 0/1 update mask.

### User prompts (verbatim)
- "Have the relu training take an extra configuration option. It should take a 0/1 parameter mask that tells whether each individual parameter is to be updated or kept frozen during training. ReLU single seed should also take this mask as argument while creating the config."

### Changes
- Added `parameter_update_mask` to `ReLUTrainConfig`.
- Validates mask keys, shapes, and binary 0/1 values against model parameter names.
- Applies masks to gradients and restores frozen entries after each optimizer step,
  including against AdamW weight decay.
- Threaded `parameter_update_mask` through `run_single_relu_seed`.
- Added regression tests for low-level training and single-seed wrapper behavior.
- Updated `README_detailed.md`.

### Decisions
- Use a dictionary keyed by `model.named_parameters()` names, with arrays/tensors of
  the same shape as each parameter.
- Require complete masks when provided so missing parameter-freeze behavior is not
  ambiguous.

### Current state / where to pick up
- `uv run pytest tests/test_relu_analysis.py tests/test_relu_training.py` passes with 11 tests.

### Open questions
- None.

---

## 2026-05-06 08:43 IST — Gaussian ReLU initialization

### Goal
Remove the axis-aligned box artifact from ReLU MLP initialization while keeping
the same per-parameter variance as PyTorch's default linear uniform init.

### User prompts (verbatim)
- "Right now the parameters of the ReLU net are initialised using a uniform distribution, and when the ODT hyperplanes are axis parallel it creates an unusual correlation pattern between ODT hyperplanes and parameters. Can you change that behaviour ? (maybe replace uniform with Gaussian so that they still have same variance?)"

### Changes
- Added explicit Gaussian initialization to `ReLUMLP.reset_parameters`.
- Hidden and output linear weights, and biases when present, now use
  `Normal(0, 1 / (3 * fan_in))`.
- Added `tests/test_relu_mlp.py` to check variance scale and that weights are
  not constrained to the old default uniform interval.
- Updated `README_detailed.md`.

### Decisions
- Match the variance of PyTorch's default `nn.Linear` uniform initialization:
  `Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))` has variance `1 / (3 * fan_in)`.

### Current state / where to pick up
- `uv run pytest tests/test_relu_mlp.py tests/test_relu_analysis.py tests/test_relu_training.py` passes with 12 tests.

### Open questions
- None.

---

## 2026-05-06 23:53 IST — Legacy pickle import compatibility

### Goal
Explain and fix failures when loading pickle files created before the project
rename from `first_experiment` to `layer_lenses`.

### User prompts (verbatim)
- "I had previously generated a pickle file when the project name was "first-experiment", now when I try to load that file I get an error "No module named 'first_experiment'" . The project was thoroughly renamed to "layer-lenses", why is this happening? Can it be fixed?"

### Changes
- Added `src/first_experiment/__init__.py` as a backward-compatibility shim
  that forwards old `first_experiment.*` module paths to `layer_lenses.*`.
- Added `tests/test_rename_compat.py` to verify old pickle global paths resolve.
- Updated `README_detailed.md`.

### Decisions
- Keep source code and new imports under `layer_lenses`; use `first_experiment`
  only as a compatibility alias for legacy serialized artifacts.

### Current state / where to pick up
- `uv run pytest` passes with 36 tests.

### Open questions
- None.

---

## 2026-05-07 07:09 IST — DLGN first ODT alignment hits

### Goal
Add notebook analysis code to find the first checkpoint epoch where DLGN gating
neurons align strongly with each ODT internal node.

### User prompts (verbatim)
- "I am doing some analysis on the DLGN training in @notebooks/DLGN_multiseed_multipath_runner.ipynb . Can you write code which takes the "all_results" object, and for each seed and each ODT inernal node finds the first epoch where some neuron hits a cosine similarity greater than 0.8?"

### Changes
- Added a notebook helper `first_odt_alignment_hit_epochs(...)`.
- The helper scans `all_results` across master seeds, run modes, ODT internal
  nodes, and checkpoint epochs.
- It returns a tidy dataframe with first hit epoch, winning global neuron,
  layer/local neuron id, signed cosine, and absolute cosine.
- Added a notebook execution cell that builds `odt_alignment_hit_epochs`.

### Decisions
- Default thresholding uses `abs(cosine) > 0.8`, matching existing scatter
  plots based on `max |cosine|`; pass `use_abs=False` for signed positive
  cosine only.

### Current state / where to pick up
- Notebook edited only; no tests were run.

### Open questions
- None.

---

## 2026-05-07 08:09 IST — DLGN ODT hit-epoch tree plots

### Goal
Visualize first-hit epochs on the ODT internal-node tree for the
`two_phase_phase1` DLGN run mode.

### User prompts (verbatim)
- "Take the ODT_alignment_hit_epochs object for just the "two_phase_phase1" run mode, and for each seed can you create a tree visualisation that gives the hit epoch for each ODT node?"

### Changes
- Added notebook helper `_complete_binary_internal_node_positions(...)`.
- Added `plot_phase1_odt_hit_epoch_trees(...)` to
  `notebooks/DLGN_multiseed_multipath_runner.ipynb`.
- The helper filters hit-epoch rows by run mode, creates one subplot per seed,
  draws internal ODT tree edges, colors nodes by first hit epoch, and labels
  nodes with `node_id` and hit epoch (`NA` if none).

### Decisions
- Use a shared color scale across all seed subplots for easier comparison.
- Keep the default run mode as `"two_phase_phase1"`, while allowing override.

### Current state / where to pick up
- Notebook edited only; no tests were run.

### Open questions
- None.

---

## 2026-05-07 08:14 IST — DLGN alignment norm threshold

### Goal
Require DLGN ODT alignment hits to pass both a cosine threshold and a gating-vector
norm threshold.

### User prompts (verbatim)
- "Can you modify the alignment hit epoch computation to also take a norm threshold in addition to a abs cosine threshold?"

### Changes
- Updated `first_odt_alignment_hit_epochs(...)` in
  `notebooks/DLGN_multiseed_multipath_runner.ipynb` with a `norm_threshold`
  argument.
- A hit now requires the selected neuron's gating-vector norm to exceed
  `norm_threshold` and the configured cosine score to exceed `threshold`.
- The returned dataframe now records `gating_norm`, `cosine_threshold`, and
  `norm_threshold`.
- Updated the notebook example call to pass `norm_threshold=0.0` explicitly.

### Decisions
- Preserve the existing default behavior by setting `norm_threshold=0.0`.
- Keep the cosine threshold parameter name as `threshold` for compatibility, but
  label it as `cosine_threshold` in output rows.

### Current state / where to pick up
- Notebook edited only; no tests were run.

### Open questions
- None.

---

## 2026-05-07 08:37 IST — DLGN hit-tree epoch labels

### Goal
Make the ODT hit-epoch tree labels cleaner by showing only the epoch inside each
node, with stronger typography.

### User prompts (verbatim)
- "In the tree visualisation can you remove the node id, and make the epoch inside the node boger or bold?"

### Changes
- Updated `plot_phase1_odt_hit_epoch_trees(...)` in
  `notebooks/DLGN_multiseed_multipath_runner.ipynb`.
- Node labels now show only the hit epoch (`NA` if no hit), not the ODT node id.
- Increased the label font size and set `fontweight="bold"`.

### Decisions
- Kept node positions, colors, and tree edges unchanged.

### Current state / where to pick up
- Notebook edited only; no tests were run.

### Open questions
- None.

---

## 2026-05-07 11:11 IST — ReLU gradient projection node selector

### Goal
Allow ReLU single-seed runs to project first-layer gradients orthogonal to any
chosen ODT internal node, instead of only the root node.

### User prompts (verbatim)
- "Change the relu single seed running option where project_first_layer_grad_orthogonal is an integer or None. If it is an integer corresponding to an ODT internal node it projects the gradients orthogonal to that node. Right now it always projects orthogonal to the root."

### Changes
- Replaced `run_single_relu_seed(...)`'s
  `project_first_layer_grad_orthogonal_to_root` boolean with
  `project_first_layer_grad_orthogonal: int | None`.
- Added validation that the projection selector is either `None` or a valid ODT
  internal node id.
- Passed the selected `tree.w_list[node_id]` into the existing
  `ReLUTrainConfig.first_layer_grad_orthogonal_to` projection vector.
- Updated `notebooks/ReLU_runner.ipynb` and `notebooks/scratch.ipynb` calls.
- Updated `README_detailed.md`.
- Added regression tests for selected-node projection and invalid node ids.

### Decisions
- Keep the lower-level training config unchanged because it already supports an
  arbitrary projection vector.
- Treat `None` as no projection and reject booleans explicitly, since bools are
  integers in Python but are ambiguous for this API.

### Current state / where to pick up
- `uv run pytest tests/test_relu_analysis.py` passes with 8 tests.
- Edited notebooks parse as valid JSON.

### Open questions
- None.

---

## 2026-05-07 11:22 IST — ReLU ODT capture epochs

### Goal
Add notebook analysis code to find when first-layer ReLU neurons first capture
ODT internal nodes by absolute cosine alignment.

### User prompts (verbatim)
- "Can you write a piece of code for the ReLU_runner.ipynb notebook that take ckpts and gives the first epoch where any ODT node is "captured" by some first layer neuron? Define captured via an abs cosine threshold."

### Changes
- Added a "First ODT capture epochs" section to `notebooks/ReLU_runner.ipynb`.
- Added `first_odt_capture_epochs(...)`, which scans checkpoint snapshots and
  returns one row per ODT internal node.
- The returned dataframe records first capture epoch, capturing first-layer
  neuron id, signed cosine, absolute cosine, and threshold.
- Added an example call producing `odt_capture_epochs` and displaying it with
  `itables.show`.

### Decisions
- Define capture as `abs(cosine) > threshold`, with default `threshold=0.8`.
- Keep this as notebook-local analysis code for now.

### Current state / where to pick up
- Notebook edited only; JSON parsing was validated.

### Open questions
- None.

---

## 2026-05-07 11:32 IST — ReLU ODT capture tree

### Goal
Visualize the hierarchical order in which first-layer ReLU neurons capture ODT
internal nodes.

### User prompts (verbatim)
- "Can you create similar tree like figures as was done for DLGN as well? To illustrate the hierarchical capturing order."

### Changes
- Added an "ODT capture-epoch tree" section to `notebooks/ReLU_runner.ipynb`.
- Added `_complete_binary_internal_node_positions(...)`.
- Added `plot_odt_capture_epoch_tree(...)`, which colors ODT internal nodes by
  first capture epoch and labels each node with the epoch (`NA` if uncaptured).
- Added an example call plotting `odt_capture_epochs`.

### Decisions
- Reused the same complete-binary-tree layout and bold epoch-only labels used
  in the DLGN visualization.
- Kept the ReLU plot single-panel because `ReLU_runner.ipynb` is a single-seed
  runner.

### Current state / where to pick up
- Notebook edited only; JSON parsing was validated.

### Open questions
- None.

---

## 2026-05-07 (IST) — DLGN hit-epoch tree label contrast

### Goal
Make epoch numbers easier to read on dark viridis-colored ODT hit-epoch tree nodes.

### User prompts (verbatim)
- "IN the @notebooks/DLGN_multiseed_multipath_runner.ipynb notebook, can you make some changes to the ODT-node alignment hit-epoch figures? The dark colors make it hard to read the epoch number inside it."

### Changes
- `notebooks/DLGN_multiseed_multipath_runner.ipynb`: in `plot_phase1_odt_hit_epoch_trees`, added `_epoch_tree_label_style` using sRGB luminance to pick white vs black label fill and `matplotlib.patheffects.withStroke` for a contrasting outline on each background.

### Decisions
- Use luminance threshold 0.5; dark viridis gets white text with dark stroke, light/NA gets black text with white stroke.

### Current state / where to pick up
- Notebook-only change; re-run the cell to refresh figures.

### Open questions
- None.

---

## 2026-05-07 (IST) — ReLU ODT capture tree matches DLGN style

### Goal
Align the ReLU ODT capture-epoch tree figure with the DLGN hit-epoch tree styling.

### User prompts (verbatim)
- "In the @notebooks/ReLU_runner.ipynb notebook there is a similar figure for ODT capture epochs. Can you make it also in the same style as the DLGN figures?"

### Changes
- `notebooks/ReLU_runner.ipynb`: added `_epoch_tree_label_style` (same as DLGN), marker size `s=780`, epoch labels `fontsize=9` with luminance-based fill and stroke; `fig.suptitle` + `ax.set_title(f\"Seed {SEED}\")` at `fontsize=14`; removed standalone `fig.colorbar` to match DLGN trees.

### Decisions
- No colorbar on the tree (same as `plot_phase1_odt_hit_epoch_trees`); epoch numbers remain the numeric legend.

### Current state / where to pick up
- Notebook-only; re-run the plotting cell.

### Open questions
- None.

---

## 2026-05-07 (IST) — ReLU ODT capture tree geometry matches DLGN

### Goal
Match edge length/angle perception in the ReLU capture-epoch tree to DLGN hit-epoch trees for side-by-side figures.

### User prompts (verbatim)
- "The edges between the nodes seem longer and at different angle than the DLGN figures. Can you make it match exactly so that they can be not jarring when put side-to-side."

### Changes
- `notebooks/ReLU_runner.ipynb`: `plot_odt_capture_epoch_tree` now defaults to `figsize_per_panel=(6.0, 3.8)` (same as DLGN) and, when `use_dlgn_panel_geometry` is True, builds a `(2 * w) × h` figure with `Figure.add_gridspec(1, 2)` and only `add_subplot(gs[0, 0])`, so the axes box matches one column of DLGN’s two-column layout (not a full-width 6×3.8 panel). Added `use_dlgn_panel_geometry=False` to recover a single full-panel figure if desired.

### Decisions
- Mismatch was from axes aspect: one full-width subplot vs half-width in a 2-col grid, plus the old 8×4.8 default size.

### Current state / where to pick up
- Notebook-only; re-run the cell. Exported PDFs are now 12 in wide unless `use_dlgn_panel_geometry=False` (crop or use the left half for layouts).

### Open questions
- None.

---

## 2026-05-07 (IST) — Generic complete-binary tree plot utility

### Goal
Provide importable functions to lay out and plot numeric labels on the COB-ODT-style complete binary tree (e.g. 31 internal nodes), matching notebook DLGN tree styling.

### User prompts (verbatim)
- "Can you create a generic plot function which takes 31 numbers and puts those numbers in a tree similar to the plot_phase1_odt_hit_epoch_trees function in@notebooks/DLGN_multiseed_multipath_runner.ipynb  and returns the figure and axis? Put such a function in a separate utilitles.py file."

### Changes
- Added `src/layer_lenses/utilities.py` with `complete_binary_internal_node_positions`, `epoch_tree_label_style`, and `plot_complete_binary_tree_values` (viridis normalization, gray edges, markers/labels as in notebooks; optional `use_dlgn_panel_geometry`).
- Added `tests/test_utilities.py`.

### Decisions
- Support any valid complete-binary internal count (`2**d-1`), not only 31; use `float('nan')` for missing.
- Figure layout: `layout="constrained"` instead of `tight_layout` to avoid GridSpec warnings.

### Current state / where to pick up
- Notebooks still duplicate helpers; optional follow-up is to import from `layer_lenses.utilities`.

### Open questions
- None.

---

## 2026-05-07 (IST) — DLGN notebook cell for utility tree plot

### Goal
Demonstrate `plot_complete_binary_tree_values` using `odt_alignment_hit_epochs` in the DLGN notebook.

### User prompts (verbatim)
- "Can you give a small cell that uses the odt_alignment_hit_epochs object from the DLGN notebook and uses the newly created utility plotting function to make the tree figure?"

### Changes
- `notebooks/DLGN_multiseed_multipath_runner.ipynb`: new code cell after `odt_alignment_hit_epochs` is built; filters by run mode + first sorted seed, `reindex` of `first_epoch`, calls `plot_complete_binary_tree_values`.

### Decisions
- Example seed is `sorted(unique(master_seed))[0]`; override `_seed_utility` / `_run_mode_utility` in the cell as needed.

### Current state / where to pick up
- Re-run notebook through the new cell.

### Open questions
- None.

---

## 2026-05-07 (IST) — ReLU notebook: utility capture-epoch tree cell

### Goal
Add a cell using `plot_complete_binary_tree_values` with `odt_capture_epochs` after the ODT capture-epoch tree section.

### User prompts (verbatim)
- "Can you use the same function and the odt_capture_epochs object in the @notebooks/ReLU_runner.ipynb notebook to create a similar tree plot? Insert a cell aftre the \"ODT capture-epoch tree\" cell."

### Changes
- `notebooks/ReLU_runner.ipynb`: new code cell (index 10) after the capture-epoch tree implementation; builds a reindexed `first_epoch` vector and calls `plot_complete_binary_tree_values` with `use_dlgn_panel_geometry=True`.

### Current state / where to pick up
- Re-run from training / capture epochs through the new cell.

### Open questions
- None.

---

## 2026-07-21 00:25 IST — Per-layer masks for ReLU graph

### Goal
Allow the layered ReLU graph to mask hidden layers with different widths.

### User prompts (verbatim)
- "Edit plot_layered_relu_graph function in relu_analysis.py to take in node masks as a list of numpy arrays also. That way hidden layers with different sizes can also be handled."

### Changes
- `src/layer_lenses/relu_analysis.py`: accepted and validated one 1D NumPy mask per hidden layer while retaining the existing rectangular-array interface.
- `tests/test_relu_analysis.py`: added coverage for a `[4, 3, 2]` hidden-width graph with per-layer masks.

### Decisions
- Per-layer masks must exactly match their corresponding hidden-layer widths; rectangular masks remain supported with padding to the widest layer.

### Current state / where to pick up
- Focused ReLU analysis tests pass (`9 passed`).

### Open questions
- None.

---

## 2026-07-26 21:16 IST — Variable-depth ReLU graph

### Goal
Plot ReLU MLP connectivity for models with one, two, or three hidden layers.

### User prompts (verbatim)
- "Change the plot_layered_relu_graph function so that it can also plot neural nets with one or two or three hidden layers. Not just three."

### Changes
- `src/layer_lenses/relu_analysis.py`: dynamically discovers hidden-layer weights and builds layer positions, edges, labels, masks, and plot bounds for one to three hidden layers.
- `tests/test_relu_analysis.py`: added graph and edge-count coverage for one-, two-, and three-hidden-layer models.

### Decisions
- The plotting API explicitly supports one through three contiguous hidden layers and raises a clear error outside that range.

### Current state / where to pick up
- Focused ReLU analysis tests pass (`10 passed`).

### Open questions
- None.

---

## 2026-07-29 09:26 IST — Variable-depth scratch notebook analysis

### Goal
Remove fixed three-layer/equal-width assumptions from the latter part of the ReLU scratch notebook and its reusable analysis helpers.

### User prompts (verbatim)
- "Go through the scratch.ipynb notebook. You may edit cells starting from the markdown cell \"## Load neural net parameters from given epoch into W0, W1, W2 and Wout\".

Several places require that the ReLU network have 3 hidden layers with same neurons. Find and edit those to handle 1 or 2 or 3 hidden layers with different number of neurons in each layer. Some functions in relu_analysis.py would also need changes for this. Go ahead and do this.

Use the HIDDEN_DIMS object defined early on in the notebook. You may assume that the seed_result contains neural networks compatible with HIDDEN_DIMS."
- "Perform any necessary follow-up actions in response to the subagent completion above. If no follow-up work is needed, no further action is required. If you mention an agent or subagent in your response, link it with the `[Name](id)` Don't use generic label such as `[agent]`, `[worker]`, or `[subagent]`. For cloud subagents, when the agent has edited code, link to `[Review](bc-id#changes)`, or, if you know the exact added and deleted line counts, `[Review +A −D](bc-id#changes)`, replacing A and D with those counts. Never write A or D literally. Use `[Try Live](bc-id#desktop)` only when the agent used computer use. Don't repeat the same confirmation every time."

### Changes
- `notebooks/scratch.ipynb`: generalized weight loading, graph masks, ablation masks, masked evaluation, activity queries, weighted-lens gradient analysis, and all dependent projection/plot/example cells from the authorized markdown cell onward using `HIDDEN_DIMS` and dynamic ODT dimensions.
- `src/layer_lenses/relu_analysis.py`: centralized hidden-layer/mask validation, added list-mask support to masked inference and responsibility analysis, and added `first_layer_neuron_weighted_lens` for architecture-independent active-path sums.
- `tests/test_relu_analysis.py`: covered heterogeneous list masks, responsibility analysis, and weighted-lens agreement with autograd for one to three hidden layers.

### Decisions
- Per-layer mask lists are the notebook default because they represent heterogeneous widths without padding.
- Weighted-lens path sums use backward dynamic programming rather than architecture-specific nested path loops.

### Current state / where to pick up
- All notebook code cells from the authorized section onward compile and contain no remaining searched fixed-depth/equal-width patterns.
- Full test suite passes (`47 passed`).

### Open questions
- Notebook cells were syntax-checked but not executed because doing so would rerun expensive training and analysis.

## 2026-08-19 09:34 IST — SGD momentum on ReLUTrainConfig

### Goal
Add a momentum parameter to the ReLU training config for the SGD optimizer.

### User prompts (verbatim)
- "in the relu training config can you add a momentum parameter for use with the sgd optimizer option?"

### Changes
- `src/layer_lenses/relu_training.py`: added `momentum: float = 0.0` to `ReLUTrainConfig`. SGD already passed `config.momentum` into `torch.optim.SGD`.
- `src/layer_lenses/relu_analysis.py`: threaded `momentum` through `run_single_relu_seed`.
- `tests/test_relu_training.py`: default-zero check and a same-init SGD run showing momentum `0.0` vs `0.9` produce different weights.
- `README_detailed.md`: documented that momentum applies only to SGD.

### Decisions
- Default `momentum=0.0` preserves the previous plain-SGD behavior.
- AdamW ignores `momentum`; it is only passed to `torch.optim.SGD`.
- Nesterov remains off.

### Current state / where to pick up
- Full test suite passes (`49 passed`).

### Open questions
- None.

---

## 2026-09-14 IST — Single-path ReLU cell notebook

### Goal
Create a clean notebook for the Track A "ReLU cell" experiments — the ReLU
analog of the single-path DLGN — to test the lens / positive-feedback
narrative in a minimal, fully interpretable model.

### User prompts (verbatim)
- "Do you have access to the chats in which we discuss ReLU network feature learning and in particular setups to validate the notion of positive feedback extending the lens notion of DLGNs? \n\nIf so, create a new jupyter python notebook in the notebooks folder for testing ReLU network single path. Use the other python files in src like odt, relu_training and relu_analysis. You can look at the scratch.ipynb notebook for a rough idea of how they are used. But create a clean version here."

### Changes
- Added `notebooks/ReLU_single_path_runner.ipynb` (36 cells, interleaved
  markdown/code, no stored outputs). Sections: config + run wrapper,
  reference run, loss/dead-path diagnostics, signed-cos trajectories,
  convergence labels (upsilon-tuple) with the root-to-leaf path test,
  capture epochs + ODT tree plot, norm-vs-|cos| scatter, lens leaf
  concentration and its entropy over training, first-layer gradient mass
  by ODT level, layered graph, frozen-downstream control (H11), width-1
  first-layer control, and a mini seed census.
- No `src/` or test changes; the notebook composes existing APIs
  (`run_single_relu_seed`, `first_layer_odt_alignment`,
  `first_layer_neuron_weighted_lens`, `activation_tensor_by_layer_neuron_data`,
  `log_loss_gradients`, `plot_layered_relu_graph`,
  `plot_complete_binary_tree_values`, `collect_start_end_metrics`).

### Decisions
- "Single path" for ReLU is implemented as the **ReLU cell**
  `hidden_dims=[D, 1, 1]` from `notes/research_spec.md` Sec. H Track A, not a
  literal `[1,1,1]` chain: a width-1 chain is a single-index function with no
  lens, so it cannot exhibit the dual-role feedback. The `[1, W, W]` control
  is included to make that degeneracy visible.
- Defaults follow the Track A recommendation: D=4 primary, d=100,
  `bias=False`, full-batch GD, and `weight_decay=0` so alignment cleanup is
  not attributable to the regulariser.
- Labels use **signed** cosine (ReLU has no sign symmetry); `COS_THRESHOLD`
  defaults to 0.85 per the spec's tau.
- Frozen-downstream control reuses the existing `parameter_update_mask`
  plumbing in `ReLUTrainConfig` rather than new training code, and reuses the
  same master seed so data and init match the free run exactly.
- Helper functions (trajectories, labels, capture epochs, lens profiles) are
  kept in the notebook for now; the notebook's closing section flags that they
  should migrate to `src/layer_lenses/relu_analysis.py` once stable, per the
  notebook-migration policy.

### Current state / where to pick up
- Notebook was smoke-executed out-of-place with a tiny config
  (d=20, D=3, n=4000, 60 epochs, CPU): all 36 cells run without error. It is
  committed with outputs cleared and has **not** been run at the real config.
- Next: run at D=4 defaults, then the items in the notebook's final section —
  bespoke near-zero init with `-eps*u_j` perturbations (paper Sec. 3.3 analog),
  second-order probes (H9/H10, CT2), full census with the data-invariance
  filter, and the `[D, 2, 2]` robustness variant.

### Open questions
- Whether `adamw` or plain full-batch `sgd` is the right optimizer for the cell
  census; the DLGN single-path study used plain GD, and Adam's per-coordinate
  scaling may distort the trajectory comparison.
- Whether a `[D,1,1]` run that stalls near `log 2` is a genuine saddle (the
  paper's failure mode) or just a dead downstream gate; the active-fraction
  diagnostic distinguishes them but the census does not yet separate the two.

---

## 2026-09-14 IST — Frozen sign-pattern ReLU cell: single-path trajectory analog

### Goal
Build a notebook for the user's proposed ReLU architecture — d -> D (trainable)
-> 2^D (frozen {+1,-1} bit-vector rows) -> 1 (frozen random {+1,-1}) — as the
ReLU analog of the single-path DLGN trajectory study, with hand-picked
first-layer inits and probes for the hierarchy-seeking / self-opposing lens
theorems.

### User prompts (verbatim)
- "Ok. Can you create a new notebook specifically to test the single path trajectory analog for ReLU nets. In an earlier unlogged conversation I asked:\n\nOk. Consider another question below analogous to the hieararchy-seeking lens theorem. This is applicable to ReLU nets.\n\nLet's say we have a depth D-ODT labelling function, d-dimensional data and a ReLU net (with no bias) with architecture given by : d input nodes, D first layer hidden nodes, 2^D second layer hidden nodes, and one output node. The second layer parameters are frozen such that each of the 2^D nodes has its incoming weights set to a unique D-dimension {+1,-1} bit vector and frozen. The last layer connecting the 2^D neurons to the output is init to a random {+1,-1} vector and frozen. Only the first layer parameters are updated by population gradient flow.   \n\nA similar setup to the DLGN where the first layer neurons are carefully initialised to emphasize hierarchy can be chosen. I will choose the inits here myself by hand.\n\nCan you set this up in a separate notebook, with D=4?"

### Changes
- Added `notebooks/ReLU_single_path_trajectory.ipynb` (38 cells, no stored
  outputs): config, data/ODT basis, frozen layer construction, hand-picked init
  cell, population gradient flow, freeze check, lens-vs-autograd validation,
  parameter and gradient trajectories in the ODT basis, upsilon-tuple and
  capture epochs, lens unpacked per sign pattern and per ODT leaf,
  hierarchy-seeking probe (Thm. 1) with split-half stability, swapped-role
  probe, output-sign sweep, self-opposing probe (Thm. 2), and an init-variant
  sweep.
- No `src/` changes; the notebook composes `generate_cob_odt_data`, `ReLUMLP`,
  `train_relu_mlp` + `ReLUTrainConfig` (with `parameter_update_mask`),
  `log_loss_gradients`, `activation_tensor_by_layer_neuron_data`,
  `first_layer_neuron_weighted_lens`, `plot_complete_binary_tree_values`.

### Decisions
- Trained via `train_relu_mlp` directly rather than `run_single_relu_seed`,
  because the init is hand-built from the ODT basis and must be constructed
  after the tree is known. Gradient flow = full batch + plain SGD, momentum 0,
  weight decay 0, and `lr_scheduler_eta_min_ratio=1.0` so the cosine schedule
  does not decay the step size.
- Init is specified as an ODT-basis coefficient matrix `init_coeffs[k, i]`; the
  off-ODT random component is projected out of the ODT span, so `w_k . u_i`
  equals the hand-set coefficient exactly regardless of `INIT_RANDOM_SCALE`.
- `W1 = 0` is NOT usable for ReLU (h1 = 0 => all layer-2 pre-activations 0 =>
  gradient vanishes identically), unlike the DLGN's W = 0 start. The default
  init is therefore the paper's Sec. 3.3 perturbation plus a small component
  orthogonal to every ODT direction. Documented in the notebook header.
- `first_layer_neuron_weighted_lens` is exactly Lambda_k for this architecture;
  the notebook asserts this by rebuilding -grad from residue x lens x blinder
  x y x and comparing against autograd (agrees to ~1e-8).

### Measured during setup (preliminary, not a result)
- Probe step size matters: at d=100, n=1e5 per half, split-half sign stability
  of the Thm-1 finite difference was 0.95 at eps=1e-2, 0.84 at 1e-3, 0.60 at
  1e-4. Default is now `PROBE_EPS = 1e-2`, and the probe reports its own
  split-half stability with a warning below 0.9.
- The frozen output vector largely decides whether the Thm-1 sign structure
  appears at all. At the paper init, agreement was 0.93 (`all_positive`), 0.94
  (`majority`), 0.76 (`parity`), and 0.37 / 0.70 / 0.50 for random +-1 vectors
  at seeds 178 / 0 / 42. `all_positive` is the direct analog of the paper's
  fixed a > 0. Default stays `"random"` per the stated spec, with the sweep
  cell in Sec. 10 making the comparison reproducible.

### Current state / where to pick up
- Smoke-executed out-of-place at tiny config (d=40, n=20k, 40 epochs, CPU): all
  38 cells run; the stability warning correctly fires at that sample size. Not
  run at the real config.
- Next: user supplies hand-picked `init_coeffs`; then check whether the
  parameter/gradient trajectories reproduce the DLGN staging (saddle, then w1
  toward -u0, then w2 toward -u1, with a brief +u0 backlash).

### Open questions
- Whether the random-sign output is the right specification for the trajectory
  comparison, given the measured sensitivity above.
- `LR = 0.5` for the flow step size is a guess; the right value depends on the
  init scale, which is hand-set.

---

## 2026-09-14 IST — One-hidden-layer control added to the trajectory notebook

### Goal
Add a simpler experiment before the two-hidden-layer one: d -> 4 (trainable)
-> 1 output with a frozen +-1 vector, no 2^D layer.

### User prompts (verbatim)
- "Can you also create a simpler experiment first with just one hidden layer, with 4 units? No second layer with 2^4 units. Directly connect to the output with a similar a vector.  Add this in the cells before the more complicated 2 hidden layer experiments."

### Changes
- Restructured `notebooks/ReLU_single_path_trajectory.ipynb` (38 -> 50 cells):
  shared helpers now precede both experiments, then Experiment A (one hidden
  layer), then Experiment B (2^D frozen sign patterns), then an A-vs-B
  comparison, the Thm-2 probe run on both, and an init sweep run in both
  architectures.
- Helpers generalised to be architecture-agnostic: `first_layer_only_mask`
  takes `hidden_dims`; `second_order_probe` and `hierarchy_seeking_table` take a
  template model and probe data explicitly; added `train_first_layer`, `as_run`,
  `lens_matrix`, `check_lens_reproduces_gradient`, `upsilon_summary`,
  `capture_epoch_table`, `gradient_trajectory`, `report_hierarchy_seeking`.
- Both experiments share one hand-edit init cell (`INIT_COEFFS`), with per-
  experiment override points (`coeffs_A`, `coeffs_B`).

### Decisions
- Experiment A is framed as the **no-lens control**, not merely a smaller model:
  with one hidden layer the lens is the constant `a_k`, so a neuron is a
  separator but nobody's lens. The only remaining coupling is the shared
  residue. The notebook header derives the resulting closed form,
  `-a_k a_l E[sigma' g_k g_l (h.x)(z.x)]`, which vanishes at an ODT-orthogonal
  init except through the label dependence of sigma'.
- `first_layer_neuron_weighted_lens` handles the one-hidden-layer case
  correctly (returns `a_k * blinder`); verified in the notebook by asserting the
  lens takes only values {0, a_k} and by the residue x lens x blinder rebuild.
- The A-vs-B comparison reports median |response| ratio, not just sign
  agreement, because A's predicted response is near zero and sign agreement is
  then meaningless.
- Softened the split-half stability warning: low stability in A is the expected
  outcome (the estimand is ~0), not a misconfigured probe.

### Measured during setup (preliminary, not a result)
- Probes at the paper init, d=100, 1e5 per half, PROBE_EPS=1e-2, no training:
  - A (no lens): Thm-1 agreement 0.483, median |response| 1.8e-06, split-half
    stability 0.566, correlation 0.181.
  - B (lens): agreement 0.365, median |response| 5.0e-03, stability 0.951,
    correlation 0.859.
  - So the cross-neuron second-order coupling is ~2700x larger with the lens
    present; the residue-only channel is negligible, matching the derivation.
  - B's *sign* agreement at this random output vector is below chance, which is
    the `OUTPUT_SIGN_CHOICE` sensitivity logged in the previous entry.

### Current state / where to pick up
- Smoke-executed out-of-place at tiny config (d=40, n=20k, 40 epochs, CPU): all
  50 cells run. Not run at the real config; outputs cleared.
- Next: user's hand-picked `INIT_COEFFS`, then compare the A and B trajectories
  for the DLGN staging.

### Open questions
- Unchanged from the previous entry: random vs all-positive output signs, and
  the `LR = 0.5` flow step size.

## 2026-09-15 IST — ReLU subgradient convention at zero (phi'(0) = 1)

### Goal
Give a bias-free neuron sitting exactly at `w_k = 0` a way to leave the origin
without hand-picking its escape direction, so the paper's Sec. 3.3 init (which
starts `w_1 = 0`) can be run literally with `INIT_RANDOM_SCALE = 0.0`.

### User prompts (verbatim)
- "what derivative does pytorch assign to relu at 0?"
- "is there a modified relu that we can use that behaves the same as relu on a forward pass but assigns a derivative of 1 at 0 instead? that would also be a valid interpretation of relu, right?"
- "yes, add it to src and add the switch to the notebook"

### Changes
- New `src/layer_lenses/activations.py`: `ReLUOneAtZero` (autograd Function with
  `phi'(0) = 1`), `relu_one_at_zero`, an `ActivationSpec` dataclass pairing a
  forward map with its matching blinder predicate, the `RELU` /
  `RELU_ONE_AT_ZERO` specs, `ACTIVATIONS`, and `resolve_activation`.
- `ReLUMLP` takes `activation: str | ActivationSpec = "relu"` and applies
  `self.activation.apply` in `forward`. Default preserves existing behaviour.
- `relu_analysis`: `masked_relu_logits` and `_precompute_hidden_activation_masks`
  now use the model's own spec instead of hardcoding `torch.relu` / `> 0`;
  `run_single_relu_seed` gained an `activation` kwarg for the no-`model_init`
  path.
- New `tests/test_activations.py` (7 tests): forward bit-identity to
  `torch.relu` in float32/float64 including `-0.0`, the gradients differing only
  at zero, name resolution, gate/backward agreement, the zero-row neuron being
  frozen under `relu` and freed otherwise (with other neurons' gradients
  unchanged), and analysis blinders following the model's convention.
- `notebooks/ReLU_single_path_trajectory.ipynb` (50 -> 53 cells): `ACTIVATION`
  config switch (default `"relu_one_at_zero"`), passed to both model builders; a
  post-init cell reporting `||w_k||` and warning if a neuron is at the origin
  under `"relu"`; a diagnostic cell in Experiment A tabulating per-neuron
  gradient norms under both conventions and printing the freed neuron's top ODT
  gradient coordinates; rewritten intro caveat; an ablation note in Sec. 8.

### Decisions
- `phi'(0) = 1` is the upper endpoint of the Clarke subdifferential
  `d phi(0) = [0, 1]`; PyTorch picks the lower one. Neither is canonical, so this
  is a convention switch, not a modified activation. Forward is bit-identical.
- The `ActivationSpec` carries a `gate` predicate alongside `apply` because the
  analysis code reconstructs blinders from pre-activations rather than from
  autograd: `relu` gates on `> 0`, `relu_one_at_zero` on `>= 0`. Keeping these
  in one object is what stops the lens computation from silently disagreeing
  with the gradient.
- Notebook default is `"relu_one_at_zero"` because the user's current config has
  `INIT_RANDOM_SCALE = 0.0` and an all-zero `INIT_COEFFS` row 0, i.e. `w_1 = 0`
  exactly, which `"relu"` would freeze for the entire run.
- Framing: this is an *escape rule for a degenerate point*, not an ongoing
  change. The conventions differ only on `{x : w_k . x = 0}`, measure-zero once
  the neuron is off the origin. At the origin the blinder is 1 everywhere, so
  the neuron takes one step as if linear, seeing the full unmasked
  `E[residue * a_k * y x]`.

### Measured during setup (preliminary, not a result)
- Paper Sec. 3.3 init with `w_1 = 0` exactly, d=100, D=4, n=20k, one-hidden-layer
  model: loss identical under both conventions (0.691647); per-neuron gradient
  norms `[0.0, 0.008037, 0.008244, 0.008180]` under `relu` versus
  `[0.014436, 0.008037, 0.008244, 0.008180]` under `relu_one_at_zero`. Only the
  degenerate row changes, to every printed digit.

### Process note
- Re-running the previous session's scratchpad builder (`build_nb3.py`)
  overwrote the user's own notebook edits (`N_TRAIN` 1e6, `EPS` 0.01,
  `INIT_RANDOM_SCALE` 0.0, `LR` 0.25, flipped init signs, both output presets
  `"all_positive"`). Restored from the committed copy and switched to in-place
  `nbformat` edits. The builder is now stale and was renamed
  `STALE_DO_NOT_RUN_build_nb3.py`; the notebook in the repo is the source of
  truth. **Do not regenerate this notebook from a builder script.**

### Current state / where to pick up
- Full test suite green (56 tests). Notebook smoke-executed out-of-place at tiny
  config (d=40, n=20k, 40 epochs): all 53 cells run. The user's 17 committed
  cell outputs were preserved, not cleared.
- Next: rerun at the real config and use the `ACTIVATION` ablation to check
  whether any hierarchy staging depends on how `w_1` leaves the origin.

### Open questions
- Whether `phi'(0) = 1` biases where `w_1` lands: at the origin the neuron sees
  both half-spaces, so its first step is the full linear signal. Worth comparing
  against several `INIT_RANDOM_SCALE > 0` seeds to see if they agree on the
  landing node.
- Unchanged from previous entries: random vs all-positive output signs, and the
  flow step size.

## 2026-09-17 IST — Math comments for the lens path-sum accumulator

### Goal
Make the `downstream` reverse accumulation in
`first_layer_neuron_weighted_lens` readable without re-deriving it.

### User prompts (verbatim)
- "Can you add some detailed comments expressing mathematically what the "downstream" variable in function first_layer_neuron_weighted_lens of relu_analysis.py is doing? It seems too dense for a quick check."

### Changes
- Comments and docstring only in `src/layer_lenses/relu_analysis.py`; no
  behaviour change. The docstring now states the identity the function computes,
  `Lambda_k(x) = d yhat / d z_0[k]`, its path-sum form, its role in
  `dL/dw_k = E[(dL/d yhat) * Lambda_k(x) * x]`, and the `M = 1` collapse to
  `a_k * g_0[k]`. Inline comments label each loop step
  (`d yhat / d h_i` -> `d yhat / d z_i` -> `d yhat / d h_{i-1}`) with shapes.

### Decisions
- Documented the return as the *weighted* lens (lens x own blinder), not the
  lens alone, since that is what the code returns and what the notebooks
  multiply by the residue.
- No new test: `test_first_layer_weighted_lens_matches_autograd_for_one_to_three_layers`
  already pins exactly this identity by comparing against
  `first_preactivation.grad[:, neuron_id]`.

### Verified
- Re-derived against autograd for `hidden_dims` `[4]`, `[5, 3]`, `[4, 6, 3]`:
  max |manual - autograd| = 0.0, 9.3e-10, 1.9e-09.
- `M = 1` lens takes exactly the two values `{0, a_k}` for every neuron.
- Full suite green (56 tests).

## 2026-09-17 IST — Correction: notebook half of the ACTIVATION entry was overwritten

### What happened
The 2026-09-15 entry above records a `notebooks/ReLU_single_path_trajectory.ipynb`
change (50 -> 53 cells: an `ACTIVATION` config switch, a post-init degeneracy
report, and a both-conventions gradient diagnostic in Experiment A). Those cells
are **not** in the repo. The notebook was open in an editor and a later save from
it overwrote the agent's edits; the file is back to 50 cells with no `ACTIVATION`
constant. The `src/` and `tests/` half of that entry is accurate and intact.

### Current notebook state (the user's own edits, committed here)
- `activation="relu_one_at_zero"` hardcoded at both `ReLUMLP(...)` call sites
  instead of read from a config constant, so the convention is not a one-line
  ablation switch and the two call sites can drift apart.
- `V_NODES = [11, 12, 13, 14]` with the extra tilts moved to `u_13` / `u_14`
  (was `[7, 8, 9, 10]` with tilts on `u_7` / `u_8`). The `# paper's v = ...`
  comment still names the old nodes.
- New `train_all_layers` helper: `parameter_update_mask=None`, so it does train
  every layer; its docstring is a stale copy of `train_first_layer`'s and still
  says "W_1 only".
- No diagnostic for the `w_1 = 0` degeneracy, so with `INIT_RANDOM_SCALE = 0.0`
  nothing in the notebook reports which neurons sit at the origin.

### Process note
Second clobber of this notebook in three days (the first was the agent's stale
builder script, logged above). The notebook is edited interactively; agents
should re-read it immediately before editing and prefer `src/` for anything
durable.

---

## 2026-09-22 12:12 IST — Replayable training: optimizer-state snapshots and one-epoch replay

### Goal
Separate the optimizer's contribution (Adam, mini-batches) from the population
gradient in the ReLU-on-ODT learning-dynamics analysis. Concretely: snapshot the
optimizer state alongside the model state so any snapshot epoch can be re-run
step-by-step, exposing the per-mini-batch gradients, updates, and Adam step
sizes. Also stop storing the data arrays in `seed_result`, which had pushed
pickles past 1 GB.

### User prompts (verbatim)
- "Populate your context from the relevant notes in the project.\n\nI have trained several ReLU nets on the depth 5 ODT dataset with various hyperparameters, and only low batchsize Adam actually trains (current hyperparams in scratch.ipynb).\n\nI am trying to build a theory of feature learning in multiple layers, and even population gradient flow is hard to analyse so Adam with min-batches is beyond my current ability. I do however believe that the fundamental feature learning mechanism is mostly in population gradient flow itself, and the mini-batches and Adam simply enable the existing mechanism in population gradient flow.  How this \"enabling\" happens is also an intriguing question I want to analyse, but later.\n\nEither way, I want to separate out the effect of Adam/SGD from my analysis. I am right now saving a longish run (about 20 minutes) at checkpoint epochs and trying to analyse the gradient. And the gradient (over the entire dataset) at some intermediate epoch seems to be pointing in a very different direction to the difference in parameters between nearby epochs. This is likely due to the effect of the parameters in neighbouring epochs coming from a dynamic update over multiple steps and also Adam step-size shenanigans. \n\nI want a way to see the sequence of actual updates in an intermediate epoch. To this end I need my seed_result to save the state of the optimizer.  That way I can give a checkpoint and get the sequence of gradients, updates, and step-size for that epoch. \n\nCan you create a new notebook with just this training functionality similar to scratch.ipynb? No analysis cells after the loss plot required for now. Also, have the seed_result not store the data, but rather just generate it again when needed. As the file sizes are getting too big to store the snapshots and the majority of it is coming from the data size (1million 100 dimensional data is more than 1 GB). \n\nAlso create helper functions in ReLU_analysis which takes in a snapshot epoch and reruns the training from that point for just one epoch and gives the relevant gradients, updates, and step-sizes etc. \n\nOnly create this new notebook and add functions to ReLU_analysis. Make no other changes."
- "Yes, add the session log entry"

### Changes
- `src/layer_lenses/relu_analysis.py` (+776 lines, appended at the end plus an
  extended import block):
  - `train_relu_mlp_with_optimizer_snapshots` — mirrors `train_relu_mlp` but
    additionally records, at each snapshot epoch, `optimizer.state_dict()` on CPU
    (Adam moments and step counters, or SGD momentum buffers, plus the LR in
    force) and, under `shuffle_each_epoch=True`, the state of the generator that
    draws the mini-batch permutation.
  - `run_single_relu_seed_replayable` — same seed convention and arithmetic as
    `run_single_relu_seed`, but returns `data_spec` instead of `data`, plus
    `model_spec`, the `ReLUTrainConfig` used, and the optimizer snapshots.
  - `regenerate_seed_result_data` — rebuilds the train/eval split from
    `data_spec`, asserting the regenerated ODT matches the stored tree. Falls
    through to a legacy `seed_result["data"]` if present.
  - `replay_epoch_from_snapshot` — restores model + optimizer at the start of a
    snapshot epoch, re-runs that epoch's mini-batches, and records per step: the
    gradient the optimizer consumed (post projection/mask), the parameter update,
    the optimizer state after the step, the Adam decomposition
    (`adam_step_size`, `adam_direction`, `adam_update`, `weight_decay_update`),
    and optionally the full-dataset population gradient at that step's starting
    parameters. Verifies the replayed end-of-epoch parameters and Adam moments
    against the next stored snapshot.
  - `epoch_replay_summary_table` — one row per step: gradient/update norms and
    cosines between update, mini-batch gradient, population gradient, and the
    whole-epoch parameter change; Adam per-coordinate step-size quantiles and the
    weight-decay fraction of the update.
  - `epoch_replay_parameter_series`, `flatten_named_tensors`, and the
    `save_seed_result_pickle` / `load_seed_result_pickle` pair (moved in from
    `scratch.ipynb` cell 1), plus small private helpers
    (`_build_relu_optimizer`, `_relu_training_step`, `_epoch_permutation`,
    `_generate_cob_odt_split`, `_adam_step_decomposition`, ...).
- `notebooks/ReLU_replayable_training.ipynb` (new, 10 cells): training-only
  counterpart of `scratch.ipynb` with the same hyperparameters (seed 5178, d=100,
  depth 5, 1M train, [40,40,40], 10k epochs, Adam, batch 10k, cuda). Imports,
  config, run, save/load, data regeneration, loss plot, then a single usage cell
  demonstrating `replay_epoch_from_snapshot` + `epoch_replay_summary_table`.

### Decisions
- **The replay trainer lives in `relu_analysis.py`, not `relu_training.py`.** The
  user asked for changes confined to the new notebook and `relu_analysis`, so
  `train_relu_mlp` is untouched and the replayable variant imports
  `relu_training`'s private step helpers (`_relu_training_step` is a thin wrapper
  around the same projection → mask → step → restore sequence). **Consequence:
  if `train_relu_mlp`'s step arithmetic changes, `_relu_training_step` must be
  updated in lockstep or the bit-for-bit parity below silently breaks.**
- **Data is regenerated, never stored.** `data_spec` holds `(dim, depth, n_train,
  data_seed, threshold, odt_hyperplanes)` — everything
  `run_single_relu_seed` uses to build the split. The regenerated tree is checked
  against the stored `tree` so a stale spec fails loudly instead of silently
  analysing the wrong data. Pickles drop from >1 GB to parameters + 2x parameters
  per Adam snapshot.
- **`shuffle_each_epoch` is a flag, default `False`.** The committed
  `train_relu_mlp` shuffles; the user's uncommitted working-tree change replaces
  that with a fixed identity permutation. The default matches the working tree.
  With shuffling on, replay is still exact because the permutation generator's
  state is snapshotted separately from the global RNG.
- **Adam step size is reported as the per-coordinate effective LR**
  `lr / (sqrt(v_hat) + eps)`, with the update split into the Adam part and the
  decoupled weight-decay part, so the two can be compared against the population
  gradient independently. Dead coordinates (zero second moment, e.g. dead ReLU
  rows) give `lr/eps` and dominate the max column; this is real AdamW behaviour,
  noted in the docstring.

### Verified
- Checkpoints and epoch losses from `run_single_relu_seed_replayable` are
  **bitwise identical** to `run_single_relu_seed` for {adam, sgd} x {cpu, cuda}.
- Regenerated `x_train/y_train/x_eval/y_eval` are exactly equal to the arrays the
  legacy runner stored.
- `replay_epoch_from_snapshot` reproduces the next snapshot with
  `max|param diff| = 0.0` and `max|optimizer state diff| = 0.0` in all four
  combinations, and also after a pickle round trip where the data is regenerated
  rather than passed in. Same with `shuffle_each_epoch=True`.
- Adam decomposition residual `|adam_update + wd_update - update|` ~ 3e-8
  (float32 rounding); `sum(step updates) - (params_after - params_before)`
  ~ 1e-10.
- Full suite green (56 tests). The new notebook executes end to end
  out-of-place at a tiny config (d=20, depth 3, 4k train, 20 epochs); the
  in-repo copy has no outputs.

### Current state / where to pick up
- The tooling answers the question that prompted it: for a given snapshot epoch,
  `epoch_replay_summary_table` shows `cos_grad_full_grad` (mini-batch vs
  population gradient) and `cos_update_epoch_delta` (single step vs the epoch's
  net parameter change) side by side, which is where the "gradient points in a
  very different direction to the parameter difference" discrepancy should
  resolve.
- Next: run the real config and inspect an intermediate epoch. Expect the
  population gradient and the epoch delta to disagree mostly through the Adam
  preconditioner rather than through mini-batch noise; the step-size columns are
  there to test that.

### Open questions
- Whether the per-coordinate Adam step size should be summarised per *parameter
  tensor* (or per first-layer neuron) rather than globally — the current
  min/median/max over all coordinates is coarse, and the dead-coordinate max is
  not informative.
- No tests were added for the new helpers (the user scoped the change to the
  notebook and `relu_analysis`). The parity and exact-replay checks currently
  live only in a scratchpad script, not in `tests/`.

---

## 2026-09-23 13:38 IST — "sgd_adam" hybrid optimizer: SGD on early layers, AdamW on later ones

### Goal
Test whether Adam's per-coordinate rescaling is actually useful in the *first* layer.
The first layer's targets are the ODT's random orthogonal normals, which are generic
with respect to the coordinate basis, so a coordinatewise preconditioner has no
reason to help there. Later layers are different: ReLU acts coordinatewise, so the
axis directions are genuinely special. Add an optimizer that splits on exactly that
boundary.

### User prompts (verbatim)
- "At epoch 0 (from random init) with decay of 0, shouldn't cos_update_grad be -1 at the first update?"
- "I am unable to make sense of why Adam with its co-ordinate rescaling alone is useful in this setting. The \"correct\" hyperparameters for the first layer are random orthogonal vectors, and damping co-ordinate directions is unlikely to induce the parameters to go there faster. For the later layers however, the axis directions are special because of ReLU. \n\nI want to test a variant of optiminsation, let's callit sgd-adam where the first layers are trained by sgd (with momentum) and later layers are trained by adam. \n\nCan you change the relu_training.py and also the new replay training notebook to take in such an option and the hybrid training code?"

### Changes
- `src/layer_lenses/relu_training.py`:
  - `ReLUTrainConfig` gains `sgd_adam_split_layer` (default 1), `sgd_lr`, and
    `sgd_weight_decay` (both `None` = reuse `lr` / `weight_decay`).
    `optimizer` now accepts `"sgd_adam"` alongside `"adamw"` and `"sgd"`.
  - New: `normalized_optimizer_name`, `split_named_parameters_sgd_adam`,
    `optimizer_group_names`, `primary_optimizer_index`, `build_relu_optimizers`,
    `optimizer_parameter_names`, `build_relu_schedulers`, `relu_training_step`,
    `suggest_sgd_lr_for_adam_parity`.
  - `train_relu_mlp` refactored onto those helpers and now also returns
    `lr_history_by_group`.
- `src/layer_lenses/relu_analysis.py`: the replay path is multi-optimizer.
  `_build_relu_optimizer`, `_is_adam_config`, and the duplicated
  `_relu_training_step` were **deleted** in favour of the `relu_training` versions,
  closing the drift risk flagged in the previous entry. `optimizer_snapshots[e]` is
  now a *list* of state dicts, with `_as_optimizer_state_list` accepting the old
  single-dict form so pickles written yesterday still replay. `replay_epoch_from_snapshot`
  returns `lr_by_group` and `parameter_groups`; the Adam decomposition is restricted
  to AdamW-managed parameters. New `build_seed_model_and_data` exposes the untrained
  model plus data for pre-run diagnostics, and `run_single_relu_seed_replayable` now
  calls it. Both seed runners accept the three new hybrid kwargs.
- `notebooks/ReLU_replayable_training.ipynb` (10 -> 13 cells, edited in place, user's
  edits and outputs preserved): `OPTIMIZER` documented as `"adam" | "sgd" | "sgd_adam"`,
  new `SGD_ADAM_SPLIT_LAYER` / `SGD_LR` / `SGD_WEIGHT_DECAY` knobs, a new cell that
  measures a step-size-matched `SGD_LR` before the expensive run, per-group LR
  reporting next to the loss plot, and a closing per-group replay comparison.
- `tests/test_relu_training.py` (+6 tests) and `tests/test_relu_analysis.py` (+4 tests,
  one parameterized over all three optimizers). These are the first tests covering the
  replay machinery, which the previous entry listed as an open gap.

### Decisions
- **The split is by hidden-layer index, not by a hand-built parameter list.**
  `sgd_adam_split_layer=k` puts hidden layers `[0, k)` on SGD and everything after,
  including the output layer, on AdamW. `k` is restricted to `[1, len(hidden_dims)]`:
  the degenerate ends are just `"adamw"` and `"sgd"` and must be asked for by name.
- **Two optimizers and two schedulers, not one optimizer with two param groups.**
  PyTorch cannot mix algorithms within one optimizer. Each gets its own
  `CosineAnnealingLR` annealing from its own base LR, so the shape of the schedule is
  shared but the scales are independent.
- **`sgd_lr` defaults to `lr` and this is deliberately a bad default.** Per AGENTS.md
  "do not silently guess": the honest default is the value the user already supplied.
  But Adam's step is ~`lr` per coordinate regardless of gradient size while SGD's is
  `sgd_lr * |g|`, so reusing `lr` would test "a tiny step instead of Adam" rather than
  "SGD instead of Adam". `suggest_sgd_lr_for_adam_parity` measures `lr / median|g|`
  over the SGD-managed parameters, and the notebook runs it before the expensive cell.
  At the real config this came out around 100, not 1e-3.
- **SGD weight decay is called out separately** because it is coupled, unlike AdamW's
  decoupled decay. The per-step shrink factor is `1 - sgd_lr * sgd_weight_decay`, so a
  step-matched `sgd_lr` near 100 with `weight_decay = 0.01` would shrink the SGD layers
  by ~100% per step. The helper reports `decay_fraction_per_step` and the notebook
  warns above 0.1.
- **Snapshot format changed to a list**, with backward-compatible reads. Writing a
  single dict for the non-hybrid case would have made the reader's job ambiguous.

### Verified
- **The refactor of `train_relu_mlp` is bitwise identical to the pre-change code.**
  Captured epoch losses, LR history, and SHA-256 of every checkpoint tensor for
  {adam, sgd} x {cpu, cuda} x {both seed runners} before touching anything, and
  re-compared after each of the two refactoring passes. Identical both times.
- Hybrid replay reproduces the next snapshot with `max|param diff| = 0.0` and
  `max|optimizer state diff| = 0.0`, including after a pickle round trip.
- Layer 0 under the hybrid follows plain SGD exactly: `max|update + sgd_lr * g|` was
  1.5e-8, and `cos(update, -grad) = 1.0`. Layer 1 follows the AdamW `t=1` sign step
  `-lr * g / (|g| + eps)` instead, to 1e-7.
- SGD-managed parameters carry `momentum_buffer` and no `exp_avg`; AdamW-managed ones
  carry `exp_avg` / `exp_avg_sq` / `step`. The Adam decomposition covers exactly the
  AdamW group.
- Full suite green, 68 tests. Notebook smoke-executed out-of-place at a tiny config in
  both `"sgd_adam"` and `"adam"` modes, zero cell errors; the in-repo copy keeps the
  user's own outputs.

### Measured during setup (preliminary, not a result)
- Tiny config (d=20, depth 3, 4k train, [8,8,8], batch 500), replaying epoch 10, mean
  over the epoch's 8 steps of `cos_update_full_grad`, the cosine between the step
  actually taken and the full-dataset gradient at that step:

  | run | first layer | all AdamW layers |
  |---|---|---|
  | pure Adam | -0.61 | -0.61 |
  | sgd_adam, `sgd_lr = 20` | -0.96 | -0.60 |

  So the SGD-trained first layer tracks the population gradient far more closely than
  the same layer under Adam, which is the intended contrast. This is a smoke-config
  number on a 20-dimensional problem and says nothing yet about whether the hybrid
  trains the real depth-5 task.

### Current state / where to pick up
- Next: run the real config with `OPTIMIZER = "sgd_adam"`. Use the new `SGD_LR` cell
  first, set `SGD_WEIGHT_DECAY = 0.0`, and reconsider `MOMENTUM = 0.999`, which is very
  high for an `sgd_lr` of order 100.
- The interesting comparison is whether the hybrid trains at all, and if it does,
  whether the first layer reaches the ODT normals faster or slower than under Adam.

### Open questions
- Whether Adam's benefit here is the *rotation* (coordinate rescaling) or merely the
  *scale* (a gradient-magnitude-independent step). The hybrid at a step-matched
  `sgd_lr` separates these: it keeps Adam's scale in the later layers while removing
  the rescaling from the first. A pure-SGD run at the same matched LR would be the
  third arm and is not yet set up.
- Whether the split should sit at the first layer or after it. `sgd_adam_split_layer`
  is configurable, but nothing has been run with `k > 1`.
- Cosine annealing shares one schedule shape across both groups. If SGD and Adam want
  different decay profiles, that is not currently expressible.

---

## 2026-09-25 07:10 IST — Analysis sessions: rattling, PGF emulation, and the level-4 second wave

### Goal
Read-only analysis over three days (Sep 24–25) of the replayable SGD and Adam runs, to
answer whether the feature-learning phenomena are properties of population gradient
flow, and why Adam outperforms SGD. Only repo change: the analysis scripts were copied
into `notebooks/analysis_scripts/` at the user's request.

### User prompts (verbatim)
- "At epoch 0 (from random init) with decay of 0, shouldn't cos_update_grad be -1 at the first update?"
- "Ok, with the right learning rate choice SGD with momentum is not a total failure and gets to within factor 2 of the final loss and a few percentage of test error when compared to Adam. While adam still has an advantage, the investigation of feature learning with SGD is sufficient for now. \n\nWe do seem to be needing mini-batches though. Population gradient seems to be struggling even at same number of updates. \n\nAs a case study I am analysing Neuron 37 in the first hidden layer (in the ReLU_replayable_training notebook) which is one of the first neurons to go towards an ODT vector (the root node u_0 here).  I am trying to analyse what makes this neuron go there in the thrid from last cell in the notebook. Epoch 140 is my epoch to analyse. \n\nIntriguingly, I find.\n\n<pasted_content id=\"6635\">\nCos between total_update and epoch_delta: tensor(1.0000)\ntotal_update norm: tensor(0.0269) epoch_delta norm: tensor(0.0269)\nCos between total_update and odt_wts[0]: tensor(-0.9569)\nCos between total_grads_over_steps and odt_wts[0]: tensor(0.9524)\nCos between total_pop_grads_over_steps and odt_wts[0]: tensor(0.9012)\n</pasted_content id=\"6635\">\n\nWhile the population gradient  at every one of the 100 steps align only very roughly with u_0, the total aligns quite strongly at a cosine of about 0.9. Why do you think this is the case? It cannot be just averaging over data points as the population gradient is computed over the entire trainset at each step and is hence average already. \n\nFeel free to run snippets from the notebook but do not change any file or write any code for now. Just discuss candidate hypotheses for now."
- "I am slightly confused, I see no epoch where the cosine of neuron 37 with u_0 swings from -0.2 to -0.9. At epoch 75 it swings from -0.65 to -0.67 only."
- "So, in your opinion the causes for rattling that happens in the later stages (around epoch 140) are  the following, right?:  \n\n1. finite data causing non-smoothness in the objective (the actual expected objective is quite smooth here even with the ReLU model and sharp label boundaries in the data)\n2. finite step-size \n\n\nSo, in a theoretical analysis of population gradient flow, these would be absent?"
- "Ok, I want to isolate and study the effect of population gradient flow (from a random init). \n\nWould these interesting happenings that we see empirically (relu neurons capturing ODT vectors, and low loss/error) happen even there? i.e Is it fair to say mini-batch SGD with momentum (or Adam) simply are ways to emulate the running of population gradient flow with finite data and compute time? I can see that there exists inits that would completely stop PGF in a way that would not stop mini-batch SGDM, but such inits are rare and unlikely to happen at a random init, right?"
- "Actually, I was wrong before. I don't know why I felt the previous runs with full batch SGD failed. I did another run now, and it does ok. You can check current config in ReLU_replayable_training notebook and  the most recent picklefiles.\n\nIt does perform a bit worse in both train loss and test error when compared to mini-batch SGD, but it might just be due to insufficient epochs."
- "Yes, I also observed the lack of deepest-level nodes in the gradient descent runs. Adam does capture a few of them though, and I speculate that to be one of the reasons for Adam to outperform SGD. Did another run with full-batch replayable run in the most recent version of the notebook. Both train loss and test error are much lower."
- "I have a meta question. Where do I find this scratchpad of your that you use for running these smaller experiments? While what you are saying is believable, I am still not used to trusting AI."
- "Yes, copy the scripts into notebooks/analysis_scripts/"

### Changes
- New `notebooks/analysis_scripts/` (16 scripts + `README.md` index). Copied verbatim
  from the agent's scratchpad; every number below was printed by one of them. They
  are evidence, not reusable code (hard-coded paths, run from `notebooks/`).

### Findings (all on seed 5178, d=100, depth 5, 1M train, [40,40,40])
Runs referenced: MB-SGD (batch 10k, lr 0.02, mom 0.9, 10k epochs), FB-SGD (batch 1M,
same lr/mom, 100k epochs), FB-Adam (batch 1M, lr 0.02, 100k epochs), MB-Adam (batch
10k, lr 0.002, wd 0.01, 10k epochs). Pickles in `notebooks/results/cache/`.

1. **Adam's first step is a sign step**, `-lr * g/(|g|+eps)`, so `cos_update_grad`
   is `-||g||_1/(sqrt(n)||g||_2)`, measured -0.68, never -1. Plain SGD at wd 0 gives
   exactly -1 (float32 differencing of small updates shows -0.99995).
2. **Epoch-140 rattling of neuron 37** (MB-SGD, after it has aligned to u_0 at
   cos -0.994): the per-step population gradient is 3:1 a restoring force on the
   neuron's own orientation (cos +0.91 with its displacement from the epoch mean),
   with wobble period ~54 steps, matching heavy-ball theory at the measured curvature
   (48). Driven by white mini-batch noise 3.5x the restoring force; the epoch sum
   cancels it and keeps the radial drift, hence the 0.90 cosine with u_0 in the sum.
   Causes: mini-batch noise x step size (amplitude ~ lr*sigma^2/curvature reproduces
   the 0.02 wobble), momentum for the periodic character. Not finite-data
   non-smoothness. At epoch 75, mid-arrival, the population gradient is constant to
   ~1% across the epoch: no wobble.
3. **Arrival of neuron 37 at u_0 (epochs 55–100)** is not a straight rotation: the
   largest per-epoch update component is out of the (w, u_0) plane, first shedding
   the init, then pointing at a diffuse mix of deepest-level normals.
4. **Plateau escape is population gradient flow.** Median over 40 neurons of
   cos(displacement from init, -population gradient at init) is 0.999 at epoch 5,
   0.993 at 20, 0.93 at 60; the first mini-batch gradient has cos 0.17. The
   population gradient at init lies 89% in the span of the 16 level-4 normals
   (consistent with labels = sign of the last split), so u_0 is invisible to the
   linearized flow and is captured later via the nonlinear dynamics (pop-grad cos
   0.65 with -u_0 at epoch 70). Neuron 37 had one of the smallest initial gradients.
5. **FB-SGD vs MB-SGD at matched flow time** (lr summed over steps / (1-mom)): the
   same init, identical to 4 decimals for every neuron through flow time ~3200 (past
   u_0 capture and the first loss drop); first neuron diverges ~3600; by the end
   ~15 neurons differ but node coverage is identical (levels 0–3 complete, level 4
   empty). At equal flow time FB-SGD is *ahead* (0.169/0.050 vs 0.227/0.091); the
   user's earlier "full-batch struggles" was a 10x flow-time deficit. MB-SGD needs
   ~2x the flow time to match, at ~1% of the compute.
6. **Level-4 capture is an Adam-vs-SGD difference, not a noise difference.** FB-Adam
   (no noise) reaches 0.0038/0.0046, equal to MB-Adam, 10x better than MB-SGD.
   Dedicated level-4 neurons: SGD 0/16 in both runs, FB-Adam 8/16, MB-Adam 16/16.
   All four first layers lie inside the 31-dim ODT span (energy outside = 0.000).
   SGD's row space contains all level-4 directions but at 2–3% of the root's
   response strength; Adam 18–50%.
7. **Three phases of first-layer level-4 energy**: rise along the init gradient (all
   runs), suppression while coarse features are built and purified (all runs; SGD
   ends *below* its init level-4 energy and still falling at 100k steps), then a
   second rise under Adam only (FB-Adam share 0.025 -> 0.11 from epoch 10k, MB-Adam
   0.027 -> 0.41), coinciding with the loss falling from ~0.1 to <0.01. A replayed
   step at epochs 20k/60k shows Adam's update carries 3.8x/6x the level-4 share of
   its gradient with positive growth; SGD's update has the gradient's composition
   and shrinks level 4.
8. Test error binned by nearest on-path split level: every run has ~3x its overall
   error near level-4 hyperplanes; the SGD/Adam error ratio is roughly uniform across
   levels. So Adam refines the whole function; level-4 capture is the visible
   signature, not a level-specific fix.

### Decisions
- Scripts were copied, not cleaned up, so they remain exactly the code that produced
  the logged numbers. The README says so.

### Current state / where to pick up
- Open experiments suggested: (a) `sgd_adam` hybrid with the first layer on SGD, to
  test whether the level-4 second wave needs the preconditioner in the first layer
  or arrives through the residue from downstream Adam layers; (b) a much longer
  FB-SGD run watching only level-4 energy, to see whether gradient flow ever starts
  phase three.

### Open questions
- Whether population gradient flow reaches level 4 on any timescale, or is
  permanently confined to levels 0–3 from this init.
- Whether the "purification" that suppresses level 4 in phase two is the same
  mechanism as neuron 37 shedding its init component (measured, not proven).

---

## 2026-09-28 13:39 IST — Init sweep: the u_0 winner is an init property, decided during the plateau

### Goal
Set up and run the two-part experiment Harish designed: (A) establish that which
first-layer neuron captures u_0 depends only on the init for a fixed optimizer, and
(B) generate many inits with full parameter trajectories to analyse what makes the
winner win. Budget: one afternoon of GPU, about a quarter on A.

### User prompts (verbatim)
- "Ok. I can see that Adam does something extra, but even a lens level understanding of why PGF works in multilayer networks is still open. I  will push the analysis of the extra oomph that Adam gives for later and settle for analysing PGF for now."
- "Ok, now back to the problem at hand.\n\nI want to derive the set of conditions on the neural network under which a particular neuron, after moving along the population gradient an infintesimal amount will be more aligned to u_0 than before.\n\nOr an equally good result in the style of the lens theorem of the DLGN. Give some set of conditions on an init, and an infinitesimal change to it. After this infintesimal change the population gradient of a neuron is more aligned to u_0 than before. \n\nI would like a computational backbone to generate such ideas for inits and changes. How do you suggest we go about this?"
- "Before we do that, let's build the experimental setup. I am thinking of the following steps:\n\n\n1. Get 10 random inits for the network.\n2. For each random init do some SGD momentum runs till the log loss hits 0.3. Each of these runs will be different in the following way: different samples for x (same ODT though), different batch sizes. Let's say two different batch sizes, and 3 different data seeds for a total of 6 runs per init seed.\n3. Each of these runs look at the neurons that have hit cosine sim > 0.95 with u_0. I am guessing we will have such a node in every run. Ideally, we will find the same neuron would have aligned with u_0 in all 6 runs.\n4. Then we can analyse what properties of the init made that neuron go towards u_0.\n\n\nHow do you propose we do this? start a separate notebook? Any other suggestions/modifications?"
- "All your points sound good. Go ahead and implement it. Use about one fourth the compute to verify the hypothesis that \"which neuron goes towards u_0 is purely a property of the init (for a given optimisation algo). \n\nAnd the rest of the compute for generating inits and the eventual u_0 node analysis.  Assume a total compute time of one afternoon ( say 4 hours).\n\nKeep these two separate. If we believe the hypothesis, the former set of experiments is just for assuaging the reader and needlessly detailed. The latter can be lighter and less rigorous with random seed testing as we would have established in the former that other randomness is irrelevent for a given init.  \n\nBut do keep the entire parameter trajectory (second, third and output layer weights too) for the latter set of experiments. We will need it for lens analyses."

(Three meta questions about `/usage`, where the scratch scripts run, and what the
agent can delete were answered in the same session and are not research content.)

### Changes
- New `src/layer_lenses/init_sweep.py` (committed `2ecf0dc`, extended after):
  `SweepRunSpec` with independent tree / data / init / train seeds (the ODT is built
  once from `tree_seed` and passed to the generator, which `run_single_relu_seed`
  cannot do); `train_sweep_run` with constant LR, snapshots every k optimizer steps,
  early stop on trailing one-epoch loss <= `stop_loss`, first-layer-only or
  all-layer storage; `.npz` persistence and a resumable `run_specs` driver; CLI
  `python -m layer_lenses.init_sweep part-a|part-b`; analysis helpers
  `capture_summary`, tie-aware `leading_set`, `first_layer_agreement`,
  `part_a_invariance_table`, `init_neuron_features`, `winner_rank_table`,
  `winner_prediction_cv` (leave-one-init-out conditional logit).
- New `tests/test_init_sweep.py`, 10 tests on a tiny CPU config. Suite: 78 passing.
- New `notebooks/init_sweep_analysis.ipynb`, executed in place on the full sweep.
- `notebooks/results/init_sweep/`: the two `summary.csv`, the per-neuron feature
  table (8000 rows), manifests and the launcher `run_all.sh`. Raw trajectories (2 GB)
  stay in the gitignored `results/init_sweep/`.

### Decisions
- **Constant LR, step-indexed snapshots.** So batch sizes 1k and 10k share flow time
  per step and are compared at equal step counts. The earlier notebook runs used
  cosine annealing, but by the capture step it had decayed by <1%, so the runs match.
- **Batch sizes 1k and 10k, not full batch.** Full batch to loss 0.3 costs ~35k
  million-point steps per run and was already shown to coincide with 10k through
  the capture. A 10x batch contrast is a real noise contrast and costs 42 s a run.
- **Leading set instead of a single winner.** In Part B, two neurons cross within
  500 steps of each other in 130 of 200 inits, so ties are the normal case.
  `capture_summary` keeps the first crosser; `leading_set` keeps everyone within
  the window; both are reported.
- **Budget reallocation.** Runs cost 42 s, so Part A got 6 inits x 3 data seeds x
  2 batch sizes (36 runs, 25 min) and Part B 200 inits (2h20m), rather than 10 and
  a handful.
- The per-neuron "output weight" feature was replaced by the fan-out column norm
  into the next layer: for three hidden layers a first-layer neuron has no output
  weight, and indexing `output_layer.weight` by first-layer index was a bug caught
  by the tests on unequal widths.

### Results
**Part A (invariance).** In 35 of 36 runs the first neuron to reach |cos| >= 0.95
with u_0 is the same as in the init's reference run. The exception is init 6204,
where neurons 5 and 9 race in every run and swap order once by 400 steps. Every
run's leading set meets the reference's (36/36). The *second* capture is not
init-determined: for init 6203 the runner-up differs in four of six runs and
arrives 4000-19000 steps later. Batch size barely perturbs the trajectory (worst
row-wise cosine vs reference at the capture step 0.98-0.99 with the same data
sample); a different data sample perturbs it visibly (33-39 of 40 neurons below
0.99 by the capture step, worst 0.66), consistent with a small persistent
difference in the empirical objective being amplified in the sub-dominant modes
during the exponential escape while the dominant one, the winner, is unchanged.

**Part B (200 inits).** All 200 reached loss 0.3 and all captured u_0. First
crossing at step ~10k-13k; 1, 2 or 3 neurons in the leading set in 69, 130, 1
inits; 2-5 neurons on u_0 by loss 0.3.

Winner's percentile rank among its 39 siblings, pooled over inits (sign test):

| feature | mean pct | frac above median | p |
|---|---|---|---|
| disp_norm_plateau | 0.84 | 0.945 | 5e-43 |
| norm_ratio_plateau | 0.82 | 0.930 | 2e-39 |
| abs_cos_u0_plateau | 0.77 | 0.845 | 3e-24 |
| abs_cos_u0_init | 0.73 | 0.800 | 3e-18 |
| lens_std_init / lens_abs_mean_init | 0.67 | 0.74 / 0.73 | 1e-11 |
| grad_norm_init | 0.66 | 0.725 | 2e-10 |
| norm_init, signed cos features, neg_grad_cos_u0_init | ~0.5 | ~0.5 | n.s. |

Leave-one-init-out prediction of the winner (chance top-1 = 0.025):

| features | top-1 | top-1 in leading set | median rank | top-3 |
|---|---|---|---|---|
| abs_cos_u0_init alone | 0.115 | 0.16 | | 0.26 |
| all init features, linear | 0.15 | 0.24 | 7 | 0.30 |
| plateau-end features | 0.37 | 0.58 | 3 | 0.57 |
| init + plateau | 0.39 | 0.60 | 2 | 0.61 |

Reading: the winner *is* a deterministic function of the init (Part A), but the
static init features tried here recover only a small part of it. By the end of the
plateau, ~50 epochs of deterministic flow later, the winner is largely visible: it
is the neuron that has moved and grown the most. So the selection happens through
the coupled plateau dynamics, not the initial geometry, which is exactly the
lens-mediated second-order effect the derivation in the previous prompt targets.
Also: the earlier single-neuron anecdote (neuron 37 having one of the smallest
init gradients) does not generalise; winners tend to have *larger* init gradients
and lens magnitudes, and a larger |cos| with u_0 at init.

### Current state / where to pick up
- Everything is in `results/init_sweep/{partA,partB}`; the notebook reproduces all
  tables from the run files. Features are cached in
  `partB/init_neuron_features.csv`.
- Next, per the derivation plan: compute the autograd sensitivity of each neuron's
  u_0 alignment rate at init, and test whether *that* predicts the winner where the
  static features do not. If it does, the theorem is about the sign structure of
  that sensitivity.
- Alternative cheap predictor to try: the linearized flow. Run each init for
  ~1000 full-batch steps (a few seconds) and use the resulting displacement ranks.

### Open questions
- Why the plateau-end winner is only 37% predictable from its own plateau-end
  features: the remaining variance may be in the leading-set ties (top-1 in
  leading set is 0.58) and in the second half of the escape.
- Whether the data-sample sensitivity of non-winning neurons matters for lens
  analyses that read the other neurons' orientations at the capture step.

---

## 2026-09-28 15:40 IST — Autograd sensitivity of the u_0 alignment rate: weak at init, winner decided mid-escape

### Goal
Test whether the second-order sensitivity of each neuron's u_0 alignment rate at
initialization predicts the winner where the static features could not.

### User prompts (verbatim)
- "What are the nine static init features? How are they combined to make one score for each neuron? What do you mean by \"leave-one-init-out\" prediction test?"
- "How is the \"plateau\" defined? Exactly 50 epochs?"
- "Compute the autograd sensitivity of each neuron's u_0 alignment rate at init and test whether it predicts the winner where the static features could not"

### Changes
- `init_sweep.py`: `alignment_rate_sensitivity` (forward-mode over reverse-mode via
  `torch.func.jvp` over `torch.func.grad`, tangent = minus the full-sample gradient,
  chunked exactly) returning per neuron `A0`, `A_dot` (first and second time
  derivatives of |cos(w_k,u_0)| under the flow), `B0`, `B_dot` (same for log norm);
  `alignment_rate_flow_difference` (central finite difference along the flow with
  a displacement of `rel_step * ||theta||`).
- `tests/test_init_sweep.py`: the derivative must match a float64 finite difference
  with a 1e-7 relative step, which stays on one linear piece. Suite: 79 passing.
- Notebook: two new sections (sensitivity at init with rank/prediction tables; rates
  over time with a decision-time plot). `_fit_conditional_logit` switched from
  gradient ascent to damped Newton (identical results, ~30x faster). CSVs `alignment_sensitivity.csv`,
  `alignment_sensitivity_fd.csv`, `rates_over_time.csv` under `results/init_sweep/partB`.

### Decisions
- **The "plateau end" is a loss threshold**, first snapshot with trailing loss
  <= 0.69, not a fixed epoch: median 43 epochs, range 25-64, about 43% of the way to
  the winner's crossing. A dynamical landmark (the linear-to-exponential switch of
  the layer displacement) would be earlier and cleaner; not implemented.
- **Derivative vs finite difference.** A ReLU net's population Hessian has a
  gate-boundary term that the piecewise-linear jvp cannot see. Small finite steps on
  a 100k sample are dominated by kink-crossing noise (rank agreement with the jvp
  0.23 at a 1e-4 relative step, magnitudes 5.7x); at a 1e-2 step, comparable to
  what the flow does over the plateau, the FD agrees with the jvp at Spearman 0.85
  with magnitudes 1.3x. So the boundary term is a ~30% correction here and the jvp
  is the cleaner estimator. The float64 test uses a 1e-7 step for that reason.

### Results (200 inits, winner = first crosser)
Winner's mean percentile among 39 siblings at init: `B_dot` 0.77 (p 6e-25),
`abs_cos_u0_init` 0.73, `A0` 0.65 (p 4e-6), `A_dot` 0.60 (p 1e-3). Raw argmax of
`A_dot` is the winner 11.5% of the time, the same as `abs_cos_u0_init` alone.
Leave-one-init-out top-1: dynamical four 0.19, static nine 0.15, both 0.275
(top-3 0.455, median rank 4); plateau-end five 0.365; everything 0.385. (The
finite-difference variant of the dynamical features gives 0.22 for "both".)

Rates over time, raw argmax top-1 / in-leading-set of `A0` and of |cos| itself:

| step | 0 | 2000 | 4000 | 5000 | 6000 | 7000 | 8000 |
|---|---|---|---|---|---|---|---|
| A0 | .12/.18 | .18/.29 | .20/.33 | .27/.44 | .37/.59 | .44/.62 | .36/.55 |
| abs_cos_u0 | .12/.16 | .12/.16 | .13/.18 | .23/.33 | .41/.62 | .57/.79 | .78/.92 |
| B_dot | .10/.16 | .12/.23 | .15/.26 | .18/.26 | .23/.36 | .17/.24 | .09/.14 |

Reading: the second-order sensitivity at init carries real but weak information,
comparable to the static angle to u_0, and does not close the gap. The winner
becomes identifiable between steps 5000 and 8000, roughly a third to a half of
the way to its crossing and after the loss-defined plateau end, i.e. during the
exponential escape. The best *early* indicator is `B_dot`, the acceleration of
norm growth, not the alignment rate. So the "condition on the init" is not a
low-order local quantity at theta_0; it is the outcome of the amplification phase.

### Current state / where to pick up
- The right sensitivity is spectral, not a second derivative: the dominant growing
  mode of the flow linearized around the mid-plateau state (Jacobian of the flow
  field via jvp, power iteration), and which neuron's u_0-tangential coordinates it
  loads on. Not implemented.
- `A_dot` turns negative for the winner once its cosine saturates, so it is a bad
  late indicator by construction; `A0` and |cos| are the late indicators.

### Open questions
- Whether the winner's identity is fixed at the linear-to-exponential switch
  (~epoch 25-30 in the studied run) or only during the exponential phase. The
  rates-over-time table suggests the latter, but a spectral test would settle it.

---

## 2026-09-28 17:30 IST — Designed init: steering a probe neuron to u_0, and what stops it

### Goal
The reverse of the init-sweep question: construct a small-perturbation-of-zero init in
which a chosen first-layer neuron (the probe) moves toward u_0 under population gradient
flow, then vary it by flipping signs of groups.

### User prompts (verbatim)
- "Ok, considering this analysis is not likely to be straightforward. I want to try the reverse first. \"Can we construct initializations that are minor perturbations of 0, such that (say) first neuron of the first hidden layer moves towards u_0?\" \n\nConsider the below init scheme.\n\nneuron 1 : initialised to a small random perturbation that is orthogonal to all the ODT hyperplanes. We want to make this neuron go towards u_0 under population gradient flow.\n\nneurons 2 to 33: initialised to a small scalar factor of -u_15, +u_15, -u_16, +u_16 ,..., -u_30,+u_30.\n\nneurons 34 to 40 : random vectors. All neurons normalised to have same norm\n\nSecond hidden layer: Each neuron has an incoming weight vector in {-1,0,+1}^40. Have 10 of these neurons connect to neuron 1 of hidden layer 1 with weight +1, and some subset of neurons in the previous layer corresponding to -u_15 to -u_22 with weight +1, and zero elsewhere. \n\nHave another 10 of these neurons connect to neuron 1 of hidden layer 1 with weight +1, and some subset of neurons in the previous layer corresponding to +u_23 to +u_30 with weight +1, and zero elsewhere. \n\nHave another 10 of these neurons connect to neuron 1 of hidden layer 1 with weight -1, and some subset of neurons in the previous layer corresponding to -u_23 to -u_30 with weight +1, and zero elsewhere. \n\nHave another 10 of these neurons connect to neuron 1 of hidden layer 1 with weight -1, and some subset of neurons in the previous layer corresponding to +u_15 to +u_22 with weight +1, and zero elsewhere. \n\nThird hidden layer: Each neuron is given a random vector in {0,1}^40.\n\nOutput layer : 40 dimensional all ones vector. \n\nWe can then try variations of the above init by systematically flipping signs of groups of neurons."

### Changes
- New `src/layer_lenses/designed_init.py`: `DesignedInitSpec`, `build_designed_weights` /
  `build_designed_model` (the construction above, exactly; `group_sign` sets the probe's
  weight into groups A-D, fractional values allowed; `subset_mode` random or all),
  `logit_scale`, `odt_coordinates`, `probe_gradient_report` (ODT decomposition of the
  probe's negative gradient, per group by zeroing the other groups), `run_designed_flow`
  (full-batch constant-LR GD via `train_relu_mlp`), `neuron_trajectory_table`,
  `sign_flip_variants`.
- New `tests/test_designed_init.py` (4 tests). Suite: 83 passing.
- New `notebooks/designed_init_u0.ipynb`, executed in place. Flow sweep script and
  outputs under `results/designed_init/` (gitignored), summaries and trajectories
  copied to `notebooks/results/designed_init/`.

### Correction
An earlier message in this session said the label is *minus* the sign of the last
split. It is *plus*: leaf offsets (not leaf indices) alternate by parity, so ``y = +1``
iff the on-path level-4 split is positive. The 2026-09-25 and morning 2026-09-28 log
entries say "the sign of the last split", which is fine; nothing in the code was wrong.

### Decisions
- **eps = 1e-3.** Layers 2-4 are O(1) as specified, so the logit RMS is ~1400 * eps;
  eps 1e-3 gives RMS 0.14 and a residue near 1/2. eps 1e-2 starts at loss 0.91.
- **Full-batch GD, lr 3e-4, no momentum, 1500 steps** (60 ms per step). The probe's
  trajectory is settled by step ~200; a 12000-step base run confirms the endpoint is a
  fixed point. lr 1e-4 and 3e-4 give the same endpoint.
- **Random subsets, seed 0** for the group wiring and layer 3.

### Results
**Init gradient (t = 0), 16 sign patterns.** A group whose probe weight is +1 is
structurally inert: a unit receiving +ReLU(probe) is on whenever the probe is on, so it
cannot carry information about the probe's input. Groups with -1 gate the level-4
features relative to the probe and are informative. Group A gates on left-subtree
negative-side neurons, B on right-subtree positive-side, C on right-subtree negative-side,
D on left-subtree positive-side. When informative, A and B push the probe toward -u_0 and
C and D toward +u_0. The measured cos(-g, u_0) at init
matches this rule in sign for all 16 patterns: base (C,D informative) +0.293, A+B
informative -0.215, all four -0.020 (cancel), none +0.017; a single informative
left-subtree group gives only -0.02.

**Under the flow.** Every pattern with at least one inert (+1) group stalls at
|cos(probe, u_0)| = 0.36-0.52 with ~0.8 of the probe's direction on level-4 normals,
within ~100 steps. The three patterns with no inert group reach |cos| 0.88-0.93 with
level-4 energy 0.13-0.21 and norm growth x33 (vs x8): all-(-1) -> -0.932, (0,0,-1,-1)
-> +0.887, (-1,-1,0,0) -> -0.879. The endpoint sign matches the init-gradient sign in
all 10 patterns with |init cos| > 0.05; the balanced patterns go either way.

**Knob.** Shrinking the inert groups' probe weight w with C, D fixed at -1:
w = 1.0 -> final cos 0.43; 0.3 -> 0.48; 0.1 -> 0.89; 0.03 -> 0.90; 0 -> 0.89. Note the
init cos *falls* along this knob (0.29 -> 0.13) while the endpoint jumps.

**Lesson.** What decides whether the probe reaches u_0 is not the size of the u_0
push at init but the absence of a constant component in its lens. A constant lens pulls
the probe toward E[y x], which lies on level 4, and that pull wins the competition.
Only the input-dependent part of the lens, units gated by other features relative to
the probe, pulls toward a coarse direction. This is the lens theorem's content in the
ReLU setting, and the designed init makes it a knob.

### Current state / where to pick up
- With no inert groups and balanced left/right informative groups, the pole is decided
  by second-order effects, which makes that configuration the natural testbed for a
  lens-theorem-style statement about which perturbation decides the pole.
- Untested variations: `subset_mode="all"`, other seeds for subsets and layer 3,
  and whether the same design works with a random (non-level-4) second neuron set.

### Open questions
- The mechanism by which even the all-inert pattern reaches cos 0.52: the probe first
  acquires level-4 content, after which its own gate correlates with level-4 signs and
  its own blinder can supply the root-side modulation. Plausible, not measured.

---

## 2026-09-29 10:40 IST — Layers 3-4, learned wiring statistics, and the two-probe design

### Goal
Answer whether layers 3 and 4 of the designed init have a better strategy, mine the
learned checkpoints for wiring inspiration, and change the design goal to two probes,
one racing to +u_0 and one to -u_0.

### User prompts (verbatim)
- "In all these runs does the loss actually get minimised to a value below 0.1? Is probe the only first layer neuron that goes to u_0? how much does the norm of the probe neuron grow by from about 10^{-3} at init?"
- "Ok, I am thinking of an even more elaborate init, where similar probes are constructed for every internal node from 0 to 14 (two for each), and not just zero. The probes for these internal nodes would also start at a random vector orthogonal to all ODTs, but the second layer would be for appropriately designed so that these 28 probes would go running off to -u_j or +u_j for some internal node j. Total number of first layer hidden neurons would by 62 -- one pair for each ODT vector.\n\nKey thing that makes this init not just \"we choose an init close to solution, and gradient takes us to solution\" is that no first layer neuron has any component along u_0 to u_14. So the network has to pick up cues for it entirely from level four nodes at the init. We will have to later somehow argue that it is reasonable to expect even a totally random init to get components along level 4 ODT directions, as these have a (small) global edge.\n\nAny comments on this design strategy?"
- "Layer 3 weights are being set to {0,1} and layer 4 weights being set to all-ones. Is there a better strategy than this?"
- "Also, in the current init design the incoming weights for the second hidden layer from A,B,C,D groups first layer neurons are always +1 or 0 -- only weights from probe can be negative.  Similarly for layer 3 also, all non-negative weights only are studied. \n\nCan we take inspiration from any of the learned parameters from already stored checkpoints to craft a better scheme? \n\nI did notice that in the learned networks, the outgoing weights from the eventual u_0 aligned vectors are largely negative. So I agree with your scheme of using mostly negative values for the probe outgoing weights. \n\nFor now modify the goal of the init design to have two probes with one of them racing towards +u_0 and another racing towards -u_0."

### Changes
- `designed_init.py`: `CueGroup`, `MultiProbeSpec`, `cue_index`, `multiprobe_labels`,
  `build_multiprobe_weights` / `build_multiprobe_model` (n probes, identity or random
  0/1 layer 3, explicit output scale, `cue_mode` "relu" or "linear"), the four
  group templates `LEFT_NEG`, `RIGHT_POS`, `RIGHT_NEG`, `LEFT_POS`, `two_probe_spec`,
  `multiprobe_gradient_report`. +1 test (5 in the file). Suite: 84 passing.
- Notebook `designed_init_u0.ipynb`: sections 5 (layers 3-4, checkpoint wiring
  statistics) and 6 (two probes), executed in place; trajectories and
  `summary_twoprobe.csv` under `results/designed_init/`.

### Results
**Answers to the three questions.** The loss never goes below 0.1 in any designed run:
0.696 -> 0.68 (stalled) or 0.67 (u_0-reaching), and the 12000-step base run sits at
0.6804 from step 1000 on. These are stationary small-weight states; the design
decides directions, not the function. In the stalled patterns no other neuron reaches
u_0; in the no-inert single-probe patterns 5-8 cue neurons follow the probe to the same
pole at cosines 0.74-0.98. Probe norm from 1e-3: x8-13 stalled, x32-36 when it reaches
u_0; cue neurons collapse to 0.1x in the stalled base case, grow 5x otherwise; random
fill neurons never move (zero fan-out).

**Layers 3 and 4.** Identity layer 3 with output 20 matches random 0/1 layer 3 with
unit output to three decimals on all three patterns tried. Fan-in randomness is
irrelevant at the root; the path-sum scale sets the flow time per step (squared).
An intermediate claim in the session that identity layer 3 "fixes the cue collapse"
was a flow-time artifact of unit path sums and is withdrawn. A balanced-sign output
(so the initial logit has zero mean) gives init cosine 0.93 but the probe shrinks to
zero as it aligns, at the same flow time for lr 3e-4 and 3e-5; not understood, not
used.

**Learned wiring (200 Part B runs).** Outgoing layer-2 weights of u_0-aligned neurons
are 46-48% negative by count and 37-50% by magnitude, mean ~0, both poles, the same as
every other neuron; layer 3 and the output are 48% and 50% negative. So the observation
that they are "largely negative" is not borne out here (it may hold for the Adam runs,
not checked). Structure that is there: at loss 0.3 a unit reading two opposite-pole u_0
neurons gives them opposite signs 66% of the time (77% by magnitude), i.e. it builds a
linear readout of u_0.x from the ReLU pair; same-pole pairs get the same sign (84% by
magnitude). At capture, same-pole pairs get opposite signs (91% by magnitude).

**Two probes.** Probe 0 wired with -1 to C- and D-type groups, probe 1 to A- and
B-type, identity layer 3, output 20, eps 1e-3, lr 3e-4, 1500 steps:

| cue mode | init cos, probe 0 / 1 | final cos | probe norm/eps | cues median norm/eps | cues on u_0 | loss end |
|---|---|---|---|---|---|---|
| relu | +0.05 / -0.08 | +0.925 / -0.926 | 35 / 35 | 3.1 | 0 | 0.669 |
| linear | +0.42 / -0.43 | +0.903 / -0.901 | 47 / 46 | 12.4 | 0 | 0.652 |

Both probes reach their poles, anti-aligned at -0.94 / -0.91, with no cue neuron
dragged to u_0 (unlike the single-probe no-inert case) and the random fill inert.
Linear cues give an eight-times stronger first-order push and a lower loss.

### Decisions
- Keep all-ones output with an explicit scale; set each probe's pole by the (side,
  sign) of its informative groups; never share a group between probes.
- Advice on the 62-neuron design (recorded in the session, not implemented): for a
  probe at node j, pair a group gated on j's left-child subtree with one on its
  right-child subtree so the ancestor pushes cancel by symmetry (use all cues under
  each child, not random subsets); use -1 probe weights only; wire each probe only to
  its own groups; expect deep probes to move 2^-level slower; freeze the 32 cue rows
  with `parameter_update_mask` for a first clean run; the random-init bridge is
  supported by the measured 89% level-4 content of the init gradient, with the caveat
  that random inits acquire the single aggregate vector E[y x], not per-node cues.

### Current state / where to pick up
- `two_probe_spec` is the base for the multi-node design; `CueGroup` already expresses
  per-probe groups with arbitrary cue sets, so a node-j probe is a matter of choosing
  cue sets under j's children.
- Open: why the balanced-sign output kills the probe; whether linear cues also cancel
  ancestor pushes at depth.

---

## 2026-09-29 11:05 IST — Six probes: +-u_0, +-u_1, +-u_2

### Goal
Extend the two-probe design with four probes for nodes 1 and 2, with smaller cue sets.

### User prompts (verbatim)
- "Ok, great can you now add 4 more probes corresponding to -u_1,+u_2,-u_2,+u_2? These will have similar connection structure but with different (smaller) sets of A and B for ODT nodes 1 and 2. \n\nNumber of neurons split up: probes : 6, cues : 32, random: 2"

(Read as -u_1, +u_1, -u_2, +u_2.)

### Changes
- `designed_init.py`: `level4_descendants`, `node_probe_groups(probe, node, pole)`
  (the balanced pair for any internal node; reproduces the root A/B and C/D pairs),
  `SIX_PROBE_TARGETS`, `six_probe_spec` (root groups of 4, node groups of 3, 40 rows
  exactly), `probe_target_report`, `cue_rows`, and a `freeze_first_layer_rows`
  option on `run_designed_flow` (trainer parameter update mask). +2 tests. Suite: 86.
- Notebook section 7; sweep `results/designed_init/run_six_probe.py`, summaries and
  trajectories copied under `notebooks/results/designed_init/`.

### Decisions
- **Generalized pole rule.** For a unit gated on a level-4 cue of sign s under a child
  of node a on side `side`, the push on a -1-weighted probe toward u_a has sign
  `-s * side`. Pole +1 pairs negative-side cues under the right child with
  positive-side cues under the left child; pole -1 the reverse. Ancestor pushes cancel
  because both children sit on the same side of every ancestor; descendant pushes
  cancel by using every cue under each child.

### Results
**t = 0 (both cue modes).** Every probe's gradient points at its signed target
(relu 0.15-0.16 for node probes, 0.05-0.08 root; linear 0.35 node, 0.42 root) with
|cos u_0| < 0.02 for the node probes: the ancestor cancellation is exact.

**Under the flow** (identity layer 3, output 20, eps 1e-3, lr 3e-4), final cosine
with the signed target:

| run | +u0 | -u0 | +u1 | -u1 | +u2 | -u2 | node-probe cos u_0 | cues with coarse top node |
|---|---|---|---|---|---|---|---|---|
| relu, cues free, 1500 | 0.89 | 0.87 | 0.04 | 0.19 | 0.55 | 0.01 | up to 0.75 | 18/32 |
| linear, cues free, 1500 | 0.87 | 0.78 | 0.60 | 0.56 | 0.62 | 0.59 | 0.21-0.27 | 0/32 |
| linear, cues free, 6000 | 0.93 | 0.91 | 0.54 | 0.54 | 0.47 | 0.60 | 0.35-0.47 | 14/32 |
| relu, cues frozen, 1500 | 0.33 | 0.34 | 0.31 | 0.35 | 0.36 | 0.33 | < 0.05 | 0 |
| linear, cues frozen, 6000 | 0.64 | 0.66 | 0.52 | 0.52 | 0.52 | 0.54 | < 0.07 | 0 |

Reading: (1) relu cues fail for the node probes once the flow runs; the network-wide
u_0 signal captures them and rotates 18 cue neurons toward coarse nodes. (2) Linear
cues make every probe end with its target as the top node. (3) With frozen cues the
design is clean, no leakage, but every probe converges to a fixed point at cosine
~0.52 (node) / 0.65 (root): the level-4 stall, since a fixed lens has a fixed constant
part. (4) Freeing the cues lets the root probes escape the stall to 0.93 but injects a
u_0 component into the node probes that grows with time (equal for both poles of a
node, so second order), via the cues drifting toward coarse nodes. Rebalancing group
sizes to 2/4 made the root probes fragile and did not help.

### Current state / where to pick up
- The six-probe design is correct at first order and under the flow with frozen
  cues, up to the level-4 stall. The stall is the thing to attack: the constant part
  of each probe's lens. Candidates: a second-layer structure whose units are *off*
  by default (e.g. a small negative bias-like cue, or pairing each unit with an
  antagonist) so the constant lens vanishes; or unfreezing the cues only after the
  probes have committed.
- The u_0 leakage into node probes with free cues is a second-order effect worth a
  probe of its own: it is equal for +u_1 and -u_1, so it is not about the pole.

### Open questions
- Whether the stall cosine (0.52 for node probes) is set by the ratio of modulated to
  constant lens and can be predicted from the group wiring.

---

## 2026-09-29 16:23 IST — Full hierarchy design under SGD: 30 probes, 48 runs

### Goal
Two probes for every internal node 0..14, widths (80, 40, 40), relu vs linear cues,
cue neurons frozen vs free, several designed inits, trained with small-batch SGD
with momentum to loss 0.3 or a cap, all within two hours.

### User prompts (verbatim)
- "Can you explain what the ReLU and Linear cues are?"
- "ok, so the linear cues/relu cues are just different initialization schemes for the second hidden layer?"
- "Ok. lets just go all the way once and add two probes for each internal node, and try both relu and linear cues for second layer init strategy. Try both variants: neurons corresponding to ODT level 4 nodes held frozen at init and also updating like regular parameters. Have the number of hidden neurons in each layer be 80,40,40.  The last 18 neurons in hidden layer 1 are just random vectors at init.\n\nTry for several such inits. These inits will vary by an additional orthogonal component to all the first layer neurons, and the selected subsets chosen randomly in layer 2 and 3.\n\nTrain using SGD with momentum and small batch size till train loss goes lower than 0.3 or one run takes too much time. This way we can run more inits for a given time.\n\nTarget all the runs to be finished in 2 hours max."

### Changes
- `designed_init.py`: `HierarchyDesignSpec`, `hierarchy_unit_cues` (one balanced unit
  per probe: both children's cue sets in the same unit, so ancestor pushes cancel
  inside it), `build_hierarchy_weights` / `build_hierarchy_model`, `hierarchy_probe_table`,
  `hierarchy_labels`, `HIERARCHY_TARGETS`, `NODE_LEVEL`.
- `init_sweep.train_sweep_run` accepts a prebuilt `model` and `freeze_first_layer_rows`.
- +2 tests (suite: 88). Notebook section 8. Sweep `results/designed_init/run_hierarchy.py`;
  summaries, manifest, log and per-run JSON under `notebooks/results/designed_init/hierarchy/`;
  the 48 run files (5 MB each, all layers every 1000 steps) stay in the gitignored
  `results/designed_init/hierarchy/runs/`.

### Decisions
- **Second-layer allocation.** 40 units for 30 probes: one unit per probe with the union
  of both children's cue sets, which is balanced by itself; units per probe by level
  4 / 2 / 1 / 1 = 8 + 8 + 8 + 16 = 40 rows. Deeper probes get fewer units and a
  smaller region, so their signal is weaker by construction.
- **Scale.** Layers 2-4 keep the {-1,0,+1} / {0,1} / ones sign structure but are scaled
  by 1/3, 1/sqrt(20), 1/sqrt(40) so rows have order-one norms and the standard SGD
  settings (batch 1000, lr 0.02, momentum 0.9) are stable; eps = 0.01 gives a logit RMS
  of 0.006-0.011 at init. Without this rescaling the first layer would need a learning
  rate 100x smaller than the rest.
- **Inits vary** by the probes' off-span directions, a 0.2-relative off-span component
  added to every cue, the 18 random neurons, the random cue subsets per unit (at least
  one per child), and the random 0/1 third layer; one data sample for all.
- **Budget.** Pilot: 900 steps/s, loss 0.42 at 60k steps and flattening, so the 0.3
  target is not reachable within the budget; cap 100k steps (~110 s), 12 seeds x 4
  conditions = 48 runs, wall-clock guard 110 min. Actual: 100 min, no run reached 0.3.

### Results (means over 12 seeds; "on target" = the probe's top ODT node is its target)

| condition | loss at cap | L0 on target | L1 | L2 | L3 | cues rotated to coarse nodes | random neurons coarse-aligned |
|---|---|---|---|---|---|---|---|
| linear, cues free | 0.402 | 0.96 | 0.38 | 0.24 | 0.12 | 24 / 32 | 5.8 / 18 |
| linear, cues frozen | 0.425 | 1.00 | 1.00 | 0.93 | 0.38 | 0 | 14.4 / 18 |
| relu, cues free | 0.404 | 0.71 | 0.19 | 0.27 | 0.14 | 20 / 32 | 7.4 / 18 |
| relu, cues frozen | 0.426 | 1.00 | 1.00 | 0.96 | 0.30 | 0 | 14.7 / 18 |

Mean cosine with the signed target, frozen cues: 0.999 / 0.998 / 0.93-0.95 / 0.37-0.41
by level, with zero spread across seeds at levels 0-1. Alignment at 10k steps is already
at its final value for levels 0-2.

Reading: (1) With the cue neurons frozen the design works down to level 2 under real
SGD, in both cue modes, and the relu/linear distinction stops mattering, unlike in the
full-batch flow where frozen cues stalled at cosine 0.5. The difference is that here
layers 2-4 are at order-one scale and adapt at the standard rate, so the lens is not
fixed even though the cues are. (2) With the cues free, 20-24 of the 32 cue neurons
rotate to coarse nodes within the run and the node probes mostly lose their targets;
linear cues help at level 1 (0.38 vs 0.19). (3) Level-3 probes, one unit with two cues
and a 1/8 region each, mostly fail in every condition. (4) The frozen runs reach a
slightly higher loss (0.425 vs 0.402): 32 fixed level-4 neurons cost capacity, and
14 of the 18 random neurons take over coarse roles in those runs. (5) No run reached
loss 0.3 in 100k steps; the random-init runs at the same settings did so in 35-47k.

### Current state / where to pick up
- The strongest design so far: frozen cues, either cue mode, levels 0-2 captured by
  construction under SGD. Level 3 needs more units or more cues per unit.
- Next questions: unfreeze the cues after the probes commit (say at 10k steps) and see
  whether the targets survive; give level-3 probes 2 units each at the expense of the
  root; and whether the frozen-cue networks, trained longer, reach the loss of the
  random-init runs.

### Open questions
- Why free cues rotate to coarse nodes so quickly under SGD (20-24 of 32 by 100k
  steps) and whether that is the same drift seen in the flow experiments.

---

## 2026-09-29 18:29 IST — Releasing cues at 10k, a wider second layer, and standard-scale versions

### Goal
Three follow-ups to the hierarchy sweep: rerun the frozen conditions with the cue rows
released at 10k steps; test whether one second-layer unit per probe limits the loss by
widening to 120 units; and rescale the whole design to standard-init row norms.

### User prompts (verbatim)
- "Unfreeze the cues at 10k steps and rerun the frozen conditions. \n\nAlso, do you think one neuron in the second hidden layer per (most of the) probe(s) is affecting the ability to reduce loss further ? If so, run a separate set of experiments where you increase the number of neurons in the second hidden layer to 120. Just do one random init seed for now."
- "Ok, once the sweep is done, also try some limited experiments with the same init scheme but scale all layers appropriately, so that the init looks feasible from a standard initialization scheme like the Kaiming He init or Xavier-Glorot init."

### Changes
- `init_sweep.train_sweep_run`: `unfreeze_at_step` (drops the update mask at that step).
- `designed_init.py`: `HierarchyDesignSpec.row_norms` (per-row rescaling of every layer,
  sign structure kept), `standard_row_norms(scheme, spec)` for "default" (the model's
  own init, row norm 1/sqrt(3)), "kaiming" (sqrt(2)) and "xavier", and
  `standard_scale_spec`. +2 tests. Suite: 90 passing.
- Sweeps `run_hierarchy2.py` (24 released-cue runs, width 40, 12 seeds x 2 cue modes;
  6 width-120 runs, seed 0) and `run_hierarchy_scale.py` (6 runs: default / kaiming
  x free / frozen / released, linear cues, seed 0). Notebook sections 9-10.

### Results
**Released at 10k, width 40, means over 12 seeds** (on-target fraction by level):

| cues | loss at cap | L0 | L1 | L2 | L3 | cues rotated to coarse nodes |
|---|---|---|---|---|---|---|
| free | 0.402 / 0.404 | 0.96 / 0.71 | 0.38 / 0.19 | 0.24 / 0.27 | 0.12 / 0.14 | 24 / 20 |
| frozen | 0.425 / 0.426 | 1.00 | 1.00 | 0.93 / 0.96 | 0.38 / 0.30 | 0 |
| released at 10k | 0.399 / 0.403 | 1.00 | 1.00 | 0.97 / 0.99 | 0.61 / 0.43 | 15 / 11 |

(linear / relu). Releasing keeps levels 0-2 on target in every seed (zero spread at
levels 0-1), recovers the loss of the free condition, and raises level-3 capture
from 0.35 at release to 0.61 at the end for linear cues. The cues drift after release
(11-15 of 32) without taking the probes with them.

**Width 120, seed 0** (12/6/3/3 units per probe): loss 0.419 / 0.434 / 0.392 for
free / frozen / released with linear cues, against 0.403 / 0.419 / 0.401 at width 40.
Levels 0-2 perfect when frozen or released; level 3 0.375-0.5. Width does not lower the
loss; the free condition even loses one root probe. So one unit per probe is not what
limits the loss.

**Standard scale, seed 0, linear cues:**

| scheme | row norm | logit RMS at init | steps to loss 0.3 | loss at end | on target at 10k, L0-L3 | on target at end |
|---|---|---|---|---|---|---|
| eps = 0.01 (previous) | 0.01 | 0.006 | not reached | 0.40-0.42 | 1 / 1 / 1 / 0.19 | 1 / 1 / 1 / 0.56 (released) |
| default, free | 0.577 | 0.067 | not reached | 0.422 | 1 / 1 / 0.88 / 0.88 | 1 / 1 / 0.25 / 0.06 |
| default, frozen | 0.577 | 0.067 | not reached | 0.430 | 1 / 1 / 1 / 0.94 | 1 / 0.5 / 0.38 / 0.31 |
| default, released | 0.577 | 0.067 | not reached | 0.414 | 1 / 1 / 1 / 0.94 | 1 / 0.5 / 0 / 0.06 |
| kaiming, free | 1.41 | 2.40 | 34k | 0.297 | 1 / 0.5 / 0.75 / 0.5 | 1 / 0.75 / 0.62 / 0.19 |
| kaiming, frozen | 1.41 | 2.40 | 37k | 0.288 | 1 / 0.25 / 0.5 / 0.44 | 1 / 0.5 / 0.88 / 0.62 |
| kaiming, released | 1.41 | 2.40 | 46k | 0.299 | 1 / 0.25 / 0.5 / 0.44 | 1 / 0.5 / 0.5 / 0.38 |

Reading: (1) the loss floor near 0.4 in every earlier hierarchy run was a scale effect:
at Kaiming scale the same wiring reaches 0.3 in 34-46k steps, the range of random
inits at these settings, whatever the cue treatment. (2) At the default scale the
design captures all four levels by 10k steps, including 88-94% of level-3 probes,
the best early capture seen, but the level-2/3 probes drift off later while the loss
keeps falling slowly. (3) At Kaiming scale the early capture is weaker (the large
logit, 2.4 at init, means the residue is far from 1/2) and the end state depends on
the cue treatment; frozen cues give the best final hierarchy (1 / 0.5 / 0.88 / 0.62).
One seed only.

### Decisions
- Standard-scale means per-row normalization to the scheme's row norm with all signs
  kept; "default" = the model's `reset_parameters` (std 1/sqrt(3 fan_in)), which is
  what the random-init baselines used.

### Current state / where to pick up
- Best protocol for the hierarchy at small scale: freeze the 32 cues, release at 10k.
- The scale finding needs seeds: whether Kaiming-scale designed inits reliably reach
  0.3 faster than random inits, and whether the default-scale early capture of level 3
  can be kept (e.g. by releasing later, or a learning-rate drop at 10k).
- Open: why deep probes drift off at default scale after capturing early.

---

## 2026-09-29 20:28 IST — Kaiming scale halved, four seeds, trained toward loss 0.05

### Goal
Rerun the Kaiming-scale hierarchy design for more seeds with every layer scaled down by
2, training until the loss reaches 0.1-0.05 if feasible, within two hours.

### User prompts (verbatim)
- "Run the Kaiming scale experiments for a few more seeds. Scale the init weights down by a factor 2 in every layer. Run till loss reaches even lower values of around 0.1 to 0.05 (if at all it happens in a reasonable time frame).\n\nTotal runtime budget of 2 hours max."

### Changes
- Sweep `results/designed_init/run_hierarchy_kaiming_half.py` (row norm 0.707 in every
  layer, linear cues, seeds 0-3, cues free vs frozen-then-released at 10k, batch 1000,
  lr 0.02, momentum 0.9 constant, stop at trailing loss 0.05 or 600k steps, 110-min
  guard). Notebook section 11. Summaries and per-run wiring under
  `notebooks/results/designed_init/hierarchy_kaiming_half/`. No code changes.

### Results (8 runs, 77 min)
No run reached 0.3, let alone 0.05: every run hit the 600k-step cap with the loss at
0.373-0.384 and still the minimum at the cap (free 0.376, released 0.383 on average).
This is below the ~0.40 of the eps = 0.01 and default-scale runs at 100k steps but far
above the full-Kaiming runs, which reached 0.3 by 34-46k steps. Row norm 0.707 gives a
logit RMS of about 0.15 at init (four layers, so 1/16 of Kaiming's 2.4), i.e. it sits
in the small-logit regime with the default scale, and trains like it.

Hierarchy capture, on-target fraction by level (means over seeds):

| when | free | released at 10k |
|---|---|---|
| 10k steps | 1.00 / 1.00 / 0.88 / 0.94 | 1.00 / 1.00 / 0.69 / 0.70 |
| 100k | 1.00 / 0.56 / 0.28 / 0.17 | 1.00 / 0.62 / 0.22 / 0.06 |
| 600k | 1.00 / 0.56 / 0.25 / 0.14 | 1.00 / 0.62 / 0.25 / 0.03 |

The early capture at 10k with free cues is the best seen, all four levels, but by 100k
the level-2 and level-3 probes have drifted off and stay off; 20 of 32 cues end on
coarse nodes in both treatments. Releasing at 10k does not help here, unlike at
eps = 0.01.

### Reading
- Scale controls two things in opposite directions. Small logits at init (default,
  half-Kaiming, eps 0.01) give the cleanest early capture of the designed hierarchy and
  a loss that plateaus near 0.38-0.42 on a constant learning rate; the full Kaiming
  scale starts with a logit RMS of 2.4, captures the deep levels less cleanly early,
  and trains to 0.3 quickly. With constant LR 0.02 none of the designed inits reached
  0.1; the random-init baseline that reached 0.047 used cosine annealing over 1M steps.
- The deep-probe drift after early capture is now seen at three scales; it is a
  property of the training dynamics at this learning rate, not of the init scale.

### Current state / where to pick up
- To test whether the plateau is a constant-LR noise floor rather than a property of
  the init, rerun one half-Kaiming and one full-Kaiming run with cosine annealing (the
  replayable trainer supports it; `train_sweep_run` is constant-LR by design).
- To keep the early capture, try a learning-rate drop at 10k rather than releasing
  cues, or freezing the probes' own rows after capture.

## 2026-09-30 03:10 IST — One long Kaiming run per cue treatment, cosine annealing

### Goal
Test whether the ~0.38-0.42 loss floor of the designed hierarchy inits is a constant-LR
floor: one long run at full Kaiming scale with cosine annealing, cues free and cues frozen.

### User prompts (verbatim)
- "Just do one full and long Kaiming run with cosine annealing. Run both frozen and free cues version of the experiment."
- "The previous run did not finish running even after a long time. Can you figure out the issue and fix it and redo?"

### Changes
- Script `results/designed_init/run_hierarchy_kaiming_cosine.py`: `standard_scale_spec("kaiming")`,
  linear cues, seed 0, cues free vs frozen for the whole run (parameter update mask on the
  32 cue rows), `train_relu_mlp` with 1000 epochs x batch 1000 (1M steps), lr 0.02 cosine
  to 0.001, momentum 0.9, snapshots of all layers every 10 epochs, test error on a fresh
  200k sample. Notebook section 12. Outputs copied to
  `notebooks/results/designed_init/hierarchy_kaiming_cosine/`. No library code changes.
- Bug fixed: the first launch trained the free run to completion (18 min) and then crashed
  on `np.savez` because the `runs/` subdirectory was never created (earlier scripts saved
  through `save_run`, which creates parents). Added `os.makedirs` and relaunched; the
  crashed log is kept as `results/designed_init/hierarchy_kaiming_cosine_crashed.log`.

### Results (2 runs, 17 + 26 min)

| cues | final epoch loss | test error | epochs to 0.3 / 0.2 / 0.1 / 0.05 | on-target ep10 (L0-3) | on-target end |
|---|---|---|---|---|---|
| free | 0.0078 | 0.0064 | 35 / 52 / 120 / 318 | 1.00 / 0.50 / 0.75 / 0.50 | 0.50 / 0.50 / 0.38 / 0.31 |
| frozen | 0.0104 | 0.0067 | 37 / 51 / 135 / 317 | 1.00 / 0.25 / 0.50 / 0.44 | 0.50 / 0.25 / 0.62 / 0.44 |

- The floor was the constant LR. With annealing both runs go to loss ~0.01 and test error
  0.6-0.7%, below the random-init cosine baseline (0.047 at 1M steps). Free and frozen
  cues differ little in loss; freezing costs ~0.003.
- Node coverage (first-layer neurons with |cos| > 0.95 to some tree normal) at the end:
  free 60/80 neurons aligned covering all 31 nodes (24 probes, 22 cues, 14 random);
  frozen 76/80 aligned covering all 31 nodes (28 probes, 32 cues, 16 random). Coverage of
  all 31 nodes is reached by epoch 300 (free) and epoch 100 (frozen).
- The "on-target 0.5" at the root is not drift: the +u_0 probe sits at cosine 1.00 with
  norm 17-19, while the -u_0 probe never captured u_0 and was recruited to another node
  (also at cosine 1.00). About half the probes keep their designed node; the rest and the
  random neurons fill in the remaining nodes. With free cues 17 of 32 cues end on coarse
  nodes; the level-4 nodes are then covered by probes and random neurons instead.

### Reading
- The designed init decides *which* neurons take *which* node only partially; what it
  decides robustly is that the hierarchy gets learned in full and fast (0.3 by epoch 35,
  matching the 34-46k steps of the constant-LR Kaiming runs). Below 0.3 the annealed LR
  is what lets the remaining nodes be captured and the loss go to 0.01.
- The earlier finding that the deep probes "drift off" at constant LR should be re-read:
  the network reassigns nodes among neurons; the drifted probe's node is picked up by
  another neuron.

### Current state / where to pick up
- Compare with a random Kaiming init under the same annealed schedule (same seed, same
  batches) to see whether the design speeds the final coverage of level 4 or only the
  early capture.
- Track node coverage over epochs (rather than probe on-target) as the main hierarchy
  metric; the per-probe view undercounts what the net has learned.

## 2026-09-30 04:05 IST — Checkpoint visualisation for the cosine-annealed runs

### Goal
Tools to inspect the stored trajectories (all four weight matrices every 10 epochs): draw the
learned network at a chosen epoch like `plot_layered_relu_graph`, and track cosines of chosen
first-layer neurons to chosen ODT nodes over the run as in `scratch.ipynb`.

### User prompts (verbatim)
- "Do you have the full parameter trajectory of these runs? e.g. in the folder \"results/designed_init/hierarchy_kaiming_cosine/\" I don't see any pickles or large files at all. Are they elsewhere?"
- "Got it. I am looking to do some more analysis with these checkpoints. Can you add some plotting at the end of desgned_init notebook? \n\nI want to be able to visualise the learned network at a given epoch, similar to the \"plot_layered_relu_graph\" function. From the W0,W1,W2,W3 arrays (and also the tree.w_list for tracking components with the ODT vector. Can you add it?\n\nAlso, a similar functionality as in scratch.ipynb (in the section titled \"Neuron-wise First layer neuron alignment analysis\") to track the cosines of a given bunch of first layer neurons w.r.t. a bunch of ODT nodes."

### Changes
- `relu_analysis.py`: `state_dict_from_weight_arrays(hidden_weights, output_weight)`,
  `snapshot_state_dict(run, epoch)` (from a `w__<param>`-keyed npz trajectory),
  `first_layer_alignment_series(W0_stack, epochs, tree, neuron_ids=, node_ids=, fan_out_weights=)`
  (long table: norm, cos_<node>, fan-out column norm), `plot_first_layer_alignment_series`
  (one panel per neuron, cosines on the left axis, norms on a right axis). Two tests.
- Notebook section 13: graph of the network at `GRAPH_EPOCH` (first-layer nodes labelled by
  closest ODT node, optional mask to aligned neurons only), a table of which node each
  neuron sits on by group, and the alignment-series plot with editable neuron/node lists.
  The user's own cells (loading W0-W3 from the npz) sit just before section 13.

### Decisions
- Trajectories are stored every 10 epochs (10k steps) in `runs/*.npz`, ~5 MB each, no
  optimizer state; finer resolution or momentum buffers need a rerun.
- For a first-layer neuron in a 3-hidden-layer net the scratch notebook's "output weight"
  has no analogue, so the series reports the norm of its outgoing second-layer column.

### Current state / where to pick up
- In the free-cue run, probes 1 (-u_0) and 2 (+u_1) wander for 150-250 epochs before
  locking onto nodes 9 and 13 with cosine -1, while probes 0 and 3 lock within 20 epochs.
  The late captures are candidates for the linearized-flow / lens analysis.

## 2026-09-30 05:20 IST — 10M points, batch 10k, no random neurons, one unit per probe (cues frozen)

### Goal
Test whether the probe deviations from their designed nodes in the cosine-annealed Kaiming run
are a finite-sample effect: 10x data and batch, and remove the init randomness (no random
first-layer rows, one second-layer unit per probe reading every candidate cue, exactly half
the third-layer entries nonzero). Cues frozen only.

### User prompts (verbatim)
- "Ok, let's do one thing to see if some of this probe deviation from intended vectors is a finite sample effect. Can you increase the number of data points by a factor of 10, and batch-size also by 10? I remember my CUDA device still had sufficient memory to hold the entire 1M size dataset in memory, so 10x batchsize and data should be possible, without too much extra time. \n\nAlso, to keep the leftover random vectors (first layer neurons apart from cues and probes) from affecting the results, let's just remove them and have a first layer width of just 62. \n\nTo further reduce the effect of randomness, let's just have 30 neurons in the second hidden layer -- one per probe. That way we can just connect all the relevant cues to the second layer instead of choosing a subset.\n\nOne more modification to have the same effect, the third layer 40-by-30 matrix, let's initialize so that the number of non-zero entries in each row is exactly half (i.e.) 15.\n\nWhat do you think of this new experiment, any changes you would suggest?"
- "Can you give the current run's train config details( for the runs stored in \"results\\designed_init\\hierarchy_kaiming_cosine\\runs\\linear_free_kaiming_cosine_seed0.npz\")"
- "20  minutes is not a big deal. I would like to get cleaner numbers before analysing. \n\nRun the new experiments just on larger data and batch size, with your proposed fixes (data generator mainly) and only the frozen cues setting."

### Changes
- `init_sweep.build_data_chunked(spec, tree, chunk_size=1M)`: generates in chunks (seed
  `data_seed + k`) and stores float32 inputs / int8 labels. The one-call generator works in
  float64 and would need ~18 GB host memory for 10M points; chunked it took 18 s and 4 GB.
- `HierarchyDesignSpec.layer3_exact`: exactly round(density * width2) ones per third-layer row.
- Script `results/designed_init/run_hierarchy_kaiming_cosine_big.py`: design
  `width1=62, width2=30, width3=40, units_per_level=(1,1,1,1), subset_mode="all", layer3_exact=True`,
  Kaiming rows, linear cues, seed 0, cues frozen; 10M points, batch 10k, otherwise the section-12
  schedule (1000 epochs x 1000 steps, lr 0.02 -> 0.001 cosine, momentum 0.9). Notebook section 14.
  Two tests. First launch crashed on a `.numpy()` of a cuda tensor in a diagnostic print; fixed.
- Timing: a batch-10k step costs the same as batch-1k (0.87 ms), so the run took 24 min; the 10M
  dataset occupies 3.7 GB of GPU memory.

### Results (vs. the 1M / batch-1k frozen run of section 12)

| | 1M / 1k, 80-40-40 | 10M / 10k, 62-30-40 |
|---|---|---|
| final epoch loss | 0.010 | 0.117 |
| test error | 0.67% | 4.4% |
| epochs to 0.3 / 0.2 / 0.1 | 37 / 51 / 135 | 55 / 101 / never |
| probes on designed target at end (L0-3) | 0.5 / 0.25 / 0.62 / 0.44 | 0.5 / 0.5 / 0.75 / 0.5 |
| nodes covered at |cos| > 0.95, end | 31 / 31 | 29 / 31 |

- Not a finite-sample effect: with 10x data and 10x batch the probes deviate from their
  designed nodes just as much (17 of 30 on target). Which root probe wins also changed: in
  the 1M run probe 0 reached +u_0 and probe 1 (-u_0) went to node 9; here probe 1 reached
  its own target -u_0 (cos -1.00) and probe 0 (+u_0) went to +u_14. No probe ended on the
  opposite pole of its node; only the winner of the pair changed.
- The loss floor is coverage. Level-3 nodes 8 and 9 are never captured by any of the 62
  neurons (from epoch 100 to the end), while nodes 0-4 each end with two or three neurons
  on them (e.g. probes 2, 21 and 8 all on u_1; 9, 18, 19 all on u_4). Seven neurons end
  unaligned (|cos| < 0.95), mostly the ones that lost the competition for a coarse node.
  Without the 18 spare random neurons there is nobody left to pick up the missing nodes.
- The probe's own weight in its unit falls with the number of cues the unit reads under row
  normalisation (0.25 at the root, 0.34 / 0.47 / 0.63 at levels 1-3), an asymmetry that was
  half as strong with random cue subsets.

### Reading
The design decides *that* the hierarchy is learned, not which neuron takes which node; the
assignment is settled by competition during the escape, and duplicated coarse nodes are the
price. Spare neurons are what made the 80-40-40 run reach 0.01: they fill in whatever the
probes leave. Same conclusion as the init sweep (winner init-determined but decided
mid-escape) and the earlier per-probe "drift".

### Current state / where to pick up
- Cleanest next check of the competition story: same 62-30-40 design at 1M / 1k (isolates
  data size from architecture, 25 min), and a 62-30-40 run with 18 random rows added back.
- The gradient-cosine check (full-batch vs minibatch gradient on a wandering probe's row at
  the stored snapshots) was not run; it is the direct test of sampling noise on one run.
- Possible fix for duplicates: freeze a probe's row once it has locked (|cos| > 0.95), or
  give each unit the probe entry the same norm as its cue block.

## 2026-09-30 05:45 IST — Network graph cell for the 10M run

### User prompts (verbatim)
- "Can you add a cell in Section 14 similar to Section 13 for visualising the neural net as a whole in a given epoch?"

### Changes
- Notebook section 14: a graph cell (`plot_layered_relu_graph` on `snapshot_state_dict(zb, GRAPH_EPOCH_B)`)
  with the same editable parameters as section 13, plus the table of which node each
  neuron sits on. No code changes.

## 2026-09-30 06:00 IST — Correction to the root-probe wording

### User prompts (verbatim)
- "The probe targets are +u_0,-u_0,+u_1,...+u_14,-u_14 right? I don't see the flip in root probes you mentioned. At the end of training the probe supposed to go to +u+0 did go to +u_14 (with cos 0.76), but the probe targeting -u_0 did go to -u_0. Am I missing something in the figure here?"

### Decisions
- The 05:20 entry said the root pair "flipped"; that was wrong wording. In the 10M run the
  -u_0 probe reached its own target and the +u_0 probe went to +u_14; in the 1M run the
  +u_0 probe reached its target and the -u_0 probe went elsewhere. Which probe of the pair
  wins changed between runs; no probe reached the opposite pole. Entry above corrected.

## 2026-09-30 06:20 IST — Per-node reading of the probe outcomes (Q&A, verified from checkpoints)

### User prompts (verbatim)
- "Ok, I think we should call this a success in terms of init design. All of the 30 probes didn't go to target, but the probes for most of the 15 internal nodes at least one of +u_j or -u_j was reached. Maybe it is the case that there is no need for both +u_j and -u_j to be learned for loss minimisation.\n\nThe larger data run with architectures 62-30-40  is more well behaved in that sense. all u_j except j=8 and 9 were captured at the designed probe in atleast one polarity.\n\nThe smaller data run with 80-40-40 architecture failed to capture the nodes 1,4,13, and 3,6,7,9,10,12,14 are captured at opposite polarity (i.e probe intended tmake them go to +u_j but gradient descent took it to opposite pole). All ODT nodes were captured at some first layer neuron, but intended capture by probes was not happening.\n\nOf course, I am guessing the missing extra freedom of the 18 blnak neurons coast the 62-30-40 run in terms of accuracy.\n\nAm I reading the results right?"

### Verification (end of training, |cos| > 0.95, per internal node j with probes 2j -> +u_j, 2j+1 -> -u_j)
| run | nodes with >= 1 own probe | at intended pole | opposite pole only | missed by own probes |
|---|---|---|---|---|
| 10M/10k frozen 62-30-40 | 13/15 | 13 | 0 | 8, 9 |
| 1M/1k frozen 80-40-40 | 12/15 | 5 | 7 (3, 6, 7, 9, 10, 12, 14) | 1, 4, 13 |
| 1M/1k free 80-40-40 | 9/15 | 9 | 0 | 3, 9, 11, 12, 13, 14 |
Both poles captured by own probes: 10M run nodes 5, 6, 7, 11; frozen 1M run nodes 3 (swapped), 5.
The user's reading of the two frozen runs is exactly right. Opposite-pole captures occur only
in the random-subset design (1M frozen); with `subset_mode="all"` the ancestor pushes cancel
exactly at t=0 inside each unit, a candidate explanation to test.

### Decisions
- "One pole per node suffices for the loss" is not established: the 10M net (mostly single-pole,
  nodes 8/9 missing) sits at 4.4% test error, and the two effects cannot be separated from this
  run. Isolating runs (62-30-40 at 1M/1k; 62-30-40 plus 18 random rows) remain the next step.

## 2026-09-30 07:30 IST — Isolating data size and spare neurons; the dead-second-layer solution

### Goal
Separate the two changes of the 10M run: the 62-30-40 design at 1M / batch 1k, and the same
design with the 18 random rows restored (80-30-40) at 10M / batch 10k. Cues frozen.

### User prompts (verbatim)
- "Go ahead and start those runs."

### Changes
- Script `results/designed_init/run_hierarchy_kaiming_cosine_isolate.py` (both runs, same
  schedule as sections 12-14), outputs under `notebooks/results/designed_init/hierarchy_kaiming_cosine_isolate/`.
- `evaluate_relu_mlp` now also returns `zero_one_loss_strict` (logit > 0 means +1) and
  `zero_logit_fraction`; `zero_one_loss` (logit >= 0) is unchanged. One test.
- Notebook section 15: four-run comparison (loss, node coverage, per-node probe outcomes) and
  a zero-logit diagnosis cell.

### Results (end of training, test set of 200k; per-node counts use |cos| > 0.95)

| run | final loss | test err (>= 0 rule) | test err (strict) | zero-logit frac | neg. output weights | nodes covered | own-probe nodes / intended / opposite | missed by own probes |
|---|---|---|---|---|---|---|---|---|
| 1M/1k 80-40-40 (sec. 12) | 0.010 | 0.7% | 0.7% | 0.00 | 21/40 | 31 | 12 / 5 / 7 | 1, 4, 13 |
| 10M/10k 62-30-40 (sec. 14) | 0.117 | 4.4% | 4.4% | 0.00 | 10/40 | 29 | 13 / 13 / 0 | 8, 9 |
| 1M/1k 62-30-40 | 0.436 | 50.0% | 8.9% | 0.51 | 0/40 | 29 | 11 / 11 / 0 | 3, 8, 10, 14 |
| 10M/10k 80-30-40 | 0.371 | 50.0% | 2.1% | 0.49 | 0/40 | 31 | 13 / 13 / 0 | 0, 4 |

- The two new runs found a different solution: on about half of all inputs (the y = -1 class)
  every second-layer unit is off and the logit is exactly zero. The design's output layer is
  all ones and its third layer nonnegative, so every logit is positive at init; fitting
  y = -1 needs either negative output weights or a dead second layer. In these two runs no
  output weight ever went negative, so the net took the dead route, paying log 2 on every
  negative point (hence the 0.37-0.44 loss floor). `zero_one_loss` scores a zero logit as +1,
  which is why the summary shows chance-level test error; the strict rule gives 8.9% and 2.1%.
- The 10M 62-30-40 run of section 14 started down the same road (21% zero logits at epoch 10)
  and escaped by epoch 30 when 8 output weights turned negative; the section-12 run had 21
  negative output weights by epoch 10 and never had zero logits. The same init (62-30-40,
  seed 0) went one way at 10M/10k and the other at 1M/1k: the branch is decided early and is
  noise-sensitive.
- First-layer capture is largely unaffected by which branch is taken: 11-13 of 15 nodes are
  captured by their own probes, always at the intended pole in the all-cues design (0
  opposite-pole captures in all three all-cues runs vs 7 in the random-subset run). Restoring
  the 18 random rows brings node coverage back to 31/31.

### Reading
- The accuracy comparisons across these four runs are confounded by the output-sign branch,
  not only by spare neurons. Spare neurons do restore full coverage (31/31 with 80 rows, 29/31
  without, in both data regimes).
- The init design should let the output represent both signs from the start: random-sign or
  balanced output weights (the earlier note that a balanced-sign output "kills the probe" was
  at eps = 1e-3 under the flow and needs rechecking at Kaiming scale), or a bias.

### Current state / where to pick up
- Rerun the 62-30-40 and 80-30-40 designs with a balanced-sign output layer (and matching
  sign structure in layer 3) so the dead-layer branch is unavailable; then redo the
  spare-neuron and data-size comparison.

## 2026-09-30 09:40 IST — Balanced-sign output runs; notebook reorganised into one section per run

### Goal
Rerun the two all-cues designs (62-30-40 and 80-30-40) with a balanced-sign output layer so
the dead-second-layer branch is unavailable, and reorganise the designed-init notebook.

### User prompts (verbatim)
- "Run the balanced-sign output version for both designs. \n\nOne other change request for the notebook. Reorganise so that there is one section for each run in the notebook, with one cell for trajectory plots giving cosines or components of first layer neurons, and one cell for network visualization. Give a brief summary of the run config at the beginning of the section. And a brief summary of the finidings.\n\nRemove the earlier sections probing for smaller sets of vectors. Rename the file appropriately."

### Changes
- `HierarchyDesignSpec.output_signs` ("ones" | "balanced": alternating +1/-1 over the third-layer
  units). `probe_node_outcomes`, `hierarchy_run_summary`, `plot_probe_target_cosines` in
  `designed_init.py`. Tests for all.
- Script `results/designed_init/run_hierarchy_kaiming_cosine_balanced.py` (both designs, 10M /
  batch 10k, cues frozen, otherwise the section-14 setup); outputs under
  `notebooks/results/designed_init/hierarchy_kaiming_cosine_balanced/`.
- `notebooks/designed_init_u0.ipynb` renamed to `notebooks/hierarchy_design_runs.ipynb` and
  rebuilt: intro (design, schedule, metrics), then one section per cosine-annealed run (7 runs)
  with a config table, a findings summary, a trajectory cell (loss / zero-logit fraction / node
  coverage over epochs, every probe's cosine to its target by level, alignment series for chosen
  rows) and a network-graph cell; the balanced-output sections carry a fourth cell testing the
  path-sign rule. The single-, two- and six-probe sections and the SGD sweeps remain in git
  history (`designed_init_u0.ipynb` through commit 3f06b4a); their summaries stay under
  `notebooks/results/designed_init/`.

### Results (balanced output, 10M / 10k, cues frozen; |cos| > 0.95)

| run | final loss | test error | zero-logit frac | nodes covered | own-probe nodes / intended / opposite only | missed by own probes |
|---|---|---|---|---|---|---|
| 62-30-40 balanced | 0.081 | 3.1% | 0.00 | 30 (14 missing) | 14 / 6 / 8 | 14 |
| 80-30-40 balanced | 0.033 | 1.0% | 0.00 | 31 | 12 / 6 / 6 | 0, 1, 6 |
| (62-30-40 all-ones, run 3) | 0.117 | 4.4% | 0.00 | 29 | 13 / 13 / 0 | 8, 9 |

- The balanced output removes the dead-layer branch (zero logits 0 at every epoch, 20-21
  negative output weights throughout) and lowers the loss of both designs. With the output
  fixed, the spare-row effect is clean: 0.033 (80 rows) vs 0.081 (62 rows), coverage 31 vs 30.
- The logit RMS at init drops from 2.1 to 0.10 by cancellation; the escape starts later
  (loss 0.67 at epoch 10) but 0.3 is reached at the same epoch (31-36).
- Opposite-pole captures return (8 and 6 nodes), and they are predicted by the sign of the
  probe's unit path to the logit at init, sum_k Wout[k] W3[k, p]: of the 29 captured probes
  with a nonzero path, 25 end at the intended pole when the path is positive and at the
  opposite pole when it is negative (14/15 in the 62 run, 11/14 in the 80 run). The push-sign
  rule of the design (-s * side) implicitly assumed a positive path; with mixed output signs
  it becomes -s * side * sign(path). Several pairs therefore land on the same pole (both root
  probes on -u_0 in the 62 run).

### Decisions
- The all-ones output is retired for future designs; use balanced signs and fold the path
  sign into the cue-sign rule so that both probes of a node are pushed to distinct poles.

### Current state / where to pick up
- Implement the path-sign-aware cue rule (choose each unit's cue signs from the sign of its
  init path to the logit) and rerun the 80-30-40 balanced design; expect intended-pole capture
  to return to ~13/15 with both poles distinct.
- Node coverage still depends on spare rows; consider 2-3 random rows per level-3 node.

## 2026-09-30 10:10 IST — Handoff commit and push

### User prompts (verbatim)
- "I want to Commit everything and push to remote. This is a good handoff point for discussing with my students and collaborators. \n\nHow big would the pushed repo be? Give instructions/scripts on generating the files (parameter trajectories etc) necessary to run the \"hierarchy_design_runs\" notebook."

### Changes
- `scripts/regenerate_hierarchy_design_runs.sh`: runs the four hierarchy scripts (copies under
  `notebooks/results/designed_init/*/`) from the repo root to rebuild everything
  `notebooks/hierarchy_design_runs.ipynb` reads from the git-ignored `results/designed_init/`
  (7 runs, ~3 h on one GPU, needs >= 8 GB GPU memory for the 10M-point runs). README section.
- Committed the user's working files (`.vscode/settings.json`, `notebooks/scratch.ipynb`,
  `notebooks/ReLU_replayable_training.ipynb`) as they were; pushed `main` to origin.

### Decisions
- Raw trajectories (30 MB for the seven runs) stay out of git; collaborators regenerate them.
  The executed notebook (17 MB with figures) is committed so the results can be read without a GPU.

## 2026-10-01 — Trajectories committed for the hierarchy-runs notebook

### User prompts (verbatim)
- "Commit the trajectories required for running the notebook too for now. My students don't have access to a GPU system yet."

### Changes
- `.gitignore` now re-includes the four run folders under `results/designed_init/`
  (`hierarchy_kaiming_cosine`, `_big`, `_isolate`, `_balanced`: `runs/*.npz`, `summary.csv`,
  `per_epoch_*.csv`, `tree.npz`, `manifest.json`, ~30 MB), the only tracked content under
  `results/`. README adjusted: the notebook runs on a fresh clone without a GPU; the
  regeneration script remains the way to rebuild from scratch.

## 2026-10-01 — Path-sign-aware cue rule; 80-30-40 balanced rerun

### Goal
Make the design's push-sign rule account for the sign of each unit's initial path to the
logit, and rerun the 80-30-40 balanced-output design.

### User prompts (verbatim)
- "Implement the path-sign-aware cue rule and rerun the 80-30-40 balanced design"

### Changes
- `HierarchyDesignSpec.path_sign_cues`: after layers 3-4 are built, units whose init path
  `sum_k Wout[k] W3[k, unit]` is negative get their cue signs flipped (the probe entry stays
  -1), i.e. they are wired for the opposite pole so the first-order push points to the
  designed pole. `HierarchyDesignSpec.layer3_nonzero_paths`: with `layer3_exact`, repairs the
  third layer by row-count-preserving 2x2 swaps (`_repair_zero_unit_paths`) until no unit has
  a zero path. `meta` now carries `unit_path_sign`, `flipped_units`, `zero_path_units`. One
  test checks the cue signs, the repair, and the first-order push at init via autograd.
- Script `results/designed_init/run_hierarchy_kaiming_cosine_pathsign.py` (80-30-40, 10M /
  10k, frozen cues, balanced output, both options on). Outputs tracked under
  `results/designed_init/hierarchy_kaiming_cosine_pathsign/` (gitignore extended) and copies
  under `notebooks/results/designed_init/hierarchy_kaiming_cosine_pathsign/`. Notebook
  section 8 (with a cell showing the init push and the end pole of every probe).

### Results
- At init (200k-point population gradient), probes pushed toward their target: 13/30 with
  run 7's init, 28/30 with the flips alone (6 units still had a zero path), 30/30 with the
  repair (mean cosine of -grad with the target +0.36). 13 of 30 units were flipped.
- Trained: loss 0.030, test error 0.95% (run 7: 0.033, 1.0%); 0.3 / 0.1 / 0.05 at epochs
  28 / 110 / 456; no zero logits. All 31 nodes covered from epoch 50; 67/80 rows aligned.
- Pole outcomes: 18 of 30 probes end at their intended pole and 0 at the opposite pole
  (run 7: 6 intended / 6 opposite nodes). 12 of 15 internal nodes captured by own probes,
  both poles for 3, 4, 7, 9, 11, 12. Missed by own probes: the root and both level-1 nodes
  (0, 1, 2), taken by cues / random rows; the same three were among the misses in run 7.

### Reading
- The sign rule is now complete: push sign = -s * side * sign(path). With the path made
  positive-effective for every unit, opposite-pole capture disappears.
- The remaining failure is the coarse end. Root/level-1 units read the most cues, so under
  row normalisation their probe entry is the smallest (0.25 at the root vs 0.63 at level 3),
  and those probes lose the race to rows that start partly aligned.

### Current state / where to pick up
- Equalise the probe's effective push across levels (e.g. set the probe entry to the norm
  of the unit's cue block, or scale the unit's cue block by 1/sqrt(n_cues)) and rerun; if the
  coarse probes then capture, the design decides the whole assignment.
- Not pushed to origin; commit only.

## 2026-10-01 — Equalised cue blocks (probe push equal across levels)

### User prompts (verbatim)
- "Error when running the last cell in Section 8 :\n\nHierarchyDesignSpec.__init__() got an unexpected keyword argument 'path_sign_cues'"
- "Equalise the probe's effective push across levels, for instance by scaling each unit's cue block by one over the square root of its cue count."

### Changes
- The section-8 error was a stale kernel: `%autoreload` cannot replace a dataclass's generated
  `__init__`, so a kernel started before a field is added keeps the old constructor. Restart.
- `HierarchyDesignSpec.cue_block_scale` ("none" | "inv_sqrt"): each unit's cue block is divided
  by sqrt(number of nonzero cue entries), so after the row rescale every unit has probe entry
  1.0 and cue-block norm 1.0 (before: 0.25 at the root, 0.63 at level 3). Signs, including the
  path-sign flips, unchanged. One test.
- Script `results/designed_init/run_hierarchy_kaiming_cosine_cueeq.py` (run 8 plus the scaling),
  trajectories tracked under `results/designed_init/hierarchy_kaiming_cosine_cueeq/`, copies
  under `notebooks/results/designed_init/hierarchy_kaiming_cosine_cueeq/`. Notebook section 9.

### Results (80-30-40, 10M / 10k, frozen cues, balanced output, path-sign cues, inv_sqrt)
- Init push component along the target by level (x1e-4): 1.08 / 0.80 / 0.97 / 0.85, vs
  0.47 / 0.45 / 0.73 / 0.82 for run 8; still 30/30 probes pushed toward target.
- Loss 0.017, test error 0.52% (run 8: 0.030, 0.95%; best error of any designed run);
  0.3 / 0.1 / 0.05 at epochs 23 / 86 / 255; no zero logits.
- Both root probes lock onto +-u_0 by epoch 20. 21/30 probes at the intended pole, 0 at the
  opposite; 13/15 internal nodes by own probes, both poles for 8 nodes. Missed: node 1 (both
  probes go to level-4 nodes; node 1 covered by two other rows) and node 10, whose probes
  reached cosine 1.00 by epoch 100 and slid back to ~0.75 by epoch 300; node 10 is the only
  node uncovered at the end (30/31) while the loss keeps falling.

### Reading
- With the three fixes (balanced output, path-sign-aware cue signs, equalised cue blocks) the
  design now decides most of the assignment: the root pair, 8 full pairs, 0 sign errors.
  What is left looks like late-phase reassignment (node 10) and one level-1 node.

### Current state / where to pick up
- Look at the node-10 slide-back (epochs 100-300): which rows take over its role, and whether
  freezing captured probe rows would hold it.
- Node 1: check its two units' init push (both nonzero) and the rows that end on u_1.
- Not pushed; commit only.

## 2026-10-01 — Why probes 2 and 3 (+-u_1) fail in run 9: population-gradient analysis

### User prompts (verbatim)
- "Ok, before we do that, I want to analyse the neuron 2 and neuron 3's failure to move along +u_1 and -u_1 respectively.  Isn't the init designed so that at epoch 0, the gradient for the probe neuron would point towards the target ODT vector? I don't see that happening at even epoch 10. \n\nCan you also get the statistics on the initial gradient properties? Maybe also a wrapper function to get the parameter gradients w.r.t. the entire train loss (population gradient) at a given checkpoint epoch? We have such a function in ReLU_analysis already, I think."

### Changes
- `relu_analysis.population_gradient_at_snapshot(run, epoch, x, y, device=, batch_size=)`: rebuilds
  the model from a `w__`-keyed trajectory at a stored epoch and accumulates the mean-loss gradient
  over all given points in batches (wraps `log_loss_gradients`). 2.6 s for 2M points on the GPU.
- `designed_init.probe_gradient_stats(grad_W1, W1, tree)`: per probe, cosine of the push (-grad)
  with the designed target and with the current row, gradient norm, energy fractions of the push
  along the target / its level-4 descendants / other level-4 nodes / other internal nodes /
  off-span, closest signed node, and the row's own cosine to the target. Tests for both.
- Notebook section 9: a gradient-analysis cell (init statistics for all 30 probes, probes 0-5
  over epochs 0-100, each unit's path to the logit / probe entry / cue-sign retention, plots).

### Findings (run 9, gradient over the first 2M training points)
- At epoch 0 the push does point at the target for every probe: cosine with the target 0.68
  (root), 0.55 (level 1), 0.42 (level 2), 0.32 (level 3). The energy of the push along the target
  is only 46% / 31% / 18% / 10% by level; the rest lies along level-4 directions (for level 1:
  32% along the node's own level-4 descendants, 36% along other level-4 nodes), i.e. the
  constant part of the lens. Off-span energy is ~1%.
- Why probes 2 and 3 fail anyway: their units (2, 3) had init paths to the logit of -0.08, the
  smallest possible magnitude (one quantum: one third-layer entry times one output weight),
  so the cue-sign rule flipped their cues. Within 10 epochs layers 3-4 moved and the paths
  crossed zero (+0.02 and +1.09 at epoch 10). With the path positive the flipped wiring pushes
  toward the opposite pole: probe 3's push at epoch 10 has cosine -0.89 with -u_1 (79% of its
  energy along +u_1); unit 2's path sat near zero, so probe 2 got almost no push (gradient norm
  1.3e-4 at epoch 10 vs 15-28e-4 for probes 0, 1, 4). Units 3 and 5 then drove their probe
  entry to ~0 (by epochs 30 and 50), disconnecting the probe; the rows ended on level-4 nodes.
- Across all 30 units: 4 paths changed sign during the run (units 2, 3, 5, 12, all with the
  minimum init magnitude 0.08 and all negative, i.e. rule-flipped) and 0 of those 4 probes
  captured; of the 26 whose sign held, 21 captured. Probes whose units kept a larger initial
  path (0: +0.16, 4: -0.24) had 2-3x larger gradients and reached cosine 0.94 / 0.82 by
  epoch 10, before the downstream changed.
- The designed second-layer wiring is short-lived: by epoch 10 only ~26% of the cue entries of
  units 2-5 (52% for the root units) still carry their init sign, and the trainable cue block
  norm grows. The probe has to capture inside that window.

### Reading
The design is right at t = 0 but relies on two things that layers 2-4 overturn within ~10
epochs: the sign of each unit's path to the logit (one quantum for a third of the units) and
the cue signs in layer 2. Remedies to test: build W3 so every unit's path has magnitude of at
least a few quanta (|n+ - n-| >= 3) with the designed sign, and/or freeze layers 2-4 for the
first ~20 epochs (the freeze-then-release protocol that worked for the cues).

### Current state / where to pick up
- Implement a minimum-path-magnitude option for the third layer and rerun run 9; alternatively
  a freeze of layers 2-4 until epoch 20.
- Not pushed; commit only.

## 2026-10-01 — Balanced third-layer paths (every unit +-10 quanta); run 10

### User prompts (verbatim)
- "Terminology clarification. Does \"unit\" refer to a neuron ion the second hidden layer? And by path to logit of a unit, do you mean the sum of products of edge weights between that unit and the logit? i.e. sum of 40 terms for each unit, with each of these terms being a product of two edge weights along a path from that unit to the logit?"
- "Ok, consider a balanced init for the third layer weights then. Every unit is connected to exactly 20 neurons in the third hidden layer. Let's make it so that exactly 15 of them are connected to an third hidden layer neuron with positive W_out and 5 of them are connected to a neuron with negative W_out. Alternate with vice versa units : i.e. connected to 15 negative W_out and 5 positive W_out. \n\nIs something like the init scheme above what you had in mind?"
- "Implement it and rerun run 9's config"

### Changes
- `HierarchyDesignSpec.layer3_mode` ("random" | "balanced_paths") and `layer3_majority` (15):
  with balanced output signs, unit 2j connects to `majority` positive-output and
  `width3/2 - majority` negative-output third-layer neurons, unit 2j+1 the reverse, the two
  units of a pair using complementary subsets of each output-sign group, so every row has
  width2/2 = 15 ones and every unit's path to the logit is +-(2*majority - width3/2) = +-10
  quanta with alternating sign. The path-sign rule then flips exactly the odd units. Test.
- Script `results/designed_init/run_hierarchy_kaiming_cosine_balpaths.py` (run 9's config with
  the balanced third layer); trajectories tracked under
  `results/designed_init/hierarchy_kaiming_cosine_balpaths/`. Notebook section 10 (config,
  findings, trajectory and graph cells, and a path-sign stability cell).
- Terminology settled: "unit" = second-hidden-layer neuron; "path to the logit" = sum over the
  40 third-layer neurons of W3[k, unit] * Wout[k] (the ungated two-edge path sum).
- A first launch ran with an unmodified script for a few seconds (a sed pattern missed the
  line); killed, fixed, relaunched from a clean folder.

### Results (80-30-40, 10M / 10k, frozen cues, balanced output, path-sign cues, inv_sqrt, balanced paths)
- Init: |path| = 0.82 for every unit (run 9: 0.08-0.65), probe gradient norms 8-11e-4 (run 9
  median 1.8e-4), 30/30 pushed toward target, logit RMS 0.12.
- From epoch 20 on: all 15 internal nodes captured by their own probes at the intended pole,
  10 at both poles, all 31 nodes covered. 25/30 probes end at the intended pole, 0 opposite.
  Loss 0.3 / 0.1 / 0.05 at epochs 17 / 53 / 460; final loss 0.029, test error 0.90%
  (run 9: 23 / 86 / 255, 0.017, 0.52%). Test error at epoch 50: 3.4% (run 9: 4.4%).
- Only unit 2's path changed sign (run 9: four); its probe (+u_1) was pulled to -u_0 again,
  but node 1 is held by probe 3. Five units dropped their probe entry (6, 13, 14, 27, 29).
- The final loss sits on a plateau near 0.07 from epoch 100 to 300 before the annealed tail.

### Reading
With balanced output, path-sign-aware cues, equalised cue blocks and +-10-quanta paths, the
design decides the whole first-layer assignment: every internal node on its own probe at the
designed pole by epoch 20. The remaining gap to run 9's final loss is not about coverage.

### Current state / where to pick up
- Look at the epoch-100-300 plateau of run 10 (which parts of the net are still moving) and
  at probes 2 and 5 being pulled to the root pole: a root-direction leak in the level-1 units'
  lens after the root pair is captured.
- Not pushed; commit only.

## 2026-10-02 — Population-gradient component plots vs epoch

### User prompts (verbatim)
- "Fix the error for the following cell:" (cell calling `population_gradient_at_snapshot(run10, ...)`; KeyError 'w__output_layer.weight')
- "Can you add a cell for visualizing population gradients in the same manner as the plots for the parameter trajectories? It should take neurons and ODT nodes as arguments and plot the components of the population gradient w.r.t. epochs. Also add another line for the norm of the projection of the gradient in the orthogonal complement of all the ODT vectors."
- "I meant vs epochs and not w.r.t epochs"

### Changes
- `snapshot_state_dict` (and so `population_gradient_at_snapshot`) accepts the notebooks'
  `load_hier_run` dicts (`epochs` + `W` sub-dict) as well as `w__`-keyed mappings.
- `relu_analysis.first_layer_gradient_series(grads_by_epoch, tree, neuron_ids=, node_ids=,
  push=True, normalize=False)`: per (epoch, neuron) the signed components of the push (-grad,
  or the raw gradient) along the unit ODT normals, the in-span norm, the norm of the projection
  onto the orthogonal complement of all ODT normals, and the total norm.
  `plot_first_layer_gradient_series`: one panel per neuron vs epoch. Tests.
- Notebook: a parametrised cell at the end (RUN, GRAD_EPOCHS, NEURON_IDS, NODE_IDS, PUSH,
  NORMALIZE) that caches population gradients per run tag. Left unexecuted to avoid racing the
  user's live editor; a dry run on run 10 checked it.

## 2026-10-02 — Why the snapshot population gradient on a captured probe is not "c * u_0"

### User prompts (verbatim)
- "I am somewhat confused by how these gradients can lead to these trajectories -- e.g. probe 0, the gradient doesn't seem to settle at a constant times u_0."

### Analysis (run 10, probe 0, epochs 100-130; gradients over 1-2M training points)
- Mismatch: the snapshot push on probe 0 is 0.027 in norm, perpendicular to the row (cosine
  0.01 with it), persistent between snapshots (cosine 0.96 at 10-epoch spacing), and would
  rotate the row by ~35 degrees per epoch at lr 0.0195 / momentum 0.9. The row rotates
  0.1-0.2 degrees per ten epochs and moves radially (+0.035 per epoch, cosine 0.998 with u_0).
- Edge of stability: curvature along the full gradient gives lr * lambda = 3.9 at epoch 100
  (2.7 at 300) against the heavy-ball limit 2(1+beta) = 3.8. The full gradient (norm 3.2)
  lives in layers 2-3 (2.5 and 2.0); row 0 carries 8e-5 of its squared norm. One step along
  -g drops the loss 0.103 -> 0.073 and the gradient norm to 0.30; the opposite flank has the
  same loss, i.e. a period-2 oscillation across a sharp valley in the downstream weights.
  On the two flanks probe 0's push along u_0 is -0.009 and +0.020: the radial part averages
  to the positive drift that is seen; the off-span part (0.014) is the same at both flanks.
- Step-level replay (400 SGD-momentum steps from the snapshot, batch 10k, population gradient
  on row 0 at every step): per-step |push| 0.02-0.03 (up to 0.44 during the momentum
  transient), nearly uncorrelated from step to step (off-span direction loses its correlation
  with step 0 within ~10 steps; running mean of the push falls to 0.0033 at 400 steps, ~1/10
  of the per-step size and still falling); row rotates 0.5 degrees in 400 steps.
- So the instantaneous population gradient on a captured row is dominated by a fluctuating
  component set by the phase of the fast downstream coordinates (the lens swings with the
  oscillating layers 2-4), ~100x larger than the time-averaged push the row actually
  integrates (0.0348 per epoch / (0.195 * 1000) = 1.8e-4 per step, radial). The
  "constant times u_0" picture holds for the time average; snapshots every 10k steps sample
  the oscillation at a fixed phase, which is why they look persistent.

### Decisions
- From ~epoch 20 on (lr * lambda near the limit) the snapshot gradient plots measure the
  oscillation, not the drift; the gradient plots are meaningful early (epochs 0-10) and
  for the time-averaged push. A `mean push from the trajectory` series
  (delta w between snapshots / (sum of lr/(1-beta) * steps)) would be the right companion
  to the gradient cell; not implemented yet.

## 2026-10-03 — Run 11: replayable version of run 10 (1M points, lr 0.01, momentum snapshots)

### User prompts (verbatim)
- "Can you replay the training for any given epoch with just these checkpoints? Or do you need additional details?  \n\nI am more or less happy with run 10 initialisation as it behaves as intended, but I would like one more run with the following simplifications:\n\n1. Lesser learning rate (the loss fluctuations from epoch 100 to 300 are very wild).\n2. quick full replay between any pair of adjacent checkpoints. Similar to the replayable training notebook with population gradients per step.\n3. To further point 2, I am also looking to scale back to 1M training points as I can keep the entire data in memory without issue, but hopefully the trajectory stays the same. \n\n\nWe can do one more run with these mods, and move on to the next phase where we do detailed trajectory analysis of this run in a separate notebook."
- "hi there is a power failure, and I will have to shut down the system. Can you safely terminate the process now?"
- "System is back up, relaunch the run."

### Answer to the replay question
Not from the earlier checkpoints: exact replay needs the momentum buffers at the checkpoint (plus
the data and batch order, which are deterministic from the seeds). Runs 1-10 stored weights only.
Run 11 stores weights and momentum every 1000 steps.

### Changes
- `relu_analysis`: `designed_run_seed_result`, `save_designed_run` / `load_designed_run`
  (weights.npz + momentum.npz + meta.pkl, each < 50 MB), `regenerate_seed_result_data` extended to
  the `kind="sweep_chunked"` data spec, and `replay_interval_from_snapshot(seed_result, epoch_start,
  epoch_end=None, data=, record_full_data_gradient=, full_data_subsample=, record_params=,
  neuron_rows=)`: exact multi-epoch replay between checkpoints (stored lr history, momentum, fixed
  batch order) with per-step batch loss, mini-batch gradient, update and population gradient of
  chosen first-layer rows; verified against the next snapshot. Exactness test.
- Script `results/designed_init/run_hierarchy_replayable.py` (run 10's design; 1M points = run 10's
  first chunk; batch 10k so an epoch is 100 steps; 10 000 epochs = 1M steps; lr 0.01 -> 0.0005;
  momentum 0.9; snapshots with momentum every 10 epochs). First launch killed by a power failure
  (process terminated cleanly, code committed); relaunched after the restart, 26 min.
- Outputs tracked under `results/designed_init/hierarchy_replayable/` (replay folder 89 MB in two
  files); summaries under `notebooks/results/designed_init/hierarchy_replayable/`.
- Notebook section 11 in `hierarchy_design_runs.ipynb`; new notebook
  `notebooks/hierarchy_trajectory_analysis.ipynb` (load, overview, exact replay of one interval,
  per-step population-gradient components, mini-batch vs population gradient, network graph).

### Results
- Replay exact to the bit at every interval tested (epochs 0, 100, 500): 2 s per 1000-step
  interval plain, 13 s with per-step population gradients on 200k points, 53 s on all 1M.
- Loss 0.3 / 0.1 / 0.05 at steps 34k / 122k / 670k (run 10: 17k / 53k / 460k). All 15 internal
  nodes on their own probes at the intended pole from step 100k (14/15 at 50k), 31/31 covered
  from 100k; 24/30 probes at the intended pole, 0 opposite, 9 nodes at both poles; no unit path
  changed sign. Final loss 0.037, test error 1.58% (run 10: 0.029, 0.90%).
- Fluctuations over steps 100k-300k (loss per 1000 steps): std 0.013, range 0.056 (run 10:
  0.023, 0.141). Reusing 1M points ten times: no sign of overfitting.

### Current state / where to pick up
- Trajectory analysis phase in `hierarchy_trajectory_analysis.ipynb`: per-step population
  gradients during the capture of each node (steps 20k-100k), the root-pole pull on the +u_1 /
  -u_2 probes, and the late-phase slow drift.
- Not pushed; commit only.

## 2026-10-03 — Why runs 10 and 11 have an unstable loss phase and run 9 does not (Q&A, measured)

### User prompts (verbatim)
- "Why do you think the training loss curves for run 10 and run 11 go through an unstable phase (around epoch 200 for run 10 and epoch 2000 for run 11)? Run 9 for example does not."

### Measurements (top Hessian eigenvalue by power iteration on 100k training points; heavy-ball limit lr*lambda = 2(1+0.9) = 3.8)
- Run 9: lambda rises to ~165 by step 30k (lr*lambda 3.3-3.4) and then stays at 150-170 while the
  cosine lr anneals, so lr*lambda falls to 3.0 at 150k and 2.5 at 300k: the run leaves the edge of
  stability after ~50k steps. Per-1000-step loss std over 100k-150k: 0.0065, falling to 0.001.
- Run 10: lambda reaches 174 at step 20k and keeps pace with the lr (188 at 100k), so lr*lambda
  stays at 3.4-3.7 from 20k to 200k steps; loss windows show recurring spikes (min 0.065, max
  0.14-0.15 per 10-epoch window). Steps 210k-290k are a sequence of catapults: lambda collapses
  184 -> 155 -> 107 -> 41 (epoch 230, loss 0.137) -> 149 -> 88 -> 106 -> 126 -> 119 -> 164, with
  the top eigenvector moving from layer 2 (share 0.57) to layer 3 (0.82 at epoch 280). After
  epoch 290 lambda is ~150-160, lr*lambda 2.3-2.7, and the loss is smooth. Per-1000-step std over
  150k-300k: 0.015-0.032, then 0.003.
- Run 11 (lr halved): lambda doubles to track the limit (369 at 50k, 357 at 100k, 388 at 250k, 649
  at 500k, 1917 at 1M; lr*lambda 3.3-3.7 from 50k to 500k), top eigenvector in layers 2-3
  (0.58/0.41) throughout. Per-100-step loss spikes to 0.37-0.41 (5-8 per 20k steps) between
  100k and 180k steps, then none: std 0.001 from 180k on although lr*lambda is still 3.34 and
  lambda keeps growing. The loss plateaus (0.105 -> 0.083 over 100k-400k) while pinned.

### Reading
- The unstable phase is edge-of-stability behaviour: progressive sharpening pins lr*lambda near
  3.8, the period-2 oscillation along the top direction (layers 2-3, i.e. the unit weights and the
  third layer) grows, and catapults (loss spike, sharpness collapse, re-sharpening) follow. Run 10
  shows the catapult sequence at 210k-290k; run 11, with half the lr, shows spikes without
  catastrophic collapses at 100k-180k and then a damped oscillation while still pinned.
- Run 9 does not sharpen progressively: its lambda saturates at ~160, so the annealing lr takes it
  off the edge by ~50k steps. The one design difference is the third layer: the balanced
  +-10-quanta paths give every unit the same large gain to the logit, raising the curvature of the
  loss w.r.t. layers 2-3 (lambda 174 vs 43 at step 20k) and letting it keep rising as the lr
  anneals. Halving the lr (run 11) does not remove the edge: the sharpness doubles to compensate.
- Being at the edge costs loss: the snapshot sits on the oscillation's flank (0.103 vs 0.073 at
  the valley floor in run 10 at epoch 100), which is the plateau of run 11 at 0.08-0.1.

### Current state / where to pick up
- If the plateau/spikes matter: leave the edge rather than lower the lr uniformly (faster mid-run
  lr decay, or a smaller path majority such as 12/8 = +-4 quanta, trading sign robustness), or
  accept it and analyse the valley-floor trajectory (project out the top eigendirection).

## 2026-10-05 — Section 2b (recorded rows step by step) in the trajectory notebook

### User prompts (verbatim)
- "In the trajectory plots in the hierarchy_design_runs notebook, there is a line for \"fan-out norm\" for a first hidden layer neuron. Can you define that?"
- "In the new hierarchy_trajectory_analysis.ipynb notebook. Can you explain the schema of rep object output by replay_interval_from_snapshot? What its keys are and what is stored in them?"
- "Ok, now explain the the gser object output by first_layer_gradient_series"
- "Can you add a new section between current sections 2 and 3 of the hieararchy_design_runs.ipynb notebook that plots param_rows?"

### Changes
- `notebooks/hierarchy_trajectory_analysis.ipynb` (the notebook that has `rep` / `params_rows`;
  the prompt named the runs notebook): new section 2b between the replay and the gradient
  sections: cosines of the recorded rows to chosen ODT nodes and their norms step by step across
  the replayed interval, and each step's movement split into the part along the row (norm
  change) and perpendicular to it (rotation) next to the mini-batch gradient norm. Left
  unexecuted for the user's kernel; logic checked in a dry run (interval from epoch 100: probes
  0-5 drift smoothly, cosines 0.85 / -0.85 / 0.6 / -0.6 for the root and u_1 pairs).
- Definitions given in chat: fan-out norm of first-layer neuron i = ||W2[:, i]||, the norm of
  its outgoing column in the second layer; `rep` schema; `gser` columns.

## 2026-10-05 — Why a push with cosine 0.48 to u_0 raises row 0's cosine to u_0 (epochs 100-110 of run 11)

### User prompts (verbatim)
- "If there is no oscillation, I still cannot understand what happens from smooth phase epoch 100 to 110.\n\nFor example with row 0, the cosine with +u_0 stays around 0.85, but the push's cosine with +u_0 never exceeds 0.5. The norm is also increasing (from 1.6 to 1.65), so the push cannot be explained as the init shrinking to zero."

### Measurement (population push on row 0 over 200k points, exact replay of epochs 100-110)
- Row 0: norm 1.602 -> 1.650, cos(u_0) 0.854 -> 0.890. Push at step 0: |p| 0.00244, along u_0
  +0.00116, along the row's own off-u_0 direction e -0.00089, orthogonal to both 0.00196, along
  the row +0.00053 (cosines: 0.48 with u_0, 0.22 with the row).
- First-order rate d cos = (p.u_0 - cos * p.w_hat)/|w| with the step-0 push held for the interval
  predicts +0.044; actual +0.036. Displacement over the 1000 steps: +0.100 along u_0, -0.087 along
  e, 0.093 orthogonal to both, norm change +0.048; the sum of population pushes x lr/(1-beta)
  along u_0 is +0.102.
- The off-span part of the push (59% of its energy) has cosine -0.50 with the row's own off-span
  part and only 0.51 of its energy in the span of the cue rows' off-span components (random: 0.46),
  so it is not the cues' off-span leaking in. The row's off-span part shrinks 0.80 -> 0.71 and
  rotates by ~6 degrees over the interval (cosine 0.9954 start to end).

### Reading
cos(push, u_0) understates what the push does to the row's direction: besides adding u_0 it
removes the row's leftover init direction (the -0.00089 along e), and both raise cos(w, u_0).
The remaining sideways part rotates the off-span residue inside the orthogonal complement, which
changes cos(w, u_0) only at second order. The norm grows because the push's projection on the row
(+0.0005) is positive: 0.85 x 0.00116 - 0.53 x 0.00089. So the row moves toward u_0 by adding
u_0 and shedding its init component at the same time, not by a push along u_0.

## 2026-10-05 — Run 12: run 11 with the probe rows scaled by 0.01 at init; run selector in the trajectory notebook

### User prompts (verbatim)
- "Ok, can you create a new run  12, with only one change from run 11, where the probe incoming weight vectors at initialisation are multiplied by 0.01? \n\nKeep the trajectory_analysis notebook flexible, i.e. should be able to choose at the beginning if I want to analyse run 11 or run 12 (or any future replayable run)."

### Changes
- `HierarchyDesignSpec.probe_scale` (default 1.0): multiplies the 30 probe rows at the very end of
  the construction, after the row-norm rescale (otherwise the rescale would undo it). Test.
- Script `results/designed_init/run_hierarchy_replayable_probe001.py` = run 11's script with
  `probe_scale=0.01`; outputs tracked under `results/designed_init/hierarchy_replayable_probe001/`.
- `notebooks/hierarchy_trajectory_analysis.ipynb`: `RUN_NAME` in the first code cell selects any
  folder under `results/designed_init/` that holds a `replay/meta.pkl`; the available ones are
  listed. Executed on run 11 (default). Runs notebook section 12 with a third-layer-death cell.

### Results (run 12 vs run 11)
- Faster first-layer capture: root pair at cosine 0.95 by step 5k (run 11: 15-16k), -u_1 / +u_2
  at 17-19k (37k / 32k). Loss 0.3 at 41k (34k); then slower: 0.19 at 100k, 0.125 at 500k
  (0.113, 0.065). 14/15 internal nodes on own probes at the intended pole from 100k (node 8
  missed), 17/30 probes at the intended pole (24/30), 0 opposite, 31/31 covered; no path flips;
  14 units dropped their probe entry (8).
- Late collapse: third-layer units never active on 100k test points: 3 at 300k, 15 at 400k, 23 at
  500k, 36/40 at the end (active fraction 1.0 -> 0.54 -> 0.05 -> 0); loss spikes at ~400k (0.73)
  and a fully dead moment at ~600k (loss 0.693, 99.9% zero logits); settles into the half-dead
  solution (zero logits on the negative class, loss 0.378, test error 50% default / 3.4% strict).
  Run 11 never has a dead third-layer unit.
- Difference in the representation: second-layer units active on ~74% of inputs throughout in
  run 12 vs ~35% in run 11 (with the probes near zero the units are gated by the cue blocks
  alone); |W2| 60 vs 47; |wo| shrinks to 6 mid-run and grows to 26 as the third layer dies.

### Reading
Scaling the probes down speeds the rotation of the probe rows (angular velocity ~ push/|w|) but
removes the probe's role in gating its unit; the second layer becomes dense, and the third layer
is then driven to death during the edge-of-stability phase. The design wants the probes small in
the ODT span but not small as inputs to their units.

### Current state / where to pick up
- Trace the death of third-layer units step by step in the replay (steps 300k-600k), and test a
  milder scale (0.1) or a probe scale applied only to the in-span part.
- Not pushed; commit only.

## 2026-10-05 — Epoch window for trajectory_plots; back to run 11

### User prompts (verbatim)
- "Ok, this was a bad idea. I am going back to analysing run 11.\n\nCan you add an argument to the trajectory_plots function in hierarchy_design_runs notebook to take a start and end epoch and only plot points between the two?"

### Changes
- `trajectory_plots(run, neuron_ids, node_ids=None, start_epoch=None, end_epoch=None)` in the setup
  cell of `notebooks/hierarchy_design_runs.ipynb`: inclusive epoch window (in the run's own epoch
  units) applied to the loss curve, the zero-logit / coverage panels, the probe-target cosines and
  the alignment series; the default-node choice uses the last snapshot inside the window; an empty
  window raises with the stored epoch range. Setup cell edited in place, not re-executed (re-run it
  in the live kernel). Checked in a dry run on run 11 with epochs 100-600.

## 2026-10-05 — Do the loss fluctuations start because aligned first-layer rows sweep across decision boundaries? (Q&A, measured)

### User prompts (verbatim)
- "In run 11, I see a curious thing. The loss fluctuations start around 600 eopochs, the exact time around where all the first layer neurons capture all the ODT hyperplanes. \n\nDo you think the following is a feasible explanation? The loss is smooth as longs as the (non-frozen) first layer neurons are far from the ODT hyperplanes. Once they all get close to the ODT hyperplanes any rotation in them sweeps across a decision boundary in the data causing swift whiplash as ReLU units switch on and off for points they were not supposed to and thereby causing huge gradient changes."

### Measurements (run 11, 200k training points)
- Timeline: own-probe nodes 1 / 2 / 13 / 14 / 15 at steps 20k / 30k / 40k / 50k / 80k; lambda_max
  11 / 66 / 256 / 369 / 369 and lr*lambda 0.11 / 0.66 / 2.55 / 3.66 / 3.64 (limit 3.8); loss std
  over the following 10k steps 0.057 (escape) -> 0.011 at 40k -> 0.026-0.042 from 60k on. The
  fluctuations begin when lr*lambda reaches the limit, which happens as the capture completes.
- Where the sharpness lives: the top eigenvector has 0.3-0.7% of its energy in layer 1, 72% in
  layer 2 and 26% in layer 3 at step 60k. Curvature along a random rotation of all free first-layer
  rows is 0.23, along a random layer-2 direction 0.41, against lambda_max 362.
- Sweeping test at step 60k: rotating every free row by 1 degree flips 0.33% of (point, row) gates
  (23% of points have some flipped gate among 48 free rows) and raises the loss from 0.145 to 0.160,
  smoothly; the probe rows actually rotate 0.27 degrees (median, max 1.25) per 1000 steps then.

### Reading
Timing right, location wrong: capture does trigger the fluctuations, but not through first-layer
rotations sweeping boundaries. Once the first layer presents the hyperplane indicators, the loss
becomes stiff in the weights that combine them (layers 2-3; curvature w.r.t. W2 scales with the
second moments of the now label-aligned features), lr*lambda hits the heavy-ball limit, and the
run oscillates across that valley (period 2) with layer-2/3 ReLU units switching each step. The
first layer barely moves and its gate flips are rare and smooth.

## 2026-10-05 — Runs 13 and 14: branches of run 11 at epoch 750 with the first layer frozen (SGD continued / AdamW)

### User prompts (verbatim)
- "Ok. How about the following ideas for some new runs:\n\n1. Same as run 11 till epoch 750, and after that the first layer weights are frozen.\n2. Same as run 11 till epoch 750, after that the first layer weights are frozen and the other layers are optimized using Adam instead of SGD with momentum."

### Changes
- `train_relu_mlp_with_optimizer_snapshots(..., start_epoch=, initial_optimizer_states=)`: resumes
  the cosine schedule at an epoch (base lr restored before advancing the recursive scheduler) and
  optionally continues a stored optimizer state exactly; losses / lr history keep length
  `config.epochs` with NaN before the start. `save_designed_run` / `load_designed_run` now store
  every optimizer-state tensor (`optimizer_state.npz`: SGD momentum, or AdamW moments + step;
  the old `momentum.npz` files still load); interval replay verifies all state tensors. Test:
  resuming at epoch 4 reproduces a run to the bit; an AdamW branch with frozen first layer
  round-trips and replays exactly.
- Script `results/designed_init/run_hierarchy_branch750.py`; tracked outputs under
  `results/designed_init/hierarchy_branch750_{sgd,adam}_frozenL1/`. Runs notebook sections 13-14
  and a three-run comparison cell.

### Results (per-epoch loss mean / std over 500-epoch windows; epochs of 100 steps)
| epochs | run 11 | 13: SGD, frozen L1 | 14: AdamW, frozen L1 |
|---|---|---|---|
| 750-1250 | 0.126 / 0.041 | 0.131 / 0.003 | 0.110 / 0.004 |
| 1000-1500 | 0.114 / 0.043 | 0.128 / 0.002 | 0.106 / 0.001 |
| 2000-2500 | 0.090 / 0.002 | 0.118 / 0.002 | 0.102 / 0.0003 |
| end | 0.037 (test 1.6%) | 0.094 (test 3.9%) | 0.092 (test 3.85%) |
- Sharpness is unchanged by the freeze: lr*lambda 3.3-3.6 with the top eigenvector in layers 2-3
  in both run 11 and run 13 at epochs 800-3000; run 14 tolerates raw lambda up to 1570.
- Run 11's first-layer motion after 750 is in the random rows (63, 69, 77 rotate 74-91 degrees,
  recruited to level-4 nodes; probes 2-6 degrees); per-10-epoch first-layer movement vs loss std
  has rank correlation 0.74 (max row rotation) / 0.77 (||dW1||); random row 69 is the fastest
  row in most of the spikiest windows.
- Both branches replay exactly (params and optimizer state diff 0).

### Reading
- The loss spikes need a trainable first layer: with it frozen the period-2 oscillation in
  layers 2-3 persists (same lr*lambda) but the catapults vanish. The spikes are driven by
  first-layer rows (mostly random rows being recruited to level-4 hyperplanes) moving across
  data boundaries on top of the stiff layer-2/3 landscape, which is close to the user's
  sweeping hypothesis, with the stiffness supplying the amplification.
- The late first-layer refinement is also what buys the final loss: frozen at 750, both
  optimizers plateau near 0.093 (node 9 at cosine 0.93, random rows not yet recruited); run 11
  gets to 0.037 by recruiting random rows and growing probe norms x1.8.

### Current state / where to pick up
- Freeze later (e.g. epoch 3000, after the random rows have settled) to separate the two effects;
  or freeze only the probes and cues, leaving the random rows free.
- Not pushed; commit only.
