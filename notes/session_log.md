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
