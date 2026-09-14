# Detailed Repository Guide

This document gives a more detailed map of the repository and the Python package. It is intended for future work sessions where the first step is understanding where reusable logic lives.

## Project Purpose

The repository reimplements and extends experiments around two related research projects:

- "Deep Networks Learn Features from Local Discontinuities in the Label Function"
- "Layers as Lenses: A Narrative of Feature Learning in Deep Networks"

The central synthetic data setting is a complete orthogonal balanced oblique decision tree (COB-ODT). Current work includes both DLGN-SF baselines and ReLU MLP analysis on the same ODT data.

## Top-Level Directories

- `src/layer_lenses/` contains stable importable code.
- `notebooks/` contains exploratory and analysis notebooks.
- `notebooks/old_notebooks/` contains legacy reference notebooks and paper sources. Treat these as read-only unless explicitly asked otherwise.
- `notes/` contains project specifications, reproducibility records, paper-vs-notebook mismatch decisions, and session logs.
- `tests/` contains lightweight validation and regression tests.
- `results/` is the intended output root for generated run artifacts.

## Python Package Overview

### `src/first_experiment/__init__.py`

Backward-compatibility shim for artifacts serialized before the package rename.
It forwards old module paths like `first_experiment.relu_mlp` to `layer_lenses.relu_mlp`
so legacy pickle files can still be loaded.

### `src/layer_lenses/odt.py`

COB-ODT data generation and tree traversal utilities.

- `COBODTSpec`: dataclass describing input dimension, tree depth, and pruning threshold.
- `COBODTTree`: dataclass containing ODT internal hyperplanes, biases, and leaf labels.
- `_sample_unit_sphere`: samples input points uniformly from the unit sphere by normalizing Gaussian vectors.
- `_sample_orthonormal_vectors`: samples mutually orthonormal ODT decision vectors using QR.
- `_default_leaf_labels`: builds deterministic alternating `{−1,+1}` leaf labels.
- `_traverse_tree`: maps margin values to heap-indexed ODT leaf nodes.
- `odt_leaf_ids_for_x`: returns heap-style ODT leaf IDs reached by each input.
- `_validate_tree`: checks tree array shapes and label values against a spec.
- `_path_directions_from_root`: returns root-to-node left/right decisions.
- `_infer_depth_from_tree`: infers complete-tree depth from internal-node count.
- `samples_reaching_node`: returns samples, and optionally a mask, that reach a given internal or leaf node.
- `build_default_cob_odt_tree`: constructs a default paper-style COB-ODT tree.
- `build_axis_aligned_cob_odt_tree`: constructs a COB-ODT tree whose internal hyperplanes are coordinate axes.
- `generate_cob_odt_data`: generates sphere-uniform inputs, labels them with a COB-ODT, applies threshold pruning, and returns data/tree/metadata.

### `src/layer_lenses/dlgn.py`

DLGN-SF model definition.

- `DLGNSF`: Deep Linearly Gated Network shallow-features variant. Gating layers read raw input directly; value layers form the multiplicative value network. Supports `value_input_mode="ones"` and `"x"`.
- `DLGNSF.effective_gating_weights`: returns gating-layer weights for analysis.
- `DLGNSF.forward`: computes scalar logits for binary classification.

### `src/layer_lenses/training.py`

Training and evaluation utilities for DLGN-SF models.

- `TrainConfig`: dataclass for DLGN-SF training hyperparameters and optional freezing controls.
- `set_seed`: sets deterministic NumPy and PyTorch seeds.
- `_labels_pm1_to_binary`: converts labels from `{−1,+1}` to `{0,1}`.
- `evaluate_dlgn_sf`: evaluates BCE-with-logits loss, zero-one loss, and accuracy.
- `effective_gating_weights_from_checkpoint`: loads a checkpoint into a model copy and returns gating weights.
- `train_dlgn_sf`: trains DLGN-SF with Adam, optional scheduler, checkpoint snapshots, and optional parameter/row freezing.

### `src/layer_lenses/relu_mlp.py`

Standard ReLU MLP model definition.

- `ReLUMLP`: fully connected ReLU network for binary classification with scalar logits and Gaussian initialization matching PyTorch linear-default variance.
- `ReLUMLP.forward`: applies hidden linear layers with ReLU and a final linear output layer.

### `src/layer_lenses/relu_training.py`

Training and evaluation utilities for ReLU MLPs.

- `ReLUTrainConfig`: dataclass for single-phase ReLU MLP training (`optimizer`: `"adamw"` default or `"sgd"`; `momentum` is used only with SGD and defaults to `0.0`; optional `first_layer_grad_orthogonal_to` projection vector; optional 0/1 `parameter_update_mask` for freezing individual parameter entries).
- `_labels_pm1_to_binary`: converts labels from `{−1,+1}` to `{0,1}`.
- `evaluate_relu_mlp`: evaluates BCE-with-logits loss, zero-one loss, and accuracy.
- `checkpoint_model_from_state`: copies a template ReLU MLP and loads one checkpoint state dict.
- `train_relu_mlp`: trains a ReLU MLP with AdamW, cosine LR annealing, and checkpoint snapshots.

### `src/layer_lenses/relu_analysis.py`

Reusable ReLU-on-ODT analysis helpers. This module keeps `notebooks/scratch.ipynb` focused on plotting choices and interpretation rather than reusable mechanics.

- `quadratic_snapshot_epochs`: returns snapshot epochs dense near initialization and sparser later.
- `log_loss_gradients`: mean BCE-with-logits loss gradients w.r.t. all parameters for a given `(x, y)` batch (no optimizer step).
- `run_single_relu_seed`: generates COB-ODT data, optionally with coordinate-axis ODT hyperplanes, initializes a ReLU MLP or uses a provided `model_init`, trains one seed, can thread a 0/1 `parameter_update_mask` into training, and can project first-layer gradients orthogonal to a selected ODT internal node via `project_first_layer_grad_orthogonal`.
- `collect_start_end_metrics`: evaluates a ReLU run at first and last checkpoints.
- `first_layer_odt_alignment`: computes first-layer weight norm/cosine alignment to ODT decision vectors.
- `plot_first_layer_odt_alignment`: plots first-layer norm vs max ODT cosine, colored by ODT node level.
- `_layer_y_positions`: helper for vertical node placement in layered graph plots.
- `plot_layered_relu_graph`: draws a layered graph of hidden ReLU nodes and thresholded learned edges, with first-layer ODT labels.
- `masked_relu_logits`: runs a ReLU MLP while zeroing hidden activations according to a node mask.
- `masked_accuracy_by_odt_leaf`: groups masked-model accuracy and log loss by reached ODT leaf.
- `neuron_leaf_responsibility_from_ablation`: ablates one hidden neuron at a time and records leaves whose mean log loss rises.
- `summarize_neuron_responsibility`: converts the responsibility dictionary into a compact dataframe.
- `_precompute_hidden_activation_masks_and_leaf_ids`: precomputes hidden activation masks and ODT leaf IDs for repeated neuron/leaf queries.
- `_resolve_hidden_neuron_id`: resolves either global hidden-neuron IDs or `(layer, local_id)` tuples.
- `_precompute_hidden_activation_masks`: precomputes hidden activation masks without requiring ODT leaf IDs.
- `active_point_mask_for_neuron`: returns a boolean mask of points in `x_eval` that activate a chosen hidden neuron.
- `count_active_points_for_neuron_leaf`: counts how many points in a chosen ODT leaf activate a chosen hidden neuron.
- `summarize_neuron_leaf_activity`: builds an all-neuron/all-leaf activity dataframe.
- `neuron_leaf_activity_dict`: converts an activity dataframe into a dictionary mapping neuron IDs to leaf-ordered activity vectors.

### `src/layer_lenses/experiment_config.py`

Dataclass contracts for structured multipath DLGN experiments.

- `SeedBundle`: resolved seeds for data, initialization, phase 1, phase 2, and one-phase runs.
- `SeedOffsets`: deterministic offsets from a master seed.
- `SeedOffsets.resolve`: constructs a `SeedBundle`.
- `DataConfig`: dataset dimension, depth, train size, threshold, and derived total data count.
- `ModelConfig`: DLGN-SF architecture and initialization settings.
- `ExperimentConfig`: top-level experiment configuration.
- `ExperimentConfig.with_master_seed`: returns a config clone with a new master seed.
- `ExperimentConfig.phase2_epochs`: derived phase-2 epoch count.
- `ExperimentConfig.seeds`: resolved `SeedBundle`.
- `ExperimentConfig.train_config_phase1`: builds a `TrainConfig` for phase 1.
- `ExperimentConfig.train_config_phase2_nozero`: builds a phase-2 `TrainConfig` without zero-row freezing.
- `ExperimentConfig.train_config_phase2_zeroed`: builds a phase-2 `TrainConfig` with zero-row freezing.
- `ExperimentConfig.train_config_one_phase`: builds a one-phase `TrainConfig`.
- `EvalMetrics`: stable evaluation metric schema.
- `PruningSummary`: summary of zeroed/frozen gating rows.
- `RegimeResult`: result payload for one regime.
- `SeedRunResult`: result payload for all regimes under one master seed.
- `SeedRunResult.to_dict`: serializes a seed result.

### `src/layer_lenses/experiment_runner.py`

Single-seed regime runners for multipath DLGN experiments.

- `_build_model`: constructs a `DLGNSF` from `ExperimentConfig`.
- `_to_eval_metrics`: converts metric dictionaries into `EvalMetrics`.
- `_evaluate_regime`: evaluates train and test metrics for a trained model.
- `_zero_low_norm_gating_rows`: zeros gating rows below a norm threshold and returns freeze masks.
- `_rows_still_zero`: checks whether frozen rows stayed zero.
- `run_one_phase`: trains and evaluates a one-phase DLGN-SF run.
- `run_two_phase_nozero`: trains and evaluates phase 2 from a phase-1 model without zeroing.
- `run_two_phase_zeroed`: zeros low-norm gating rows, freezes them, then trains/evaluates phase 2.
- `run_all_regimes_for_seed`: runs phase-1 setup plus one-phase and two phase-2 variants for a master seed.

### `src/layer_lenses/multiseed.py`

API-first multi-seed orchestration.

- `run_multiseed`: runs `run_all_regimes_for_seed` over many master seeds, computes aggregate summaries, and optionally writes artifacts.

### `src/layer_lenses/analysis.py`

Aggregate descriptive statistics for multipath multi-seed results.

- `_extract_metric_rows`: flattens seed results into metric rows.
- `_rows_for_regime`: builds rows for one regime and train/test split.
- `aggregate_descriptive_stats`: computes mean, std, SEM, and normal-approximate 95% CI by regime/split/metric.
- `compute_delta_vs_baseline`: computes mean deltas relative to a baseline regime.

### `src/layer_lenses/io.py`

Result writing helpers.

- `ensure_run_root`: creates and returns a run artifact directory.
- `write_run_manifest`: writes run metadata and config to `manifest.json`.
- `write_seed_result`: writes one seed result to JSON.
- `write_summary_rows`: writes list-of-dict summaries to CSV.

### `src/layer_lenses/__init__.py`

Currently empty package initializer.

## Notebooks

- `notebooks/scratch.ipynb`: current exploratory ReLU MLP analysis on COB-ODT data. It should import reusable mechanics from `src/layer_lenses/relu_analysis.py`.
- `notebooks/DLGN_multiseed_multipath_runner.ipynb`: exploratory runner for DLGN multipath/multiseed regimes.
- `notebooks/legacy_dlgn_minimal.ipynb`: legacy/minimal DLGN reference notebook.
- `notebooks/old_notebooks/`: read-only reference notebooks and paper source material.

## Notes

- `notes/research_spec.md`: project goals, source-of-truth policy, and ReLU analysis plan.
- `notes/research_spec_math.ipynb`: notebook-rendered companion for math-heavy research notes.
- `notes/reproducibility_log.md`: hyperparameter provenance and experiment reproducibility notes.
- `notes/paper_mismatches.md`: documented paper-vs-notebook discrepancies and resolutions.
- `notes/session_log.md`: append-only log of non-trivial agent work sessions.

## Tests

- `tests/test_odt.py`: COB-ODT generation and traversal behavior.
- `tests/test_dlgn_training.py`: DLGN-SF training/evaluation behavior.
- `tests/test_relu_mlp.py`: ReLU MLP model initialization behavior.
- `tests/test_relu_training.py`: ReLU MLP training/evaluation behavior.
- `tests/test_relu_analysis.py`: ReLU-on-ODT analysis helpers (activity summaries, activation tensors, gradients).
- `tests/test_rename_compat.py`: compatibility for old `first_experiment.*` pickle module paths.
- `tests/test_experiment_runner.py`: single-seed regime runner behavior.
- `tests/test_multiseed_analysis.py`: aggregate multi-seed summaries.

Run tests with:

```bash
uv run pytest
```

## Development Conventions

- Keep stable reusable code in `src/layer_lenses/`.
- Keep notebooks exploratory; migrate repeated logic into `src/layer_lenses/`.
- Preserve seeds explicitly.
- Save generated artifacts under `results/`.
- Treat old notebooks and source papers as reference material, not as editable implementation targets.
- Prefer paper details over notebook behavior when sources conflict, and record decisions in `notes/paper_mismatches.md` or `notes/reproducibility_log.md`.
