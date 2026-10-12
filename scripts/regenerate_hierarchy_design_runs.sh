#!/usr/bin/env bash
# Regenerate every file that notebooks/hierarchy_design_runs.ipynb reads from results/designed_init/
# (parameter trajectories runs/*.npz, summary.csv, per_epoch_*.csv, tree.npz, manifest.json).
# results/ is git-ignored, so a fresh clone has none of these.
#
# Requirements: a CUDA GPU with >= 8 GB free memory (the 10M-point runs hold the whole dataset on
# the GPU, 3.7 GB), >= 8 GB free host RAM, and the project environment (`uv sync`).
# Runtime on an RTX 5060 Ti: about 25 min per run, 16 runs, ~7 hours in total. Each script is
# independent; comment out the ones you do not need. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-uv run python}"
S=notebooks/results/designed_init

$PY $S/hierarchy_kaiming_cosine/run_hierarchy_kaiming_cosine.py           # runs 1-2: 80-40-40, 1M/1k, cues free and frozen (~45 min)
$PY $S/hierarchy_kaiming_cosine_big/run_hierarchy_kaiming_cosine_big.py   # run 3:    62-30-40, 10M/10k, frozen (~25 min)
$PY $S/hierarchy_kaiming_cosine_isolate/run_hierarchy_kaiming_cosine_isolate.py  # runs 4-5: 62-30-40 at 1M/1k, 80-30-40 at 10M/10k (~50 min)
$PY $S/hierarchy_kaiming_cosine_balanced/run_hierarchy_kaiming_cosine_balanced.py  # runs 6-7: balanced output, 62 and 80 rows, 10M/10k (~50 min)
$PY $S/hierarchy_kaiming_cosine_pathsign/run_hierarchy_kaiming_cosine_pathsign.py  # run 8:    path-sign-aware cues (~25 min)
$PY $S/hierarchy_kaiming_cosine_cueeq/run_hierarchy_kaiming_cosine_cueeq.py        # run 9:    + inv_sqrt cue blocks (~25 min)
$PY $S/hierarchy_kaiming_cosine_balpaths/run_hierarchy_kaiming_cosine_balpaths.py  # run 10:   + balanced +-10-quanta third-layer paths (~25 min)
$PY $S/hierarchy_replayable/run_hierarchy_replayable.py                            # run 11:   replayable version of run 10 (1M/10k, lr 0.01; ~26 min)
$PY $S/hierarchy_replayable_probe001/run_hierarchy_replayable_probe001.py          # run 12:   run 11 with probe rows x0.01 (~26 min)
$PY $S/hierarchy_branch750/run_hierarchy_branch750.py                              # runs 13-14: run 11 branched at epoch 750, first layer frozen (needs run 11's replay files; ~50 min)
$PY $S/hierarchy_branch750_adam_all/run_hierarchy_branch750_adam_all.py            # run 15:   run 11 branched at epoch 750, all layers AdamW (needs run 11; ~26 min)
$PY $S/hierarchy_rotlock099/run_hierarchy_rotlock.py                               # run 16:   run 11 with the first-layer rotation lock at |cos| 0.99 (~26 min)
$PY $S/hierarchy_scalaradam/run_hierarchy_scalaradam.py                            # run 17:   run 11 trained with ScalarAdam (normalised step, lr 0.001; ~30 min)
$PY $S/hierarchy_scalaradam_lr003/run_hierarchy_scalaradam_lr003.py              # run 18:   ScalarAdam, lr 0.003 constant (~32 min)
$PY $S/hierarchy_scalaradam_lr01/run_hierarchy_scalaradam_lr01.py                # run 19:   ScalarAdam, lr 0.01 constant (~40 min)
$PY $S/hierarchy_scalaradam_lr03/run_hierarchy_scalaradam_lr03.py                # run 20:   ScalarAdam, lr 0.03 constant (~40 min)
$PY $S/hierarchy_replayable_62/run_hierarchy_replayable_62.py                    # run 21:   run 11 without the 18 random rows (62-30-40; ~26 min)
$PY $S/hierarchy_idealL1/run_hierarchy_idealL1.py                                # run 22:   run 21 with the first layer frozen and probes on their targets (~25 min)
$PY $S/hierarchy_idealL1_zerocue/run_hierarchy_idealL1_zerocue.py                # run 23:   run 22 with the W2 cue block zeroed at init (~25 min)
$PY $S/hierarchy_replayable_probepos/run_hierarchy_replayable_probepos.py          # run 24:   run 11 with probe->unit weights +1 (~26 min)
$PY $S/hierarchy_replayable_w2neg/run_hierarchy_replayable_w2neg.py                # run 25:   run 11 with W2 negated (~26 min)
$PY $S/hierarchy_replayable_cuev/run_hierarchy_replayable_cuev.py                  # run 26:   run 11 with cues pole*(v + u_j), free (~26 min)
$PY $S/hierarchy_symprobes_80/run_hierarchy_symprobes.py --width 80 --data-seed 5178 --out hierarchy_symprobes_80 --tag linear_free_kaiming_cosine_symprobes_80_seed0        # run 27 (~26 min)
$PY $S/hierarchy_symprobes_62/run_hierarchy_symprobes.py --width 62 --data-seed 5178 --out hierarchy_symprobes_62 --tag linear_free_kaiming_cosine_symprobes_62_seed0        # run 28 (~26 min)
$PY $S/hierarchy_symprobes_80_ds2/run_hierarchy_symprobes.py --width 80 --data-seed 6178 --out hierarchy_symprobes_80_ds2 --tag linear_free_kaiming_cosine_symprobes_80_ds6178  # run 29 (~26 min)
$PY $S/hierarchy_symprobes_62_ds2/run_hierarchy_symprobes.py --width 62 --data-seed 6178 --out hierarchy_symprobes_62_ds2 --tag linear_free_kaiming_cosine_symprobes_62_ds6178  # run 30 (~26 min)
$PY $S/hierarchy_cuev_62/run_hierarchy_cuev62.py --out hierarchy_cuev_62 --tag linear_free_kaiming_cosine_cuev_62_seed0                              # run 31 (~26 min)
$PY $S/hierarchy_cuev_62_w3seed1/run_hierarchy_cuev62.py --out hierarchy_cuev_62_w3seed1 --tag linear_free_kaiming_cosine_cuev_62_w3seed1 --w3-seed 1   # run 32 (~26 min)
$PY $S/hierarchy_cuev_62_ds2/run_hierarchy_cuev62.py --out hierarchy_cuev_62_ds2 --tag linear_free_kaiming_cosine_cuev_62_ds6178 --data-seed 6178                                  # run 33
$PY $S/hierarchy_cuev_62_w3seed1_ds2/run_hierarchy_cuev62.py --out hierarchy_cuev_62_w3seed1_ds2 --tag linear_free_kaiming_cosine_cuev_62_w3seed1_ds6178 --w3-seed 1 --data-seed 6178   # run 34
$PY $S/hierarchy_cuev_62_ps7/run_hierarchy_cuev62.py --out hierarchy_cuev_62_ps7 --tag linear_free_kaiming_cosine_cuev_62_ps7 --probe-seed 7                                          # run 35
$PY $S/hierarchy_cuev_62_w3seed1_ps7/run_hierarchy_cuev62.py --out hierarchy_cuev_62_w3seed1_ps7 --tag linear_free_kaiming_cosine_cuev_62_w3seed1_ps7 --w3-seed 1 --probe-seed 7      # run 36
$PY $S/hierarchy_equivariant_60/run_hierarchy_equivariant60.py --out hierarchy_equivariant_60 --tag linear_free_kaiming_cosine_equiv60_seed0                                      # run 37
$PY $S/hierarchy_equivariant_60_nospare/run_hierarchy_equivariant60.py --out hierarchy_equivariant_60_nospare --tag linear_free_kaiming_cosine_equiv60_nospare_seed0 --no-spare   # run 38

echo "All runs regenerated under results/designed_init/. Execute the notebook with:"
echo "  cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace hierarchy_design_runs.ipynb"
