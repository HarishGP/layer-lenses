#!/usr/bin/env bash
# Regenerate every file that notebooks/hierarchy_design_runs.ipynb reads from results/designed_init/
# (parameter trajectories runs/*.npz, summary.csv, per_epoch_*.csv, tree.npz, manifest.json).
# results/ is git-ignored, so a fresh clone has none of these.
#
# Requirements: a CUDA GPU with >= 8 GB free memory (the 10M-point runs hold the whole dataset on
# the GPU, 3.7 GB), >= 8 GB free host RAM, and the project environment (`uv sync`).
# Runtime on an RTX 5060 Ti: about 25 min per run, 7 runs, ~3 hours in total. Each script is
# independent; comment out the ones you do not need. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-uv run python}"
S=notebooks/results/designed_init

$PY $S/hierarchy_kaiming_cosine/run_hierarchy_kaiming_cosine.py           # runs 1-2: 80-40-40, 1M/1k, cues free and frozen (~45 min)
$PY $S/hierarchy_kaiming_cosine_big/run_hierarchy_kaiming_cosine_big.py   # run 3:    62-30-40, 10M/10k, frozen (~25 min)
$PY $S/hierarchy_kaiming_cosine_isolate/run_hierarchy_kaiming_cosine_isolate.py  # runs 4-5: 62-30-40 at 1M/1k, 80-30-40 at 10M/10k (~50 min)
$PY $S/hierarchy_kaiming_cosine_balanced/run_hierarchy_kaiming_cosine_balanced.py  # runs 6-7: balanced output, 62 and 80 rows, 10M/10k (~50 min)

echo "All runs regenerated under results/designed_init/. Execute the notebook with:"
echo "  cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace hierarchy_design_runs.ipynb"
