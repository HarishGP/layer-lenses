#!/usr/bin/env bash
cd "$(dirname "$0")/.."
S=results/designed_init/run_hierarchy_symprobes.py
uv run python $S --width 80 --data-seed 5178 --out hierarchy_symprobes_80 --tag linear_free_kaiming_cosine_symprobes_80_seed0 > results/designed_init/hierarchy_symprobes_80/log.txt 2>&1 &
uv run python $S --width 62 --data-seed 5178 --out hierarchy_symprobes_62 --tag linear_free_kaiming_cosine_symprobes_62_seed0 > results/designed_init/hierarchy_symprobes_62/log.txt 2>&1 &
wait
uv run python $S --width 80 --data-seed 6178 --out hierarchy_symprobes_80_ds2 --tag linear_free_kaiming_cosine_symprobes_80_ds6178 > results/designed_init/hierarchy_symprobes_80_ds2/log.txt 2>&1 &
uv run python $S --width 62 --data-seed 6178 --out hierarchy_symprobes_62_ds2 --tag linear_free_kaiming_cosine_symprobes_62_ds6178 > results/designed_init/hierarchy_symprobes_62_ds2/log.txt 2>&1 &
wait
echo "CHAIN DONE"
