#!/usr/bin/env bash
# Runs 33-34 (runs 31-32 with training-data seed 6178) then runs 35-36 (runs 31-32 with probe directions from seed 7), two at a time.
cd "$(dirname "$0")/.."
S=results/designed_init/run_hierarchy_cuev62.py
uv run python $S --out hierarchy_cuev_62_ds2 --tag linear_free_kaiming_cosine_cuev_62_ds6178 --data-seed 6178 > results/designed_init/hierarchy_cuev_62_ds2/log.txt 2>&1 &
uv run python $S --out hierarchy_cuev_62_w3seed1_ds2 --tag linear_free_kaiming_cosine_cuev_62_w3seed1_ds6178 --w3-seed 1 --data-seed 6178 > results/designed_init/hierarchy_cuev_62_w3seed1_ds2/log.txt 2>&1 &
wait
uv run python $S --out hierarchy_cuev_62_ps7 --tag linear_free_kaiming_cosine_cuev_62_ps7 --probe-seed 7 > results/designed_init/hierarchy_cuev_62_ps7/log.txt 2>&1 &
uv run python $S --out hierarchy_cuev_62_w3seed1_ps7 --tag linear_free_kaiming_cosine_cuev_62_w3seed1_ps7 --w3-seed 1 --probe-seed 7 > results/designed_init/hierarchy_cuev_62_w3seed1_ps7/log.txt 2>&1 &
wait
echo "CHAIN DONE"
