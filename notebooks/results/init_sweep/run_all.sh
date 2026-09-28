#!/usr/bin/env bash
# Init-sweep launcher (Sep 28, 2026). Part A: invariance grid. Part B: 200 inits, full trajectories.
set -u
cd /home/harish/researchcodes/layer-lenses
PY=.venv/bin/python
$PY -m layer_lenses.init_sweep part-a --out results/init_sweep/partA \
    --init-seeds 6178 6201 6202 6203 6204 6205 \
    --data-seeds 5178 5279 5380 \
    --batch-sizes 10000 1000 \
    > results/init_sweep/partA.log 2>&1
$PY -m layer_lenses.init_sweep part-b --out results/init_sweep/partB \
    --data-seed 5178 --batch-size 10000 \
    --init-seeds 6178 6201 6202 6203 6204 6205 $(seq 7000 7193) \
    > results/init_sweep/partB.log 2>&1
echo "ALL DONE $(date)" >> results/init_sweep/partB.log
