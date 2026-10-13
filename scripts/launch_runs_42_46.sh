#!/usr/bin/env bash
# Runs 42-44 (run 38 without the third layer: plain / output gradient x41 / output frozen) three at a time, then runs 45-46 (ScalarAdam lr 0.01 / 0.03 constant) two at a time.
cd "$(dirname "$0")/.."
S=results/designed_init/run_hierarchy_equivariant60.py; PY=.venv/bin/python
$PY $S --out hierarchy_equiv2L_62 --tag linear_free_kaiming_cosine_equiv2L_62_seed0 --no-spare --two-layer > results/designed_init/hierarchy_equiv2L_62/log.txt 2>&1 &
$PY $S --out hierarchy_equiv2L_62_outx41 --tag linear_free_kaiming_cosine_equiv2L_62_outx41_seed0 --no-spare --two-layer --out-scale 41 > results/designed_init/hierarchy_equiv2L_62_outx41/log.txt 2>&1 &
$PY $S --out hierarchy_equiv2L_62_outfrozen --tag linear_free_kaiming_cosine_equiv2L_62_outfrozen_seed0 --no-spare --two-layer --freeze-out > results/designed_init/hierarchy_equiv2L_62_outfrozen/log.txt 2>&1 &
wait
$PY $S --out hierarchy_equiv2L_62_sadam01 --tag linear_free_kaiming_equiv2L_62_sadam01_seed0 --no-spare --two-layer --optimizer scalar_adam --const-lr --lr 0.01 > results/designed_init/hierarchy_equiv2L_62_sadam01/log.txt 2>&1 &
$PY $S --out hierarchy_equiv2L_62_sadam03 --tag linear_free_kaiming_equiv2L_62_sadam03_seed0 --no-spare --two-layer --optimizer scalar_adam --const-lr --lr 0.03 > results/designed_init/hierarchy_equiv2L_62_sadam03/log.txt 2>&1 &
wait
echo "CHAIN DONE"
