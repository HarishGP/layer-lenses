#!/usr/bin/env bash
# Clean comparable runs, long schedule (runs 131-138, 2026-10-24): as scripts/launch_clean_runs.sh (runs 123-130) but 4000 epochs, constant
# step 0.1 for 1500 epochs then a cosine to 0.005 over 2500, so that the design-70 symmetrised run also reaches a loss below 0.01.
#   SGDA: 131 (design 60, 2 phases), 135 (design 60, 1 phase)     SGDB: 132 (design 70, 2 phases), 136 (design 70, 1 phase)
#   SYMA: 133 (design 60, 2 phases), 137 (design 60, 1 phase)     SYMB: 134 (design 70, 2 phases), 138 (design 70, 1 phase)
# Usage: scripts/launch_clean_runs_long.sh SGDA | SGDB | SYMA | SYMB   (one tmux session each)
cd "$(dirname "$0")/.."
PY=.venv/bin/python; S=results/designed_init/run_symmetric_gd.py; R=results/designed_init
run() { local name=$1; shift; mkdir -p $R/$name; $PY $S --out $name --tag $name "$@" > $R/$name/log.txt 2>&1; }
C="--alpha-json {} --epochs 4000 --warm-epochs 1500"
case "$1" in
  SGDA) run hd2_run131_clean4k_sgd_d60_2ph $C --alpha-base 60 --freeze-cues 1000 --no-symmetrise; run hd2_run135_clean4k_sgd_d60_1ph $C --alpha-base 60 --freeze-cues 0 --no-symmetrise;;
  SGDB) run hd2_run132_clean4k_sgd_d70_2ph $C --alpha-base 70 --freeze-cues 1000 --no-symmetrise; run hd2_run136_clean4k_sgd_d70_1ph $C --alpha-base 70 --freeze-cues 0 --no-symmetrise;;
  SYMA) run hd2_run133_clean4k_sym_d60_2ph $C --alpha-base 60 --freeze-cues 1000; run hd2_run137_clean4k_sym_d60_1ph $C --alpha-base 60 --freeze-cues 0;;
  SYMB) run hd2_run134_clean4k_sym_d70_2ph $C --alpha-base 70 --freeze-cues 1000; run hd2_run138_clean4k_sym_d70_1ph $C --alpha-base 70 --freeze-cues 0;;
esac
echo "CHAIN $1 DONE"
