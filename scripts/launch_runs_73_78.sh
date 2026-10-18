#!/usr/bin/env bash
# Runs 73-78 (symmetrised-flow runs from the inits of runs 70, 60 and 60 + partner links) in two chains, each meant for one tmux session.
# Usage: scripts/launch_runs_73_78.sh A   |   scripts/launch_runs_73_78.sh B
cd "$(dirname "$0")/.."
PY=.venv/bin/python; S=results/designed_init/run_symmetric_gd.py; R=results/designed_init
run() { name=$1; shift; mkdir -p $R/$name; $PY $S --out $name --tag $name "$@" > $R/$name/log.txt 2>&1; }
if [ "$1" = "A" ]; then
  run hd2_run73_sym_pgf_two_phase --init-run hd2_run70_symcue_partner --epochs 1500 --freeze-cues 1000
  run hd2_run76_sym_pgf_one_phase --init-run hd2_run70_symcue_partner --epochs 1500 --freeze-cues 0
else
  run hd2_run74_sym_pgf_run60 --init-run hd2_run60_sgd_lr1_1000 --epochs 1000 --freeze-cues 500 --out-scale 80
  run hd2_run75_sym_pgf_run60_partner --init-run hd2_run60_sgd_lr1_1000 --epochs 1000 --freeze-cues 500 --partner-probe-weight 1 --out-scale 80
  run hd2_run77_sym_pgf_run60_one_phase --init-run hd2_run60_sgd_lr1_1000 --epochs 1000 --freeze-cues 0 --out-scale 80
  run hd2_run78_sym_pgf_run60_partner_one_phase --init-run hd2_run60_sgd_lr1_1000 --epochs 1000 --freeze-cues 0 --partner-probe-weight 1 --out-scale 80
fi
echo "CHAIN $1 DONE"
