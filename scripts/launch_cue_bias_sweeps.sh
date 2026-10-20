#!/usr/bin/env bash
# Cue-bias sweeps (2026-10-20), symmetrised flow, constant step 0.1, output rate x80, from run 74 (run 60's design):
#   chain P2 (runs 99-103): phase-2 sweep. Start from run 74's epoch-500 state (all probes on their nodes), replace the cues by
#            pole (v + bias u_m) at the init's cue norm, train everything free for 1000 epochs; bias 0.5, 0.25, 0.1, 0.05, 0.02.
#   chain P1 (runs 104-106): phase-1 sweep. Run 74's init with the cue bias set to 0.25, 0.1, 0.05; cues frozen, 1500 epochs.
# Usage: scripts/launch_cue_bias_sweeps.sh P2 | P1
cd "$(dirname "$0")/.."
PY=.venv/bin/python; S=results/designed_init/run_symmetric_gd.py; R=results/designed_init
run() { local name=$1; shift; mkdir -p $R/$name; $PY $S --out $name --tag $name "$@" > $R/$name/log.txt 2>&1; }
if [ "$1" = "P2" ]; then
  n=99; for b in 0.5 0.25 0.1 0.05 0.02; do run hd2_run${n}_cuebias_p2_b${b} --init-run hd2_run74_sym_pgf_run60 --init-epoch 500 --cue-bias $b --constant-lr --epochs 1000 --freeze-cues 0 --out-scale 80; n=$((n+1)); done
else
  n=104; for b in 0.25 0.1 0.05; do run hd2_run${n}_cuebias_p1_b${b} --init-run hd2_run74_sym_pgf_run60 --cue-bias $b --constant-lr --epochs 1500 --freeze-cues 1500 --out-scale 80; n=$((n+1)); done
fi
echo "SWEEP $1 DONE"
