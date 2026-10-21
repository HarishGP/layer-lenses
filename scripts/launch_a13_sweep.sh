#!/usr/bin/env bash
# alpha_13 sweep (runs 107-122, 2026-10-21): symmetrised flow with the cues given off-span parts (in-span fraction alpha_13; 1 = the
# runs so far, 0.4 = sqrt(16/100) = a random vector's fraction on the 16 leaf normals), at run 74's settings (run 60's design, two phases
# 500 + 500, readout rate x80) and run 73's (run 70's design, 1000 + 500, rate x1); first at the design's cue bias 1 (alpha_3 = 2), then at
# bias 0.2 (alpha_3 = 1.2); the last run of each chain is the alpha_13 = 1 design with the cosine restarted at phase 2 (--restart-schedule).
# Usage: scripts/launch_a13_sweep.sh R74 | R73   (one tmux session each)
cd "$(dirname "$0")/.."
PY=.venv/bin/python; S=results/designed_init/run_symmetric_gd.py; R=results/designed_init
run() { local name=$1; shift; mkdir -p $R/$name; $PY $S --out $name --tag $name "$@" > $R/$name/log.txt 2>&1; }
if [ "$1" = "R74" ]; then
  B="--alpha-base 60 --epochs 1000 --freeze-cues 500"
  n=107; for a in 0.4 0.6 0.8; do run hd2_run${n}_a13_${a}_run74 $B --alpha-json "{\"alpha_13\": $a}"; n=$((n+1)); done
  n=110; for a in 1.0 0.4 0.6 0.8; do run hd2_run${n}_a13_${a}_b0.2_run74 $B --alpha-json "{\"alpha_13\": $a, \"alpha_3\": 1.2}"; n=$((n+1)); done
  run hd2_run114_a13_1.0_restart_run74 $B --alpha-json '{"alpha_13": 1.0}' --restart-schedule
else
  B="--alpha-base 70 --epochs 1500 --freeze-cues 1000"
  n=115; for a in 0.4 0.6 0.8; do run hd2_run${n}_a13_${a}_run73 $B --alpha-json "{\"alpha_13\": $a}"; n=$((n+1)); done
  n=118; for a in 1.0 0.4 0.6 0.8; do run hd2_run${n}_a13_${a}_b0.2_run73 $B --alpha-json "{\"alpha_13\": $a, \"alpha_3\": 1.2}"; n=$((n+1)); done
  run hd2_run122_a13_1.0_restart_run73 $B --alpha-json '{"alpha_13": 1.0}' --restart-schedule
fi
echo "SWEEP $1 DONE"
