#!/usr/bin/env bash
# Alpha sweep A (runs 79-96, 2026-10-19): symmetrised-flow runs from run 60's design with one alpha changed, each two-phase (cues frozen 500 of
# 1000 epochs) and one-phase (1000 epochs, nothing frozen). Usage: scripts/launch_alpha_sweep_A.sh A | B  (one tmux session each).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; S=results/designed_init/run_symmetric_gd.py; R=results/designed_init
run() { local name=$1; shift; mkdir -p $R/$name; $PY $S --out $name --tag $name "$@" > $R/$name/log.txt 2>&1; }
pair() { n2=$1; n1=$2; name=$3; alphas=$4; run hd2_run${n2}_alpha_${name}_2ph --alpha-json "$alphas" --epochs 1000 --freeze-cues 500; run hd2_run${n1}_alpha_${name}_1ph --alpha-json "$alphas" --epochs 1000 --freeze-cues 0; }
C=(0.17677669529663687 0.25 0.35355339059327373 0.5)
if [ "$1" = "A" ]; then
  pair 79 80 a3_1.5    '{"alpha_3": 1.5, "alpha_11": 80}'
  pair 81 82 a3_4      '{"alpha_3": 4.0, "alpha_11": 80}'
  pair 83 84 blocks_x0.5 '{"alpha_4": [0.0884, 0.125, 0.1768, 0.25], "alpha_5": [-0.0884, -0.125, -0.1768, -0.25], "alpha_11": 80}'
  pair 85 86 blocks_x2 '{"alpha_4": [0.3536, 0.5, 0.7071, 1.0], "alpha_5": [-0.3536, -0.5, -0.7071, -1.0], "alpha_11": 80}'
  pair 87 88 a6_0.1    '{"alpha_6": 0.1, "alpha_11": 80}'
else
  pair 89 90 a5_0      '{"alpha_5": 0.0, "alpha_11": 80}'
  pair 91 92 veto_2    '{"alpha_7": -2.0, "alpha_11": 80}'
  pair 93 94 readout_0.4_1.6 '{"alpha_9": 0.4, "alpha_10": 1.6, "alpha_11": 80}'
  pair 95 96 a11_8     '{"alpha_11": 8}'
fi
echo "SWEEP $1 DONE"
