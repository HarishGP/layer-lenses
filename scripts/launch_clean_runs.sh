#!/usr/bin/env bash
# Clean comparable runs (runs 123-130, 2026-10-23): one code path (run_symmetric_gd.py), one schedule (constant 0.1 for 1000 epochs, then a
# cosine to 0.005 over 1500 epochs, 2500 epochs in all), the run 60 design (alpha base 60, readout rate x80) and the run 70 design (base 70,
# x1), plain SGD (--no-symmetrise) and the symmetrised flow, cues frozen for the first 1000 epochs (two-phase) or never (one-phase).
#   chain SGD: 123 (design 60, 2 phases), 124 (design 70, 2 phases), 127 (design 60, 1 phase), 128 (design 70, 1 phase)
#   chain SYM: 125 (design 60, 2 phases), 126 (design 70, 2 phases), 129 (design 60, 1 phase), 130 (design 70, 1 phase)
# Usage: scripts/launch_clean_runs.sh SGD | SYM   (one tmux session each)
cd "$(dirname "$0")/.."
PY=.venv/bin/python; S=results/designed_init/run_symmetric_gd.py; R=results/designed_init
run() { local name=$1; shift; mkdir -p $R/$name; $PY $S --out $name --tag $name "$@" > $R/$name/log.txt 2>&1; }
C="--alpha-json {} --epochs 2500 --warm-epochs 1000"
if [ "$1" = "SGD" ]; then
  run hd2_run123_clean_sgd_d60_2ph $C --alpha-base 60 --freeze-cues 1000 --no-symmetrise
  run hd2_run124_clean_sgd_d70_2ph $C --alpha-base 70 --freeze-cues 1000 --no-symmetrise
  run hd2_run127_clean_sgd_d60_1ph $C --alpha-base 60 --freeze-cues 0 --no-symmetrise
  run hd2_run128_clean_sgd_d70_1ph $C --alpha-base 70 --freeze-cues 0 --no-symmetrise
else
  run hd2_run125_clean_sym_d60_2ph $C --alpha-base 60 --freeze-cues 1000
  run hd2_run126_clean_sym_d70_2ph $C --alpha-base 70 --freeze-cues 1000
  run hd2_run129_clean_sym_d60_1ph $C --alpha-base 60 --freeze-cues 0
  run hd2_run130_clean_sym_d70_1ph $C --alpha-base 70 --freeze-cues 0
fi
echo "CHAIN $1 DONE"
