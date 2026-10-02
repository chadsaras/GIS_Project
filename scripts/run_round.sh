#!/usr/bin/env bash
# Run the given variants for one round, then that round's baselines (08). Used to launch rounds in parallel:
#   for r in 0 1 2 3 4; do scripts/run_bg.sh round$r scripts/run_round.sh $r 8 V3 V4; done
# Variants of one round run one after another (they share Stage A/B caches); rounds are independent.
set -euo pipefail
cd "$(dirname "$0")/.."
round="$1"; threads="$2"; shift 2
py="${PYTHON:-$HOME/miniconda3/envs/gisecon/bin/python}"
for v in "$@"; do
  echo "=== $(date +%H:%M:%S) round $round variant $v"
  "$py" -u scripts/07_run_experiment.py --variant "$v" --round "$round" --threads "$threads"
done
echo "=== $(date +%H:%M:%S) round $round baselines"
"$py" -u scripts/08_baselines.py --round "$round"
echo "=== $(date +%H:%M:%S) round $round done"
