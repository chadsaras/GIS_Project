#!/usr/bin/env bash
# Run a long job in the background on the server, logging to ~/GIS/logs.
# Usage: scripts/run_bg.sh <name> <command...>
#   e.g. scripts/run_bg.sh export_gsed python scripts/02_export_gsed.py
set -euo pipefail
name="$1"; shift
logdir="$HOME/GIS/logs"
mkdir -p "$logdir"
log="$logdir/$(date +%Y%m%d_%H%M%S)_${name}.log"
nohup "$@" > "$log" 2>&1 &
echo "started pid $! -> $log"
