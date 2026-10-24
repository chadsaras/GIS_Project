#!/usr/bin/env bash
# Phase 11.2: rerun the whole pipeline, from the raw downloads to every table and figure in reports/.
#   scripts/run_all.sh [threads per round]          (about a day on gpu77: Stage C dominates)
# $GISECON_DATA_DIR (default ~/GIS/data) must already hold:
#   raw/gee/{gsed_2022,viirs_2022}.tif      Earth Engine export (05_export_gee.py, needs an EE project)
#   raw/ibge/, raw/osm/, raw/ms_buildings/, raw/worldcover/   (01, 03, 03b and 05b download these if missing)
#   raw/tiles/                               Esri caption tiles (11_download_tiles.py, needs an API key)
#   interim/captions_raw_short_v1.jsonl      Kaggle captioning output (17_kaggle_captions_run.py)
# Steps that need accounts or keys (05, 11, Kaggle) are not rerun here.
set -euo pipefail
cd "$(dirname "$0")/.."
threads="${1:-6}"
py="${PYTHON:-$HOME/miniconda3/envs/gisecon/bin/python}"
logdir="${LOG_DIR:-$HOME/GIS/logs}"
mkdir -p "$logdir"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # this checkout's code, not the editable install
step() { echo "=== $(date +%H:%M:%S) $*"; "$py" -u "$@"; }

step scripts/01_build_labels.py
step scripts/02_build_grid.py
step scripts/03_osm_features.py
step scripts/03b_buildings.py
step scripts/04_make_folds.py
step scripts/05b_worldcover_local.py
step scripts/06_build_cells.py
step scripts/10_caption_sample.py
step scripts/12_build_texts.py --captions interim/captions_raw_short_v1.jsonl --clip

# rounds are independent; variants of one round share Stage A/B caches, so they run in order
pids=()
for r in 0 1 2 3 4; do
  PYTHON="$py" scripts/run_round.sh "$r" "$threads" V3 V4 V5 V6 V7 V8 V9 > "$logdir/run_all_round$r.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "=== $(date +%H:%M:%S) all rounds done"

step scripts/09_evaluate.py
step scripts/13_diagnostics.py
step scripts/14_error_analysis.py
step scripts/15_explanations.py

# caption audit: score the committed labels against this run's cleaning decisions
labels=reports/tables/caption_audit_labels_claude.csv
if [ -f "$labels" ]; then
  step scripts/12_build_texts.py --audit-export
  cp "$labels" "${GISECON_DATA_DIR:-$HOME/GIS/data}/interim/audit/audit_labels_claude.csv"
  step scripts/12_build_texts.py --audit-score --labels audit_labels_claude.csv
fi
echo "=== $(date +%H:%M:%S) RUN_ALL_DONE"
