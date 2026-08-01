#!/usr/bin/env bash
# Regenerate every table (paper/tables/*.tex) and figure (paper/figures/*.pdf) in the report from the seeded
# simulation, and check the numbers against paper/results.sha256. Requires the `paper` extra (matplotlib).
# make_data.py replays 300 releases on every core: about 30 CPU-minutes.
set -euo pipefail
cd "$(dirname "$0")/../paper"
mkdir -p figures tables
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python make_data.py > /dev/null
python check_results.py
python figures.py
echo "Tables in paper/tables, figures in paper/figures, numbers in paper/results.json and paper/numbers.json."
