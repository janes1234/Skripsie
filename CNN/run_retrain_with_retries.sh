#!/usr/bin/env bash
# Runs run.py repeatedly until every requested (architecture, facility)
# combination has a results_*.pkl, or a retry budget is exhausted.
#
# run.py already isolates each combination in its own jupyter kernel
# subprocess, so a GPU driver watchdog crash (see tune.py's notes on
# "Xid 8" / "RC watchdog") only kills that one combination -- run.py itself
# keeps going and just marks it failed in its summary. So a retry only ever
# needs to redo whichever combinations are still missing, which plain
# `python run.py` (without --force) does on its own by skipping anything
# that already has a pkl.
#
# --force is only passed on the FIRST attempt, to overwrite the old
# pre-retune results_*.pkl files. Every retry after that omits it, so
# already-succeeded (fresh) combinations are never wastefully redone.
#
# Usage: same arguments as run.py (minus --force, which this adds itself
# on attempt 1), e.g.:
#   ./run_retrain_with_retries.sh --models resnet18,resnet50,densenet121,inception_v3,efficientnet_b0,convnext_tiny
set -u
cd "$(dirname "$0")"
source ../.venv/bin/activate

MAX_ATTEMPTS=10
attempt=1
FORCE_FLAG="--force"

while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo "[$(date +%H:%M:%S)] === run_retrain_with_retries: attempt $attempt/$MAX_ATTEMPTS (force=${FORCE_FLAG:-no}) ===" >> retrain_progress.log
    python run.py $FORCE_FLAG "$@" >> retrain_progress.log 2>&1
    status=$?
    FORCE_FLAG=""  # only force-overwrite on the very first pass

    if [ "$status" -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] === run_retrain_with_retries: all combinations succeeded ===" >> retrain_progress.log
        exit 0
    fi

    echo "[$(date +%H:%M:%S)] === run_retrain_with_retries: exited $status (some combos failed), retrying missing ones in 15s ===" >> retrain_progress.log
    sleep 15
    attempt=$((attempt + 1))
done

echo "[$(date +%H:%M:%S)] === run_retrain_with_retries: gave up after $MAX_ATTEMPTS attempts ===" >> retrain_progress.log
exit 1
