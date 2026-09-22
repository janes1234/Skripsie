#!/usr/bin/env bash
# Runs tune.py, and if it dies (e.g. the GPU driver's RC watchdog killing a
# CUDA kernel -- see the "Xid 8" / "GPU is probably locked" note in
# tune_progress.log if this happens), waits a few seconds for the GPU/driver
# to settle and relaunches it. tune.py's Optuna studies are stored in
# optuna_tuning.db (SQLite), so a relaunch resumes each architecture from
# its last completed trial instead of starting over.
#
# Usage: same arguments as tune.py, e.g.:
#   ./run_tuning_with_retries.sh --all --n-trials 20
set -u
cd "$(dirname "$0")"
source ../.venv/bin/activate

MAX_ATTEMPTS=50
attempt=1

while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo "[$(date +%H:%M:%S)] === run_tuning_with_retries: attempt $attempt/$MAX_ATTEMPTS ===" >> tune_progress.log
    python -u tune.py "$@" >> tune_progress.log 2>&1
    status=$?

    if [ "$status" -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] === run_tuning_with_retries: tune.py exited cleanly, done ===" >> tune_progress.log
        exit 0
    fi

    echo "[$(date +%H:%M:%S)] === run_tuning_with_retries: tune.py exited with code $status, retrying in 15s ===" >> tune_progress.log
    sleep 15
    attempt=$((attempt + 1))
done

echo "[$(date +%H:%M:%S)] === run_tuning_with_retries: gave up after $MAX_ATTEMPTS attempts ===" >> tune_progress.log
exit 1
