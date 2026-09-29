#!/usr/bin/env bash
# Launches pipeline.py in the background with the venv active. Follow along with:
#   tail -f pipeline_progress.log      (live overview)
#   tail -f pipeline_verbose.log       (everything)
# Extra arguments are passed through to pipeline.py (e.g. --skip-prep).
set -u
cd "$(dirname "$0")"
source ../.venv/bin/activate
nohup python -u pipeline.py "$@" >> pipeline_verbose.log 2>&1 &
echo "pipeline.py started (pid $!)"
