"""
pipeline_pad.py
===============
Retrains the aerobic zones with letterbox padding (RESIZE_MODE=pad) instead
of stretching each image to the input size, so the aspect ratio of every
image is kept. Runs the same three aerobic zone phases as pipeline.py, with
the same tuned hyperparameters (tuned_hyperparams_aerobic_zone.json), into
separate results folders so the stretched results are kept for comparison:

  1. Aerobic zone, stratified, with augmentation      -> results/aerobic_zone_pad_aug
  2. Aerobic zone, stratified, without augmentation   -> results/aerobic_zone_pad_noaug
  3. Aerobic zone, leave-one-facility-out, augmented  -> results/aerobic_zone_pad_aug

No data prep and no tuning: the existing splits and tuned hyperparameters are
reused. Model weights go to models/pad/. Skips anything that already has a
results pickle, so re-launching after a crash picks up where it stopped.

Live overview: pad_progress.log. Usage (from CNN/):
    nohup ../.venv/bin/python -u pipeline_pad.py >> pad_verbose.log 2>&1 &
"""

import os
import time
from pathlib import Path

os.environ["RESIZE_MODE"] = "pad"  # inherited by every training kernel run.run_one starts

import pipeline
import run
from run import TRAIN_NOTEBOOKS, AEROBIC_FACILITIES, STRATIFIED_ID

pipeline.PROGRESS_LOG = Path("pad_progress.log").resolve()

PHASES = [
    ("Aerobic zone (padded) - stratified 70/20/10 (augmented)", "aerobic_zone", "stratified",
     "../results/aerobic_zone_pad_aug", True, [STRATIFIED_ID]),
    ("Aerobic zone (padded) - stratified 70/20/10 (no augmentation)", "aerobic_zone", "stratified",
     "../results/aerobic_zone_pad_noaug", False, [STRATIFIED_ID]),
    ("Aerobic zone (padded) - leave-one-facility-out (augmented)", "aerobic_zone", "lofo",
     "../results/aerobic_zone_pad_aug", True, AEROBIC_FACILITIES),
]
pipeline.TOTAL_RUNS = sum(len(TRAIN_NOTEBOOKS) * len(p[5]) for p in PHASES)


def main():
    progress = pipeline.progress
    progress()
    progress("=" * 70)
    progress(f"PAD PIPELINE START - {pipeline.TOTAL_RUNS} aerobic zone runs, RESIZE_MODE=pad")
    progress("=" * 70)
    run.check_prerequisites(pipeline.KERNEL)
    t0 = time.time()
    counter = {"done": 0, "failed": []}
    for idx, phase in enumerate(PHASES, 1):
        pipeline.train_phase(idx, len(PHASES), *phase, counter)
    progress()
    progress(f"PAD PIPELINE DONE in {pipeline.fmt_min(time.time() - t0)} - "
             f"{counter['done']}/{pipeline.TOTAL_RUNS} runs have results, {len(counter['failed'])} failed")
    for f in counter["failed"]:
        progress(f"  failed: {f}")


if __name__ == "__main__":
    main()
