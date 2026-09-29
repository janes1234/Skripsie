"""
pipeline.py
===========
Runs the full retraining round end to end, in this order:

  0. Data prep: prep.py (clarifiers from relabel/, aerobic zones from Labels/),
     the 5 leave-one-facility-out splits, and the stratified 70/20/10 split.
  1. Clarifier, leave-one-facility-out, with augmentation (7 architectures x 5 facilities)
  2. Clarifier, stratified 70/20/10, with augmentation     (7 architectures)
  3. Clarifier, stratified 70/20/10, without augmentation  (7 architectures)
  4. Aerobic zone hyperparameter tuning  (6 architectures, NoordelikeWerke held
     out like the clarifier tuning -- it has no aerobic zones, so train/val come
     from all three aerobic facilities)
  5. Aerobic zone, stratified, with augmentation
  6. Aerobic zone, stratified, without augmentation
  7. Aerobic zone, leave-one-facility-out, with augmentation (x 3 facilities)

With vs without augmentation is only compared on the stratified split;
leave-one-facility-out is run once, with augmentation.

The GPU rests REST_MINUTES after every architecture (all of its folds in that
phase) and after each architecture's tuning. Anything whose results pickle (or
tuned hyperparameters) already exists is skipped, so re-launching after a
crash picks up where it stopped. After the clarifier phases and at the end,
results_clarifier.ipynb / results_aerobic_zone.ipynb are executed in place so
their outputs are always up to date.

Live overview: pipeline_progress.log (one line per event, nothing else).
Everything verbose goes to stdout (see run_pipeline.sh) and to each run's own
log under results/<experiment>/logs/.

Usage (from CNN/, venv active -- or just use ./run_pipeline.sh):
    python -u pipeline.py
    python -u pipeline.py --prep-only
    python -u pipeline.py --skip-prep
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import run
from run import TRAIN_NOTEBOOKS, ALL_FACILITIES, AEROBIC_FACILITIES, STRATIFIED_ID

PROGRESS_LOG = Path("pipeline_progress.log").resolve()
REST_MINUTES = 5
RUN_ATTEMPTS = 3          # per training run, in case the GPU driver kills a kernel
TUNE_ATTEMPTS = 10        # per architecture; tune.py resumes from optuna_tuning.db
TUNE_TRIALS = 20
KERNEL = "python3"

DISPLAY = {
    "resnet18": "ResNet-18", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
    "inception_v3": "InceptionNet v3", "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-Tiny", "alexnet": "AlexNet",
}
TUNE_ARCHS = ["resnet18", "efficientnet_b0", "densenet121", "resnet50", "convnext_tiny", "inception_v3"]

TRAIN_PHASES = [
    # (title, component, split, results_dir, augment, facilities)
    ("Clarifier - leave-one-facility-out (augmented)", "clarifier", "lofo",
     "../results/clarifier_aug", True, ALL_FACILITIES),
    ("Clarifier - stratified 70/20/10 (augmented)", "clarifier", "stratified",
     "../results/clarifier_aug", True, [STRATIFIED_ID]),
    ("Clarifier - stratified 70/20/10 (no augmentation)", "clarifier", "stratified",
     "../results/clarifier_noaug", False, [STRATIFIED_ID]),
    ("Aerobic zone - stratified 70/20/10 (augmented)", "aerobic_zone", "stratified",
     "../results/aerobic_zone_aug", True, [STRATIFIED_ID]),
    ("Aerobic zone - stratified 70/20/10 (no augmentation)", "aerobic_zone", "stratified",
     "../results/aerobic_zone_noaug", False, [STRATIFIED_ID]),
    ("Aerobic zone - leave-one-facility-out (augmented)", "aerobic_zone", "lofo",
     "../results/aerobic_zone_aug", True, AEROBIC_FACILITIES),
]
TOTAL_RUNS = sum(len(TRAIN_NOTEBOOKS) * len(p[5]) for p in TRAIN_PHASES)


def progress(msg=""):
    line = f"[{time.strftime('%Y-%m-%d %H:%M')}] {msg}" if msg else ""
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def fmt_min(seconds):
    m = seconds / 60
    return f"{m:.1f} min" if m < 90 else f"{m / 60:.1f} h"


def rest(reason):
    progress(f"    resting GPU {REST_MINUTES} min ({reason})")
    time.sleep(REST_MINUTES * 60)


def sh(cmd, env=None):
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        progress(f"!! command failed ({proc.returncode}): {' '.join(cmd)}")
        sys.exit(1)


def data_prep():
    progress("DATA PREP - rebuilding cnn_dataset and all splits")
    t0 = time.time()
    sh([sys.executable, "prep.py"])
    for fac in ALL_FACILITIES:
        sh([sys.executable, "split_dataset_facility.py", "--test", fac])
    sh([sys.executable, "split_dataset.py", "--output", run.STRATIFIED_ROOT])
    progress(f"DATA PREP done in {fmt_min(time.time() - t0)}")


def train_phase(idx, n_phases, title, component, split, results_dir, augment, facilities, counter):
    n_runs = len(TRAIN_NOTEBOOKS) * len(facilities)
    progress()
    progress(f"PHASE {idx}/{n_phases}: {title} - {n_runs} runs -> {results_dir}")
    phase_t0 = time.time()
    for save_name, notebook in TRAIN_NOTEBOOKS.items():
        name = DISPLAY[save_name]
        todo = [f for f in facilities
                if not run.results_pkl_path(component, save_name, f, results_dir).exists()]
        done_already = len(facilities) - len(todo)
        counter["done"] += done_already
        if not todo:
            progress(f"  {name}: already done, skipping")
            continue
        progress(f"  {name} training ({len(todo)} run{'s' if len(todo) > 1 else ''})")
        arch_t0 = time.time()
        for fac in todo:
            where = "stratified split" if fac == STRATIFIED_ID else f"held-out {fac}"
            for attempt in range(1, RUN_ATTEMPTS + 1):
                t0 = time.time()
                ok, msg = run.run_one(component, save_name, notebook, fac, KERNEL, False,
                                      Path(results_dir), augment)
                print(f"    {name} / {where}: {msg}", flush=True)
                if ok:
                    counter["done"] += 1
                    progress(f"    {name} - {where}: done in {fmt_min(time.time() - t0)}"
                             f"   [{counter['done']}/{TOTAL_RUNS} runs overall]")
                    break
                progress(f"    {name} - {where}: FAILED (attempt {attempt}/{RUN_ATTEMPTS}) "
                         f"after {fmt_min(time.time() - t0)}")
                if attempt < RUN_ATTEMPTS:
                    rest("after a failed run")
            else:
                counter["failed"].append(f"{title}: {name} / {where}")
        progress(f"  {name} finished in {fmt_min(time.time() - arch_t0)}")
        rest(f"after {name}")
    progress(f"PHASE {idx}/{n_phases} done in {fmt_min(time.time() - phase_t0)}")


def tune_phase(idx, n_phases):
    component = "aerobic_zone"
    tuned_json = Path(f"tuned_hyperparams_{component}.json")
    progress()
    progress(f"PHASE {idx}/{n_phases}: Aerobic zone - hyperparameter tuning "
             f"({len(TUNE_ARCHS)} architectures x {TUNE_TRIALS} trials, NoordelikeWerke held out)")
    phase_t0 = time.time()
    env = os.environ.copy()
    env["COMPONENT"] = component
    env["DATASET_ROOT"] = "../cnn_dataset_split_facility_test-NoordelikeWerke"
    env["AUGMENT"] = "1"
    for arch in TUNE_ARCHS:
        name = DISPLAY[arch]
        if tuned_json.exists() and arch in json.loads(tuned_json.read_text()):
            progress(f"  {name}: already tuned, skipping")
            continue
        progress(f"  {name} tuning")
        t0 = time.time()
        for attempt in range(1, TUNE_ATTEMPTS + 1):
            with open(f"tune_progress_{component}.log", "a") as log_f:
                proc = subprocess.run(
                    [sys.executable, "-u", "tune.py", "--arch", arch, "--n-trials", str(TUNE_TRIALS),
                     "--progress-log", str(PROGRESS_LOG)],
                    env=env, stdout=log_f, stderr=subprocess.STDOUT)
            if proc.returncode == 0:
                break
            progress(f"    {name} tuning crashed (attempt {attempt}/{TUNE_ATTEMPTS}), "
                     f"resuming after a rest")
            rest("after a crash")
        else:
            progress(f"!! {name} tuning failed {TUNE_ATTEMPTS} times - stopping, aerobic "
                     f"training needs tuned hyperparameters for every architecture")
            sys.exit(1)
        progress(f"  {name} tuned in {fmt_min(time.time() - t0)}")
        rest(f"after tuning {name}")
    progress(f"PHASE {idx}/{n_phases} done in {fmt_min(time.time() - phase_t0)}")


def execute_results_notebook(nb):
    if not Path(nb).exists():
        progress(f"  (skipped {nb}: not found)")
        return
    proc = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                           f"--ExecutePreprocessor.kernel_name={KERNEL}",
                           "--ExecutePreprocessor.timeout=1800", nb])
    progress(f"  {nb} {'updated' if proc.returncode == 0 else 'FAILED to execute'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prep-only", action="store_true")
    parser.add_argument("--skip-prep", action="store_true")
    args = parser.parse_args()

    progress()
    progress("=" * 70)
    progress(f"PIPELINE START - {TOTAL_RUNS} training runs + aerobic tuning, "
             f"{REST_MINUTES} min GPU rest after each architecture")
    progress("=" * 70)
    if not args.skip_prep:
        data_prep()
    if args.prep_only:
        return

    run.check_prerequisites(KERNEL)
    t0 = time.time()
    counter = {"done": 0, "failed": []}
    n_phases = len(TRAIN_PHASES) + 1

    clarifier_phases = [p for p in TRAIN_PHASES if p[1] == "clarifier"]
    aerobic_phases = [p for p in TRAIN_PHASES if p[1] == "aerobic_zone"]
    idx = 0
    for phase in clarifier_phases:
        idx += 1
        train_phase(idx, n_phases, *phase, counter)
    progress("Updating results_clarifier.ipynb")
    execute_results_notebook("results_clarifier.ipynb")

    idx += 1
    tune_phase(idx, n_phases)

    for phase in aerobic_phases:
        idx += 1
        train_phase(idx, n_phases, *phase, counter)
    progress("Updating results notebooks")
    execute_results_notebook("results_clarifier.ipynb")
    execute_results_notebook("results_aerobic_zone.ipynb")

    progress()
    progress(f"PIPELINE DONE in {fmt_min(time.time() - t0)} - "
             f"{counter['done']}/{TOTAL_RUNS} runs have results, {len(counter['failed'])} failed")
    for f in counter["failed"]:
        progress(f"  failed: {f}")


if __name__ == "__main__":
    main()
