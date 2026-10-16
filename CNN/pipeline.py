"""
pipeline.py
===========
Runs the full retraining round end to end:

  0. Data prep: prep.py, the 5 leave-one-facility-out splits and the
     chronological split (2020-01 cutoff), for both components.
  1. Clarifier, leave-one-facility-out, with augmentation (7 architectures x 5 facilities)
  2. Clarifier, chronological, with augmentation
  3. Clarifier, chronological, without augmentation
  4. Aerobic zone hyperparameter tuning (6 architectures, Atlantis held out,
     since NoordelikeWerke has no aerobic zones)
  5. Aerobic zone, chronological, with augmentation
  6. Aerobic zone, chronological, without augmentation
  7. Aerobic zone, leave-one-facility-out, with augmentation (x 3 facilities)

Clarifier tuning is not part of the pipeline; it is run separately with
`python tune.py --all`.

The GPU only rests REST_MINUTES after a failed run or a tuning crash. Runs whose results pickle (or tuned hyperparameters)
already exist are skipped, so a crashed pipeline can simply be relaunched. The
results notebooks are executed after the clarifier phases and at the end.

Progress: pipeline_progress.log (one line per event). Verbose output goes to
stdout (see run_pipeline.sh) and to each run's log under results/<experiment>/logs/.

Usage (from CNN/, venv active, or use ./run_pipeline.sh):
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
from run import TRAIN_NOTEBOOKS, ALL_FACILITIES, AEROBIC_FACILITIES, CHRONOLOGICAL_ID

PROGRESS_LOG = Path("pipeline_progress.log").resolve()
REST_MINUTES = 5
RUN_ATTEMPTS = 3          # per training run, in case the GPU driver kills the kernel
TUNE_ATTEMPTS = 10        # per architecture; tune.py resumes from optuna_tuning.db
TUNE_TRIALS = 20
KERNEL = "python3"

DISPLAY = {
    "resnet18": "ResNet-18", "resnet18_scratch": "ResNet-18 (scratch)",
    "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
    "inception_v3": "InceptionNet v3", "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-Tiny",
}
TUNE_ARCHS = ["resnet18", "efficientnet_b0", "densenet121", "resnet50", "convnext_tiny", "inception_v3"]

TRAIN_PHASES = [
    # (title, component, split, results_dir, augment, facilities)
    ("Clarifier - leave-one-facility-out (augmented)", "clarifier", "lofo",
     "../results/clarifier_aug", True, ALL_FACILITIES),
    ("Clarifier - chronological (augmented)", "clarifier", "chronological",
     "../results/clarifier_aug", True, [CHRONOLOGICAL_ID]),
    ("Clarifier - chronological (no augmentation)", "clarifier", "chronological",
     "../results/clarifier_noaug", False, [CHRONOLOGICAL_ID]),
    ("Aerobic zone - chronological (augmented)", "aerobic_zone", "chronological",
     "../results/aerobic_zone_aug", True, [CHRONOLOGICAL_ID]),
    ("Aerobic zone - chronological (no augmentation)", "aerobic_zone", "chronological",
     "../results/aerobic_zone_noaug", False, [CHRONOLOGICAL_ID]),
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
    # 2020-01 (from split_dataset_chronological.py --list) is the earliest
    # cutoff where every facility's test set contains every class that
    # facility has.
    sh([sys.executable, "split_dataset_chronological.py", "--cutoff", "2020-01"])
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
            where = "chronological split" if fac == CHRONOLOGICAL_ID else f"held-out {fac}"
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
    progress(f"PHASE {idx}/{n_phases} done in {fmt_min(time.time() - phase_t0)}")


def tune_phase(idx, n_phases):
    component = "aerobic_zone"
    tuned_json = Path(f"tuned_hyperparams_{component}.json")
    progress()
    progress(f"PHASE {idx}/{n_phases}: Aerobic zone - hyperparameter tuning "
             f"({len(TUNE_ARCHS)} architectures x {TUNE_TRIALS} trials, Atlantis held out)")
    phase_t0 = time.time()
    env = os.environ.copy()
    env["COMPONENT"] = component
    env["DATASET_ROOT"] = "../cnn_dataset_split_facility_test-Atlantis"
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
    progress(f"PIPELINE START - {TOTAL_RUNS} training runs + aerobic tuning")
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
